#include "qwen_modulo_software_pipeline_pass.h"

#include "Dialect/AveLang/IR/AveLangOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/Support/Process.h>

#include <optional>
#include <functional>
#include <string>

namespace causalflow::avelang::dialect {
namespace {

constexpr llvm::StringLiteral kSoftwarePipelineMode =
    "gfx942_bt64_bv32_software_pipeline";
constexpr llvm::StringLiteral kPipelineRoleAttr =
    "avelang.qwen.software_pipeline.role";
constexpr llvm::StringLiteral kPipelinePartAttr =
    "avelang.qwen.software_pipeline.part";
constexpr llvm::StringLiteral kPipelineStageAttr =
    "avelang.qwen.software_pipeline.stage";
constexpr llvm::StringLiteral kPipelineDistanceAttr =
    "avelang.qwen.software_pipeline.distance";
constexpr llvm::StringLiteral kPipelineSlotAttr =
    "avelang.qwen.software_pipeline.slot";
// These two markers separate the symbolic schedule edge from its physical
// realization.  The loop keeps its four i64 iter_args (so the generic modulo
// schedule, prologue and epilogue remain explicit), while the direct commit
// writes the actual BF16x8 payload to its rotating LDS destination.
constexpr llvm::StringLiteral kLogicalLdsConsumerAttr =
    "avelang.qwen.software_pipeline.logical_lds_consumer";
constexpr llvm::StringLiteral kDirectLdsCommitAttr =
    "avelang.qwen.software_pipeline.direct_lds_commit";

bool softwarePipelineEnabled() {
    return llvm::sys::Process::GetEnv(
               "AVELANG_PERSISTENT_RECURRENCE_LOWERING") ==
           std::optional<std::string>(kSoftwarePipelineMode.str());
}

bool isBarrier(mlir::Operation *op) {
    return mlir::isa<mlir::gpu::BarrierOp>(op);
}

bool isJointWOperand(AMDGPUQwenK64PipelineStageLoadOp stage) {
    for (llvm::StringRef name : {"avelang.qwen.joint_v1.operand",
                                 "avelang.qwen.joint_v2.operand",
                                 "avelang.qwen.joint_v3.operand",
                                 "avelang.qwen.joint_v4.operand",
                                 "avelang.qwen.joint_v5.operand"}) {
        if (auto operand = stage->getAttrOfType<mlir::StringAttr>(name);
            operand && operand.getValue() == "w") {
            return true;
        }
    }
    return false;
}

/// This is intentionally the small, reusable part of a modulo scheduler.  A
/// stage producer is allowed to pull only its local scalar/address recipe
/// across an opaque compute region; the compute region itself is never
/// inspected or reordered.  This is the same separation Triton's pipeline
/// utility uses between dependency hoisting and loop expansion.
bool moveRecipeAfter(mlir::Operation *stage, mlir::Operation *anchor) {
    auto *block = stage->getBlock();
    if (!block || block != anchor->getBlock()) {
        return false;
    }

    llvm::SmallPtrSet<mlir::Operation *, 16> recipe;
    llvm::SmallPtrSet<mlir::Operation *, 16> visiting;
    std::function<bool(mlir::Value)> collect = [&](mlir::Value value) {
        auto *def = value.getDefiningOp();
        if (!def || def->getBlock() != block || !anchor->isBeforeInBlock(def) ||
            !def->isBeforeInBlock(stage)) {
            return true;
        }
        if (!visiting.insert(def).second) {
            return true;
        }
        // The frontend represents scalar values such as `next_start` as a
        // private store/load pair.  A load therefore carries a true ordering
        // dependence on its matching store even though its memref is private;
        // without MemorySSA/alias analysis it must not be hoisted.  Pure
        // arithmetic recipes remain movable, while these stages retain their
        // legal source position inside the scheduler's steady state.
        if (!mlir::isMemoryEffectFree(def)) {
            return false;
        }
        for (mlir::Value operand : def->getOperands()) {
            if (!collect(operand)) {
                return false;
            }
        }
        recipe.insert(def);
        return true;
    };

    for (mlir::Value operand : stage->getOperands()) {
        if (!collect(operand)) {
            return false;
        }
    }

    llvm::SmallVector<mlir::Operation *> ordered;
    for (auto *candidate = anchor->getNextNode(); candidate && candidate != stage;
         candidate = candidate->getNextNode()) {
        if (recipe.contains(candidate)) {
            ordered.push_back(candidate);
        }
    }
    if (ordered.size() != recipe.size()) {
        return false;
    }
    auto *cursor = anchor;
    for (auto *op : ordered) {
        op->moveAfter(cursor);
        cursor = op;
    }
    stage->moveAfter(cursor);
    return true;
}

struct PipelineStages {
    llvm::SmallVector<AMDGPUQwenK64PipelineStageLoadOp> prologueLoads;
    llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> prologueCommits;
    mlir::scf::ForOp loop;
    llvm::SmallVector<AMDGPUQwenK64PipelineStageLoadOp> steadyLoads;
    llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> steadyCommits;
};

std::optional<PipelineStages>
findDistanceOneStages(AMDGPUQwenPersistentRecurrenceOp recurrence) {
    PipelineStages result;
    recurrence.walk([&](AMDGPUQwenK64PipelineStageLoadOp stage) {
        auto loop = stage->getParentOfType<mlir::scf::ForOp>();
        if (!loop || stage->getBlock() != loop.getBody()) {
            result.prologueLoads.push_back(stage);
            return;
        }
        if (!result.loop) {
            result.loop = loop;
        }
        if (result.loop == loop) {
            result.steadyLoads.push_back(stage);
        }
    });
    recurrence.walk([&](AMDGPUQwenK64PipelineStageCommitOp commit) {
        auto loop = commit->getParentOfType<mlir::scf::ForOp>();
        if (!loop || commit->getBlock() != loop.getBody()) {
            result.prologueCommits.push_back(commit);
            return;
        }
        if (result.loop == loop) {
            result.steadyCommits.push_back(commit);
        }
    });
    if (!result.loop || result.prologueLoads.size() != 4 ||
        result.prologueCommits.size() != 4 || result.steadyLoads.size() != 4 ||
        result.steadyCommits.size() != 4) {
        return std::nullopt;
    }
    return result;
}

void tagStage(mlir::Operation *op, llvm::StringRef part, int64_t stage,
              int64_t distance) {
    mlir::Builder builder(op->getContext());
    op->setAttr(kPipelineRoleAttr, builder.getStringAttr("packet_global_to_lds"));
    op->setAttr(kPipelinePartAttr, builder.getStringAttr(part));
    op->setAttr(kPipelineStageAttr, builder.getI64IntegerAttr(stage));
    op->setAttr(kPipelineDistanceAttr, builder.getI64IntegerAttr(distance));
}

void collectTokenTransport(AMDGPUQwenK64PipelineStageLoadOp stage,
                           llvm::SmallPtrSetImpl<mlir::Operation *> &skip) {
    for (mlir::OpOperand &use : stage.getStageToken().getUses()) {
        if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(use.getOwner())) {
            skip.insert(store);
        }
    }
}

void cloneCore(mlir::OpBuilder &builder, mlir::scf::ForOp source,
               mlir::IRMapping &mapping,
               const llvm::SmallPtrSetImpl<mlir::Operation *> &skip,
               llvm::function_ref<void(mlir::Operation &)> beforeClone = {},
               llvm::function_ref<void(mlir::Operation &, mlir::Operation *)>
                   afterClone = {}) {
    for (mlir::Operation &op : source.getBody()->without_terminator()) {
        if (skip.contains(&op)) {
            continue;
        }
        if (beforeClone) {
            beforeClone(op);
        }
        auto *cloned = builder.clone(op, mapping);
        if (afterClone) {
            afterClone(op, cloned);
        }
    }
}

AMDGPUQwenK64PipelineStageCommitOp
cloneCommit(mlir::OpBuilder &builder,
            AMDGPUQwenK64PipelineStageCommitOp source, mlir::Value token,
            llvm::StringRef part, mlir::Value sharedBank = {}) {
    auto commit = AMDGPUQwenK64PipelineStageCommitOp::create(
        builder, source.getLoc(), token,
        sharedBank ? sharedBank : source.getSharedKBank());
    commit->setAttrs(source->getAttrs());
    tagStage(commit, part, /*stage=*/1, /*distance=*/1);
    return commit;
}

class QwenModuloSoftwarePipelinePass
    : public mlir::PassWrapper<QwenModuloSoftwarePipelinePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QwenModuloSoftwarePipelinePass)

    llvm::StringRef getArgument() const final {
        return "qwen-modulo-software-pipeline";
    }

    llvm::StringRef getDescription() const final {
        return "Expand annotated Qwen packet stages into a modulo software pipeline";
    }

    void runOnOperation() override {
        if (!softwarePipelineEnabled()) {
            return;
        }

        bool failed = false;
        getOperation().walk([&](AMDGPUQwenPersistentRecurrenceOp recurrence) {
            if (failed) {
                return;
            }
            if (!recurrence->hasAttr(
                    "avelang.qwen.persistent_recurrence.software_pipeline_planned")) {
                recurrence.emitError("software pipeline requires a matching plan marker");
                failed = true;
                return;
            }
            auto stages = findDistanceOneStages(recurrence);
            if (!stages) {
                // AveLang deliberately unrolls a constexpr one-chunk body.
                // There is no cross-iteration dependence to pipeline in that
                // legal T=64 case; retain the identical R4 prologue/epilogue
                // computation instead of fabricating a fake SSA lookahead.
                llvm::SmallVector<AMDGPUQwenK64PipelineStageLoadOp> loads;
                llvm::SmallVector<AMDGPUQwenK64PipelineStageCommitOp> commits;
                recurrence.walk([&](AMDGPUQwenK64PipelineStageLoadOp op) {
                    loads.push_back(op);
                });
                recurrence.walk([&](AMDGPUQwenK64PipelineStageCommitOp op) {
                    commits.push_back(op);
                });
                if (loads.size() == 8 && commits.size() == 8) {
                    recurrence->setAttr(
                        "avelang.qwen.persistent_recurrence.software_pipeline_expanded",
                        mlir::UnitAttr::get(&getContext()));
                    recurrence->setAttr(
                        "avelang.qwen.persistent_recurrence.software_pipeline_structure",
                        mlir::StringAttr::get(&getContext(),
                            "single_chunk_unrolled:prologue->epilogue"));
                    return;
                }
                recurrence.emitError(
                    "software pipeline requires four prologue and four loop W/K stages");
                failed = true;
                return;
            }

            // The generic scheduling description is one distance-one packet
            // stage. The only target facts are carried by the existing V4
            // producer attributes; pred/V-new/update are opaque core ops.
            for (auto [slot, stage] : llvm::enumerate(stages->prologueLoads)) {
                tagStage(stage, "prologue", /*stage=*/0, /*distance=*/1);
                stage->setAttr(kPipelineSlotAttr,
                               mlir::IntegerAttr::get(mlir::IntegerType::get(
                                   stage.getContext(), 64), slot));
            }
            for (auto [slot, stage] : llvm::enumerate(stages->steadyLoads)) {
                tagStage(stage, "steady_state_issue", /*stage=*/0,
                         /*distance=*/1);
                stage->setAttr(kPipelineSlotAttr,
                               mlir::IntegerAttr::get(mlir::IntegerType::get(
                                   stage.getContext(), 64), slot));
            }

            // Pull all next-W/K issue recipes to the beginning of the stage
            // window. This uses only producer operands, not a Qwen operation
            // list: the first stage supplies the anchor and each following
            // stage is scheduled by its dependency closure.
            auto *issueAnchor = stages->steadyLoads.front().getOperation();
            for (auto stage : llvm::drop_begin(stages->steadyLoads)) {
                if (!moveRecipeAfter(stage.getOperation(), issueAnchor)) {
                    // This is a scheduling decision, not a fallback to the
                    // old recurrence.  The stage still crosses one loop
                    // iteration through the generated iter_arg; only its
                    // intra-iteration issue point is constrained by a
                    // scalar-memory dependence that this pass does not own.
                    stage->setAttr(
                        "avelang.qwen.software_pipeline.issue_decision",
                        mlir::StringAttr::get(&getContext(),
                            "retained_due_to_memory_dependence"));
                    continue;
                }
                stage->setAttr("avelang.qwen.software_pipeline.issue_decision",
                               mlir::StringAttr::get(&getContext(), "hoisted"));
                issueAnchor = stage.getOperation();
            }

            auto loop = stages->loop;

            // The two R4 workgroup banks already have exactly one W and one
            // K packet tile.  A second pair would raise this kernel from
            // 53,248 B to more than 85 KiB LDS, which is not a legal or
            // useful gfx942 workgroup.  Instead, after current-W is consumed
            // by pred, its bank receives next-K; after current-K is consumed
            // by update, its bank receives next-W.  The following iteration
            // flips which physical bank plays each logical role.
            mlir::Value wBank;
            mlir::Value kBank;
            for (auto [slot, stage] : llvm::enumerate(stages->prologueLoads)) {
                auto bank = stages->prologueCommits[slot].getSharedKBank();
                if (isJointWOperand(stage)) {
                    wBank = bank;
                } else {
                    kBank = bank;
                }
            }
            if (!wBank || !kBank || wBank.getType() != kBank.getType()) {
                recurrence.emitError(
                    "rotating LDS pipeline requires matching R4 W and K shared banks");
                failed = true;
                return;
            }

            llvm::SmallPtrSet<mlir::Operation *, 32> skipped;
            for (auto stage : stages->steadyLoads) {
                collectTokenTransport(stage, skipped);
            }
            for (auto commit : stages->steadyCommits) {
                skipped.insert(commit);
                if (auto *wrapper = commit.getStageToken().getDefiningOp()) {
                    if (mlir::isa<mlir::memref::LoadOp>(wrapper)) {
                        skipped.insert(wrapper);
                    }
                }
            }
            if (auto *tailBarrier = stages->steadyCommits.back()->getNextNode();
                isBarrier(tailBarrier)) {
                skipped.insert(tailBarrier);
            }

            // The recurrence's audit and FP32-feedback paths may still read
            // either current shared bank after the lexical pred/update sites.
            // To make bank retirement explicit without pretending that an
            // opaque operation has a narrower memory effect, issue both
            // next-packet producers after the complete current core.  A
            // single workgroup barrier then separates that retiring core from
            // the two direct writes into the rotating slots.
            auto *steadyTail = loop.getBody()->getTerminator();
            AMDGPUQwenK64PipelineStageLoadOp firstWStage;
            for (auto stage : stages->steadyLoads) {
                stage->moveBefore(steadyTail);
                stage->setAttr(
                    "avelang.qwen.software_pipeline.issue_decision",
                    mlir::StringAttr::get(
                        &getContext(),
                        "delayed_until_full_current_packet_core_retires"));
                if (!firstWStage && isJointWOperand(stage)) {
                    firstWStage = stage;
                }
            }
            if (!firstWStage) {
                recurrence.emitError(
                    "rotating LDS pipeline could not identify its tail issue point");
                failed = true;
                return;
            }
            mlir::OpBuilder builder(loop);
            auto loc = loop.getLoc();
            // Preserve the verified R4 prologue commit verbatim.  Besides
            // being the first packet-ring fill, this retains the frontend's
            // existing W/K shared-bank ownership and its barrier.  The first
            // cloned compute iteration then issues packet 1 and becomes the
            // initial value of the generic modulo loop.
            for (auto commit : stages->prologueCommits) {
                tagStage(commit, "prologue_commit", /*stage=*/1,
                         /*distance=*/1);
            }
            mlir::IRMapping prologueMapping;
            prologueMapping.map(loop.getInductionVar(), loop.getLowerBound());
            auto emitRotatingDirectCommit =
                [&](mlir::Value nextWBank, mlir::Value nextKBank) {
                    return [&, nextWBank, nextKBank](mlir::Operation &source,
                                                     mlir::Operation *cloned) {
                        auto sourceStage =
                            mlir::dyn_cast<AMDGPUQwenK64PipelineStageLoadOp>(&source);
                        if (!sourceStage) {
                            return;
                        }
                        std::optional<unsigned> slot;
                        for (auto [index, candidate] :
                             llvm::enumerate(stages->steadyLoads)) {
                            if (candidate == sourceStage) {
                                slot = index;
                                break;
                            }
                        }
                        if (!slot) {
                            return;
                        }
                        auto clonedStage =
                            mlir::cast<AMDGPUQwenK64PipelineStageLoadOp>(cloned);
                        auto direct = cloneCommit(
                            builder, stages->steadyCommits[*slot],
                            clonedStage.getStageToken(),
                            "rotating_lds_direct_commit",
                            isJointWOperand(sourceStage) ? nextWBank : nextKBank);
                        direct->setAttr(kDirectLdsCommitAttr,
                                        mlir::UnitAttr::get(&getContext()));
                    };
                };
            auto emitReuseBarrierBefore = [&](mlir::Operation &source) {
                if (&source == firstWStage.getOperation()) {
                    mlir::gpu::BarrierOp::create(builder, source.getLoc());
                }
            };
            // This first no-private-memory cut retains the verified physical
            // W/K bank identities.  The scheduler still carries its
            // distance-one token/slot recurrence, but direct LDS producers
            // are committed after the complete core.  A true role-rotating
            // variant is gated separately: the late opaque recurrence step
            // still owns a single `phaseStage` whose identity cannot be
            // rewritten by merely remapping the outer packet SSA values.
            cloneCore(builder, loop, prologueMapping, skipped,
                      emitReuseBarrierBefore,
                      emitRotatingDirectCommit(/*nextWBank=*/wBank,
                                               /*nextKBank=*/kBank));

            llvm::SmallVector<mlir::Value> initialTokens;
            initialTokens.reserve(stages->steadyLoads.size());
            for (auto stage : stages->steadyLoads) {
                auto token = prologueMapping.lookupOrNull(stage.getStageToken());
                if (!token) {
                    stage.emitError(
                        "software pipeline lost a prologue-issued stage token");
                    failed = true;
                    return;
                }
                initialTokens.push_back(token);
            }

            auto steadyLower = mlir::arith::AddIOp::create(
                builder, loc, loop.getLowerBound(), loop.getStep());
            auto steadyUpper = mlir::arith::SubIOp::create(
                builder, loc, loop.getUpperBound(), loop.getStep());
            auto steady = mlir::scf::ForOp::create(
                builder, loc, steadyLower, steadyUpper, loop.getStep(),
                initialTokens);
            if (auto trailing = mlir::dyn_cast<mlir::scf::YieldOp>(
                    &steady.getBody()->back())) {
                trailing.erase();
            }
            steady->setAttr(kPipelineRoleAttr,
                            builder.getStringAttr("distance_one_modulo_loop"));
            steady->setAttr(kPipelinePartAttr, builder.getStringAttr("steady_state"));
            steady->setAttr(kPipelineStageAttr, builder.getI64IntegerAttr(1));
            steady->setAttr(kPipelineDistanceAttr, builder.getI64IntegerAttr(1));

            builder.setInsertionPointToStart(steady.getBody());
            for (auto [index, commit] : llvm::enumerate(stages->steadyCommits)) {
                auto token = steady.getBody()->getArgument(index + 1);
                auto logical = cloneCommit(builder, commit, token,
                                           "steady_state_logical_commit");
                logical->setAttr(kLogicalLdsConsumerAttr,
                                 mlir::UnitAttr::get(&getContext()));
            }
            mlir::gpu::BarrierOp::create(builder, loc);

            mlir::IRMapping steadyMapping;
            steadyMapping.map(loop.getInductionVar(), steady.getInductionVar());
            cloneCore(builder, loop, steadyMapping, skipped,
                      emitReuseBarrierBefore,
                      emitRotatingDirectCommit(/*nextWBank=*/wBank,
                                               /*nextKBank=*/kBank));
            llvm::SmallVector<mlir::Value> nextTokens;
            nextTokens.reserve(stages->steadyLoads.size());
            for (auto stage : stages->steadyLoads) {
                auto token = steadyMapping.lookupOrNull(stage.getStageToken());
                if (!token) {
                    stage.emitError("software pipeline lost a stage token while cloning");
                    failed = true;
                    return;
                }
                nextTokens.push_back(token);
            }
            mlir::scf::YieldOp::create(builder, loc, nextTokens);

            // The peeled epilogue consumes the final packet ring entry but
            // does not issue another load. This removes the old clamp-to-last
            // fake iteration and makes the drain visible in MLIR/LLVM/MIR.
            builder.setInsertionPointAfter(steady);
            auto lastIV = mlir::arith::SubIOp::create(
                builder, loc, loop.getUpperBound(), loop.getStep());
            for (auto [index, commit] : llvm::enumerate(stages->steadyCommits)) {
                auto logical = cloneCommit(builder, commit, steady.getResult(index),
                                           "epilogue_logical_commit");
                logical->setAttr(kLogicalLdsConsumerAttr,
                                 mlir::UnitAttr::get(&getContext()));
            }
            mlir::gpu::BarrierOp::create(builder, loc);
            // A drain has no successor iteration.  In particular, do not
            // clone the original distance-one issue operations here: leaving
            // them in the epilogue both issued an unused packet and gave the
            // late packet lowering an unconsumed stage token.  The steady
            // state is the sole producer of the next ring entry.
            llvm::SmallPtrSet<mlir::Operation *, 32> epilogueSkipped;
            for (auto *op : skipped) {
                epilogueSkipped.insert(op);
            }
            for (auto stage : stages->steadyLoads) {
                epilogueSkipped.insert(stage.getOperation());
            }
            mlir::IRMapping epilogueMapping;
            epilogueMapping.map(loop.getInductionVar(), lastIV);
            cloneCore(builder, loop, epilogueMapping, epilogueSkipped);

            loop.erase();
            recurrence->setAttr(
                "avelang.qwen.persistent_recurrence.software_pipeline_expanded",
                mlir::UnitAttr::get(&getContext()));
            recurrence->setAttr(
                "avelang.qwen.persistent_recurrence.software_pipeline_structure",
                builder.getStringAttr(
                    "prologue(issue+consume)->steady(iter_args,direct_lds,"
                    "distance=1)->epilogue"));
        });
        if (failed) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createQwenModuloSoftwarePipelinePass() {
    return std::make_unique<QwenModuloSoftwarePipelinePass>();
}

} // namespace causalflow::avelang::dialect
