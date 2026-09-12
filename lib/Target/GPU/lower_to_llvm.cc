#include "lower_to_llvm.h"
#include "Dialect/AveLang/IR/AveLangDialect.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Dialect/AveLang/Transforms/allocation_op_interface_impl.h"
#include "Dialect/AveLang/Transforms/bounded_packet_schedule_pass.h"
#include "Dialect/AveLang/Transforms/hoist_alloca_pass.h"
#include "Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.h"
#include "Dialect/AveLang/Transforms/lower_gpuop_to_intrinsics_pass.h"
#include "Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.h"
#include "Dialect/AveLang/Transforms/lower_qwen_gdn_recurrence_step_pass.h"
#include "Dialect/AveLang/Transforms/lower_qwen_k64_pipeline_stage_pass.h"
#include "Dialect/AveLang/Transforms/lower_qwen_kfrag_lds_pass.h"
#include "Dialect/AveLang/Transforms/qwen_kfrag_producer_consumer_rewrite_pass.h"
#include "Dialect/AveLang/Transforms/qwen_modulo_software_pipeline_pass.h"
#include "Dialect/AveLang/Transforms/qwen_persistent_recurrence_pass.h"
#include "IR/builtin_module.h"
#include "IR/ir_context.h"
#include "avelang/config.h"
#include "gpu_backend.h"
#include "gpu_passes.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include <optional>

namespace mlir::func {
class FuncDialect;
}

namespace causalflow::avelang::target::gpu {

using namespace mlir;

namespace {

void writeQwenAuditSnapshot(llvm::StringRef environment, llvm::StringRef tag,
                            mlir::Operation *op, llvm::StringRef phase) {
    const auto dumpDir = llvm::sys::Process::GetEnv(environment);
    if (!dumpDir) {
        return;
    }
    std::error_code ec = llvm::sys::fs::create_directories(*dumpDir);
    if (ec) {
        llvm::errs() << "[" << tag << "] cannot create " << *dumpDir << ": "
                     << ec.message() << "\n";
        return;
    }
    llvm::SmallString<256> path(*dumpDir);
    llvm::sys::path::append(path, phase + ".mlir");
    std::error_code writeEc;
    llvm::raw_fd_ostream stream(path, writeEc);
    if (writeEc) {
        llvm::errs() << "[" << tag << "] cannot write " << path << ": "
                     << writeEc.message() << "\n";
        return;
    }
    op->print(stream);
}

void writeQwenAuditLLVM(llvm::StringRef environment, llvm::StringRef tag,
                        const llvm::Module &module, llvm::StringRef phase) {
    const auto dumpDir = llvm::sys::Process::GetEnv(environment);
    if (!dumpDir) {
        return;
    }
    std::error_code ec = llvm::sys::fs::create_directories(*dumpDir);
    if (ec) {
        llvm::errs() << "[" << tag << "] cannot create " << *dumpDir << ": "
                     << ec.message() << "\n";
        return;
    }
    llvm::SmallString<256> path(*dumpDir);
    llvm::sys::path::append(path, phase + ".ll");
    std::error_code writeEc;
    llvm::raw_fd_ostream stream(path, writeEc);
    if (writeEc) {
        llvm::errs() << "[" << tag << "] cannot write " << path << ": "
                     << writeEc.message() << "\n";
        return;
    }
    module.print(stream, nullptr);
}

// This is an audit-only snapshot hook for the Qwen K-fragment A/B experiment.
// It is inert unless AVELANG_QWEN_KFRAG_AB_DUMP_DIR is set.
void writeQwenKFragAuditSnapshot(mlir::Operation *op, llvm::StringRef phase) {
    writeQwenAuditSnapshot("AVELANG_QWEN_KFRAG_AB_DUMP_DIR", "qwen-kfrag-ab",
                           op, phase);
}

void writeQwenKFragAuditLLVM(const llvm::Module &module,
                             llvm::StringRef phase) {
    writeQwenAuditLLVM("AVELANG_QWEN_KFRAG_AB_DUMP_DIR", "qwen-kfrag-ab",
                       module, phase);
}

void writeQwenPersistentRecurrenceSnapshot(mlir::Operation *op,
                                            llvm::StringRef phase) {
    writeQwenAuditSnapshot("AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR",
                           "qwen-persistent-recurrence", op, phase);
}

void writeQwenPersistentRecurrenceLLVM(const llvm::Module &module,
                                       llvm::StringRef phase) {
    writeQwenAuditLLVM("AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR",
                       "qwen-persistent-recurrence", module, phase);
}

bool isQwenKFragConvergenceAuditEnabled() {
    return llvm::sys::Process::GetEnv("AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT") ==
           std::optional<std::string>("1");
}

std::string qwenKFragAuditSafePassName(llvm::StringRef passName) {
    std::string safe;
    safe.reserve(passName.size());
    for (char ch : passName) {
        safe.push_back(llvm::isAlnum(ch) ? ch : '_');
    }
    return safe.empty() ? "unnamed" : safe;
}

class QwenKFragAuditSnapshotPass
    : public PassWrapper<QwenKFragAuditSnapshotPass, OperationPass<ModuleOp>> {
  public:
    explicit QwenKFragAuditSnapshotPass(std::string phase)
        : phase_(std::move(phase)) {}

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QwenKFragAuditSnapshotPass)

    StringRef getArgument() const override { return "qwen-kfrag-ab-snapshot"; }

    StringRef getDescription() const override {
        return "Write an audit-only Qwen K-fragment MLIR snapshot";
    }

    void runOnOperation() override {
        writeQwenKFragAuditSnapshot(getOperation(), phase_);
    }

  private:
    std::string phase_;
};

std::unique_ptr<Pass> createQwenKFragAuditSnapshotPass(llvm::StringRef phase) {
    return std::make_unique<QwenKFragAuditSnapshotPass>(phase.str());
}

class QwenPersistentRecurrenceSnapshotPass
    : public PassWrapper<QwenPersistentRecurrenceSnapshotPass,
                         OperationPass<ModuleOp>> {
  public:
    explicit QwenPersistentRecurrenceSnapshotPass(std::string phase)
        : phase_(std::move(phase)) {}

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        QwenPersistentRecurrenceSnapshotPass)

    StringRef getArgument() const override {
        return "qwen-persistent-recurrence-snapshot";
    }

    StringRef getDescription() const override {
        return "Write an audit-only Qwen persistent recurrence MLIR snapshot";
    }

    void runOnOperation() override {
        writeQwenPersistentRecurrenceSnapshot(getOperation(), phase_);
    }

  private:
    std::string phase_;
};

std::unique_ptr<Pass>
createQwenPersistentRecurrenceSnapshotPass(llvm::StringRef phase) {
    return std::make_unique<QwenPersistentRecurrenceSnapshotPass>(phase.str());
}

class EraseAveLangEndLifetimePass
    : public PassWrapper<EraseAveLangEndLifetimePass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseAveLangEndLifetimePass)

    StringRef getArgument() const override {
        return "erase-avelang-end-lifetime";
    }

    StringRef getDescription() const override {
        return "Erase AveLang lifetime markers after they have survived "
               "memref lowering and GPU outlining.";
    }

    void runOnOperation() override {
        SmallVector<causalflow::avelang::dialect::EndLifetimeOp> ops;
        getOperation()->walk(
            [&](causalflow::avelang::dialect::EndLifetimeOp op) {
                ops.push_back(op);
            });
        for (auto op : ops) {
            op.erase();
        }
    }
};

std::unique_ptr<Pass> createEraseAveLangEndLifetimePass() {
    return std::make_unique<EraseAveLangEndLifetimePass>();
}

} // namespace

class LowerToLLVM::Impl {
  public:
    explicit Impl(causalflow::avelang::ir::IRContext *ir_context)
        : ir_context_(ir_context) {}

    std::unique_ptr<::llvm::Module>
    compile(mlir::ModuleOp module, ::llvm::LLVMContext &llvmContext,
            const GPUCompilationOptions &options) {
        // Set up target machine to get correct data layout first
        std::string targetTriple = options.triple.str();
        llvm::Triple triple(targetTriple);

        std::string error;
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget(triple, error);
        if (!target) {
            llvm::errs() << "Failed to lookup target for triple "
                         << targetTriple << ": " << error << "\n";
            return nullptr;
        }

        llvm::TargetOptions targetOptions;
        auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
            target->createTargetMachine(triple, options.chipset.str(), "",
                                        targetOptions, llvm::Reloc::PIC_));
        if (!targetMachine) {
            llvm::errs() << "Failed to create target machine for triple "
                         << targetTriple << "\n";
            return nullptr;
        }

        // Set the data layout on the MLIR module
        std::string dataLayoutStr;

        auto dataLayout = targetMachine->createDataLayout();
        dataLayoutStr = dataLayout.getStringRepresentation();

        module->setAttr(
            "llvm.data_layout",
            ::mlir::StringAttr::get(module.getContext(), dataLayoutStr));
        module->setAttr(
            "llvm.target_triple",
            ::mlir::StringAttr::get(module.getContext(), targetTriple));

        {
            ::mlir::DialectRegistry registry;
            ::mlir::func::registerAllExtensions(registry);
            ::mlir::arith::registerConvertArithToLLVMInterface(registry);
            ::mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
            ::mlir::registerConvertComplexToLLVMInterface(registry);
            ::mlir::registerConvertFuncToLLVMInterface(registry);
            ::mlir::index::registerConvertIndexToLLVMInterface(registry);
            ::mlir::LLVM::registerInlinerInterface(registry);
            ::mlir::NVVM::registerInlinerInterface(registry);
            ::mlir::registerConvertMathToLLVMInterface(registry);
            ::mlir::registerConvertMemRefToLLVMInterface(registry);
            ::mlir::registerConvertNVVMToLLVMInterface(registry);
            ::mlir::ub::registerConvertUBToLLVMInterface(registry);
            ::mlir::vector::registerConvertVectorToLLVMInterface(registry);
            causalflow::avelang::dialect::
                registerAllocationOpInterfaceExternalModels(registry);
            ::mlir::memref::registerAllocationOpInterfaceExternalModels(
                registry);
            auto *context = ir_context_->GetMLIRContext();
            context->appendDialectRegistry(registry);
            context->loadDialect<::mlir::affine::AffineDialect>();
            context->loadDialect<::mlir::func::FuncDialect>();
            context->loadDialect<::mlir::vector::VectorDialect>();
            context->loadDialect<::mlir::gpu::GPUDialect>();
            context->loadDialect<::mlir::arith::ArithDialect>();
            context->loadDialect<::mlir::math::MathDialect>();
            context->loadDialect<::mlir::bufferization::BufferizationDialect>();
            context->loadDialect<::mlir::scf::SCFDialect>();
            context->loadDialect<::mlir::ub::UBDialect>();
            context->loadDialect<::mlir::LLVM::LLVMDialect>();
            context->loadDialect<::mlir::memref::MemRefDialect>();
            context
                ->loadDialect<causalflow::avelang::dialect::AveLangDialect>();
        }

        // Declare intrinsic modules to make intrinsic functions available
        // This is crucial for the GPUOp lowering pass to find intrinsic
        // functions
        if (targetTriple.find("nvptx") != std::string::npos) {
            // NVVM target
            auto nvvmModule =
                causalflow::avelang::ir::CreateNVVMIntrinsicModule();
            nvvmModule->DeclareModules(module);
        } else if (targetTriple.find("amdgcn") != std::string::npos) {
            // AMDGPU target
            auto amdgpuModule =
                causalflow::avelang::ir::CreateAMDGPUIntrinsicModule();
            amdgpuModule->DeclareModules(module);
        }

        PassManager pm(ir_context_->GetMLIRContext());

        pm.addPass(causalflow::avelang::dialect::
                       createLowerAveLangGPUToIntrinsicsPass());

        pm.addPass(::mlir::createInlinerPass());
        pm.addPass(::mlir::createCanonicalizerPass());
        pm.addPass(::mlir::createCSEPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::createHoistAllocaPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::bufferization::createBufferHoistingPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::bufferization::createBufferLoopHoistingPass());

        // Lower ave memref types and dialect ops to memref dialect
        pm.addPass(
            causalflow::avelang::dialect::createLowerAveLangToMemRefPass());
        // R0 first forms a region around the exact B0 full device-side loop.
        // At this point pred, the BF16 boundary and block-dot are still one
        // semantic unit; target planning must observe this before either the
        // recurrence or block-dot late lowering can expand it.
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::
                createFormQwenPersistentRecurrencePass());
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "persistent_recurrence_formed"));
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::createPlanQwenPersistentRecurrencePass());
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "post_recurrence_joint_planner"));
        // The software-pipeline expander runs while the complete recurrence
        // still owns its outer scf.for and opaque W/K producer/commit tokens.
        // It produces prologue/steady/epilogue structure before any Qwen
        // arithmetic or stage-token late lowering expands the body.
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::createQwenModuloSoftwarePipelinePass());
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "post_software_pipeline_scheduler"));
        // The persistent fragment op is still present here.  The generic and
        // direct-LDS runs must hash this identical snapshot before branching.
        pm.addPass(createQwenKFragAuditSnapshotPass("pre_kfrag_branch"));
        pm.addPass(causalflow::avelang::dialect::
                       createQwenKFragProducerConsumerRewritePass());
        pm.addPass(createQwenKFragAuditSnapshotPass("post_kfrag_rewrite"));
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::createHoistAllocaPass());
        // B1 keeps a BT64 recurrence step semantic through AveLang-to-memref.
        // Its stream32 expansion must happen before block-dot lowering and
        // intrinsic implementation linking, while typed BF16 operands are
        // still visible as workgroup memrefs.
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::
                createLowerQwenGdnRecurrenceStepPass());
        pm.addPass(
            createQwenKFragAuditSnapshotPass("post_recurrence_step_lowering"));
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "pre_legacy_b0_lowering"));
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::
                createLowerQwenPersistentRecurrencePass());
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "post_legacy_b0_lowering"));
        // Keep the typed block-dot semantic through AveLang-to-memref, then
        // expand it while the MFMA implementation linker can still see the
        // newly generated calls. Lowering after SymbolDCE would leave the
        // outlined GPU module with calls to an erased inline intrinsic.
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::createLowerQwenBlockDotPass());
        pm.addPass(createQwenKFragAuditSnapshotPass("post_block_dot_lowering"));
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "post_block_dot_lowering"));
        // Z9S is a compiler-only, structurally guarded load/commit fission.
        // It is inert unless explicitly enabled and runs while raw-buffer
        // packet values and their existing consumer loops remain visible.
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::createBoundedPacketSchedulePass());
        pm.addPass(createQwenKFragAuditSnapshotPass(
            "post_bounded_packet_schedule"));
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "post_bounded_packet_schedule"));
        pm.addPass(createLinkIntrinsicImplementationPass());
        pm.addPass(::mlir::createCanonicalizerPass());
        pm.addPass(::mlir::createCSEPass());

        // Add GPU outlining pass first to move kGlobalKernel functions to GPU
        // modules
        pm.addPass(createGpuOutliningPass());
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "post_gpu_outlining"));
        if (isQwenKFragConvergenceAuditEnabled()) {
            pm.addPass(createQwenKFragAuditSnapshotPass("post_gpu_outlining"));
            pm.addPass(createQwenKFragAuditSnapshotPass(
                "pre_block_dot_operand_materialization"));
        }
        // P2 consumes the internal generic block-dot operand plan only after
        // outlining.  This is the semantic-preservation boundary: the
        // logical role/layout/packed-word identity is still visible in the
        // GPU module, immediately before target-specific LDS/MFMA materialization.
        pm.addNestedPass<mlir::gpu::GPUModuleOp>(
            causalflow::avelang::dialect::
                createLowerQwenBlockDotMfmaOperandPass());
        if (isQwenKFragConvergenceAuditEnabled()) {
            pm.addPass(createQwenKFragAuditSnapshotPass(
                "post_block_dot_operand_materialization"));
        }
        pm.addNestedPass<mlir::gpu::GPUModuleOp>(
            ::mlir::createCanonicalizerPass());
        pm.addNestedPass<mlir::gpu::GPUModuleOp>(::mlir::createCSEPass());
        pm.addPass(::mlir::createSymbolDCEPass());
        if (isQwenKFragConvergenceAuditEnabled()) {
            pm.addPass(
                createQwenKFragAuditSnapshotPass("post_outline_cleanup"));
        }

        // Keep ave.end_lifetime alive through AveLang->memref lowering and GPU
        // outlining so the marker can be audited at a later phase. It is erased
        // here because no real LLVM lifetime.end lowering is implemented yet,
        // and the backend conversion pipeline does not legalize AveLang ops.
        pm.addPass(createEraseAveLangEndLifetimePass());
        // The producer/consumer rewrite keeps this guarded op through GPU
        // outlining. Lower it here, after outlining but before the generic
        // AMDGPU conversion pipeline, to avoid vector.load/GEP lowering.
        pm.addNestedPass<mlir::gpu::GPUModuleOp>(
            causalflow::avelang::dialect::createLowerQwenKFragLDSPass());
        // S0 keeps the K64 stage token opaque until this post-outline point.
        // It must not become a generic vector.load/memref.store chain before
        // the distributed load placement has been selected.
        pm.addPass(createQwenKFragAuditSnapshotPass(
            "pre_k64_pipeline_stage_lowering"));
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "pre_joint_v1_stage_lowering"));
        pm.addNestedPass<mlir::gpu::GPUModuleOp>(
            causalflow::avelang::dialect::createLowerQwenK64PipelineStagePass());
        pm.addPass(
            createQwenKFragAuditSnapshotPass("post_kfrag_load_lowering"));
        pm.addPass(createQwenPersistentRecurrenceSnapshotPass(
            "post_joint_v1_stage_lowering"));
        if (isQwenKFragConvergenceAuditEnabled()) {
            pm.addPass(createQwenKFragAuditSnapshotPass("post_late_lowering"));
        }

        // Propagate data layout to GPU modules
        auto propagateDataLayoutPass = [&](::mlir::Operation *op) {
            if (auto gpuModule = dyn_cast<::mlir::gpu::GPUModuleOp>(op)) {
                gpuModule->setAttr("llvm.data_layout",
                                   ::mlir::StringAttr::get(module.getContext(),
                                                           dataLayoutStr));
                gpuModule->setAttr(
                    "llvm.target_triple",
                    ::mlir::StringAttr::get(module.getContext(), targetTriple));
            }
        };
        module.walk(propagateDataLayoutPass);

        // Use backend registry to find appropriate backend for triple
        auto &registry = GPUBackendRegistry::getInstance();
        auto backend = registry.createBackendForTriple(options.triple);
        if (!backend) {
            llvm::errs() << "No GPU backend registered for triple "
                         << targetTriple << "\n";
            return nullptr;
        }

        backend->buildLoweringPipeline(pm, options);
        if (isQwenKFragConvergenceAuditEnabled()) {
            pm.addPass(
                createQwenKFragAuditSnapshotPass("post_amdgpu_mlir_pipeline"));
        }

        pm.addPass(::mlir::createFinalizeMemRefToLLVMConversionPass());

        pm.addPass(::mlir::createCanonicalizerPass());
        pm.addPass(::mlir::createCSEPass());
        pm.addPass(createQwenKFragAuditSnapshotPass("final_mlir"));

        pm.addPass(::mlir::createReconcileUnrealizedCastsPass());

        pm.addPass(::mlir::createCanonicalizerPass());
        pm.addPass(::mlir::createCSEPass());

        // The late GPU/LLVM conversion pipeline can fail after all of the
        // regular audit snapshots have been written.  Keep a narrow,
        // opt-in failure dump so backend experiments can identify the
        // failing pass without changing the normal pipeline or its IR.
        if (llvm::sys::Process::GetEnv("AVELANG_DEBUG_MLIR_PASS_FAILURE")) {
            pm.enableIRPrinting(
                [](Pass *, Operation *) { return false; },
                [](Pass *, Operation *) { return true; }, true, false, true,
                llvm::errs());
        }

        if (failed(pm.run(module))) {
            llvm::errs() << "Pass manager failed for triple " << targetTriple
                         << "\n";
            return nullptr;
        }

        SmallVector<::mlir::gpu::GPUModuleOp> gpuModules;
        module.walk([&](::mlir::gpu::GPUModuleOp gpuModule) {
            gpuModules.push_back(gpuModule);
        });

        if (gpuModules.size() != 1) {
            llvm::errs() << "Expected exactly one GPU module after lowering, "
                         << "found " << gpuModules.size() << "\n";
            return nullptr;
        }

        // Translate MLIR to LLVM IR
        auto llvmModule = translateModuleToLLVMIR(gpuModules[0], llvmContext);
        if (!llvmModule) {
            return nullptr;
        }

        writeQwenKFragAuditLLVM(*llvmModule, "preopt_llvm");
        writeQwenPersistentRecurrenceLLVM(*llvmModule, "preopt_llvm");

        // Configure the LLVM module with correct target triple and data layout
        llvmModule->setTargetTriple(llvm::Triple(targetTriple));
        llvmModule->setDataLayout(targetMachine->createDataLayout());

        // Only global kernels should have external linkage.
        for (auto &func : llvmModule->functions()) {
            if (func.isDeclaration())
                continue;
            if (func.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL ||
                func.getCallingConv() == llvm::CallingConv::PTX_Kernel) {
                func.setLinkage(llvm::GlobalValue::ExternalLinkage);
                continue;
            }
            func.setLinkage(llvm::GlobalValue::InternalLinkage);
        }

        if (options.optimization_level > 0) {
            llvm::OptimizationLevel optLevel;
            switch (options.optimization_level) {
            case 1:
                optLevel = llvm::OptimizationLevel::O1;
                break;
            case 2:
                optLevel = llvm::OptimizationLevel::O2;
                break;
            case 3:
                optLevel = llvm::OptimizationLevel::O3;
                break;
            default:
                optLevel = llvm::OptimizationLevel::O0;
                break;
            }

            llvm::PipelineTuningOptions tuningOptions;
            tuningOptions.LoopUnrolling = true;
            tuningOptions.LoopInterleaving = true;
            tuningOptions.LoopVectorization = true;
            tuningOptions.SLPVectorization = true;

            std::optional<llvm::PassInstrumentationCallbacks> instrumentation;
            unsigned llvmPassOrdinal = 0;
            if (isQwenKFragConvergenceAuditEnabled()) {
                instrumentation.emplace();
                instrumentation->registerAfterPassCallback(
                    [&llvmPassOrdinal](llvm::StringRef passName, llvm::Any ir,
                                       const llvm::PreservedAnalyses &) {
                        auto modulePtr =
                            llvm::any_cast<const llvm::Module *>(&ir);
                        if (!modulePtr || !*modulePtr) {
                            return;
                        }
                        const auto phase =
                            "llvm_pass_" + std::to_string(llvmPassOrdinal++) +
                            "_" + qwenKFragAuditSafePassName(passName);
                        writeQwenKFragAuditLLVM(**modulePtr, phase);
                    });
            }
            llvm::PassBuilder pb(targetMachine.get(), tuningOptions,
                                 std::nullopt,
                                 instrumentation ? &*instrumentation : nullptr);

            llvm::LoopAnalysisManager lam;
            llvm::FunctionAnalysisManager fam;
            llvm::CGSCCAnalysisManager cgam;
            llvm::ModuleAnalysisManager mam;

            pb.registerModuleAnalyses(mam);
            pb.registerCGSCCAnalyses(cgam);
            pb.registerFunctionAnalyses(fam);
            pb.registerLoopAnalyses(lam);
            pb.crossRegisterProxies(lam, fam, cgam, mam);

            llvm::ModulePassManager mpm =
                pb.buildPerModuleDefaultPipeline(optLevel);

            mpm.run(*llvmModule, mam);
        }

        writeQwenKFragAuditLLVM(*llvmModule, "postopt_llvm");
        writeQwenPersistentRecurrenceLLVM(*llvmModule, "postopt_llvm");

        return llvmModule;
    }

  public:
    causalflow::avelang::ir::IRContext *ir_context_;
};

LowerToLLVM::LowerToLLVM(causalflow::avelang::ir::IRContext *ir_context)
    : impl_(std::make_unique<Impl>(ir_context)) {}

LowerToLLVM::~LowerToLLVM() = default;

std::unique_ptr<::llvm::Module>
LowerToLLVM::compile(mlir::ModuleOp module, ::llvm::LLVMContext &llvmContext,
                     const GPUCompilationOptions &options) {
    return impl_->compile(module, llvmContext, options);
}

mlir::MLIRContext *LowerToLLVM::getContext() {
    return impl_->ir_context_->GetMLIRContext();
}

} // namespace causalflow::avelang::target::gpu
