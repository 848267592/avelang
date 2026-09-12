#include "AST/ast_nodes_expr.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "IR/Intrinsics/amdgpu_mfma_signatures.h"
#include "IR/builtin_module.h"
#include "IR/constant_folder.h"
#include "IR/generator_context.h"
#include "IR/mlir_generator_impl.h"
#include "IR/named_module.h"
#include "Utils/assert.h"
#include "Utils/embedded_filesystem_view.h"
#include "intrinsic_support.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cstdint>
#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

extern "C" const unsigned char _binary_amdgpu_intrinsics_mlirbc_start[];
extern "C" const unsigned char _binary_amdgpu_intrinsics_mlirbc_end[];

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>

namespace causalflow::avelang::ir {

using namespace mlir;
using namespace mlir::ROCDL;
using namespace causalflow::avelang::dialect;
namespace cf = causalflow::avelang::dialect;
namespace amdgpu_mfma = causalflow::avelang::amdgpu::mfma;

namespace {

static llvm::StringRef GetAmdgpuIntrinsicLibrary() {
    auto *start =
        reinterpret_cast<const char *>(_binary_amdgpu_intrinsics_mlirbc_start);
    auto *end =
        reinterpret_cast<const char *>(_binary_amdgpu_intrinsics_mlirbc_end);
    return {start, static_cast<size_t>(end - start)};
}

constexpr llvm::StringRef kAmdgpuIntrinsicLibraryName =
    "amdgpu_intrinsics.mlirbc";
constexpr llvm::StringRef kAmdgpuIntrinsicLibraryTag =
    "embedded:amdgpu_intrinsics.mlirbc";
static constexpr unsigned kDataFormatU32Config = 4u << 15;

static mlir::Location GetCallLocation(GeneratorContext *ctx,
                                      const ast::ASTNode *node) {
    auto *func_gen = ctx ? ctx->GetCurrentFunctionGenerator() : nullptr;
    auto *builder = func_gen ? &func_gen->GetBuilder() : nullptr;
    SS_ASSERT(ctx && builder);
    return ctx->GetMLIRLocation(builder->getContext(), node);
}

// C15 is a numerical-closure probe for the existing generic block-dot
// operation.  It deliberately remains an environment-gated representation
// mode; no Qwen-specific source intrinsic is introduced.
static bool UseC15RealTile() {
    return llvm::sys::Process::GetEnv("AVELANG_C15_REAL_TILE") ==
           std::optional<std::string>("1");
}

// C15 score@V closure keeps the existing generic C15 V producer and asks the
// late consumer lowering to preserve both 32-column output halves.  This is
// intentionally a separate opt-in gate because the historical C15 probe only
// validated one accumulator chain and duplicated it at the public boundary.
static bool UseC15ScoreVFull64() {
    return llvm::sys::Process::GetEnv("AVELANG_C15_SCOREV_FULL64") ==
           std::optional<std::string>("1");
}

// C16 is the Q/H/K counterpart of the C15 V numerical-closure probe.  Keep
// the role as a single generic physical-tile selector; the source still uses
// the existing block-dot ABI and the late pass selects the role's
// representation plan.  This must remain opt-in so all existing C15 and
// recurrence paths stay unchanged.
static std::optional<std::string> C16RealTileRole() {
    auto role = llvm::sys::Process::GetEnv("AVELANG_C16_REAL_TILE_ROLE");
    if (!role || (*role != "Q" && *role != "H" && *role != "K"))
        return std::nullopt;
    return role;
}

static bool UseC18FullPhysicalRegion() {
    return llvm::sys::Process::GetEnv("AVELANG_STAGE6Z_FULL_PHYSICAL_REGION") ==
           std::optional<std::string>("c18");
}

static bool UseC19FullPhysicalRegion() {
    return llvm::sys::Process::GetEnv("AVELANG_STAGE6Z_FULL_PHYSICAL_REGION") ==
           std::optional<std::string>("c19");
}

static bool UseC21SelectedNativePipeline() {
    return llvm::sys::Process::GetEnv("AVELANG_STAGE6Z_FULL_PHYSICAL_REGION") ==
           std::optional<std::string>("c21");
}

static bool UseC19CompatibleFullPhysicalRegion() {
    return UseC19FullPhysicalRegion() || UseC21SelectedNativePipeline();
}

// C22 reuses the generic block-dot source operation but needs Phase-C's
// BF16 V-new to enter the late physical bridge.  This is intentionally not a
// full-region mode: it has no producer ownership or scheduling semantics.
static bool UseC22SchedulePreservingPhysical() {
    return llvm::sys::Process::GetEnv(
               "AVELANG_STAGE6Z_SCHEDULE_PRESERVING_PHYSICAL") ==
           std::optional<std::string>("c22");
}

static llvm::StringRef C19LogicalSourceRole(
    llvm::ArrayRef<mlir::Value> resolved_args, bool transposed) {
    auto getShape = [](mlir::Type type) -> llvm::ArrayRef<int64_t> {
        if (auto aveType = mlir::dyn_cast<cf::MemRefType>(type))
            return aveType.getShape();
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type))
            return memrefType.getShape();
        return {};
    };
    if (UseC19CompatibleFullPhysicalRegion()) {
        auto shape = getShape(resolved_args[2].getType());
        if (shape.size() == 4) {
            if (shape[2] == 8)
                return "V";
            if (resolved_args[2] == resolved_args[3] &&
                shape[2] == 4)
                return "Q";
        }
    }
    if (UseC22SchedulePreservingPhysical()) {
        auto shape = getShape(resolved_args[2].getType());
        if (shape.size() == 4 && shape[2] == 8)
            return "V";
    }
    return transposed ? "H" : "K";
}

} // namespace

// AMDGPU Intrinsics Module
class AMDGPUIntrinsic : public NamedModule {
  public:
    explicit AMDGPUIntrinsic();

    void Initialize() override;
    void DeclareModules(mlir::ModuleOp module) override;

    mlir::Value CreateMfmaFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                   llvm::ArrayRef<mlir::Value> resolved_args,
                                   const amdgpu_mfma::MFMAConfig &config) const;
    mlir::Value CreateRawBufferLoadX1Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferLoadX2Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferLoadX4Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferLoadX1LdsFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateQwenUpdateKFragLoadBF16x4Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateQwenPredStateKVFragLoadBF16x4Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateQwenK64PipelineStageLoadFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateQwenK64PipelineStageCommitFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateBlockDotBF16F32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateBlockDotBF16F32OperandFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, bool transposed) const;
    mlir::Value CreateBlockDotBF16F32LogicalFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, bool transposed) const;
    mlir::Value CreateBlockDotBF16F32PrecomputedVDecayFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateBlockDotBF16F32StagedVDecayFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateBlockDotBF16F32StagedVDecayPreloadedKFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateBlockDotBF16F32StagedVDecayPreloadedKStateKVFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateQwenGdnRecurrenceStepBF16F32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateQwenPersistentRecurrenceBeginFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateQwenPersistentRecurrenceEndFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferStoreX1Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferStoreX2Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferStoreX4Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateMakeRsrcFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreatePermFunction(ast::Call *call_expr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateRcpFunction(ast::Call *call_expr, GeneratorContext *ctx,
                      llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateSWaitcntFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateSchedGroupBarrierFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

  private:
    mlir::Value
    CreateGenericMFMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                              llvm::ArrayRef<mlir::Value> resolved_args,
                              const amdgpu_mfma::MFMAConfig &config) const;

    mlir::Value CreateGenericRawBufferLoadFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int width) const;

    mlir::Value CreateGenericRawBufferStoreFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int width) const;

    bool
    CheckGenericMFMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckGenericRawBufferLoadFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int width) const;
    bool CheckRawBufferLoadX1LdsFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckQwenUpdateKFragLoadBF16x4Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckQwenPredStateKVFragLoadBF16x4Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckQwenK64PipelineStageLoadFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckQwenK64PipelineStageCommitFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckBlockDotBF16F32Function(ast::Call *call_expr,
                                      GeneratorContext *ctx,
                                      llvm::ArrayRef<mlir::Value> resolved_args,
                                      bool allow_staged_vdecay = false,
                                      bool allow_preloaded_k = false,
                                      bool allow_operand_mode = false) const;
    bool CheckQwenGdnRecurrenceStepBF16F32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckQwenPersistentRecurrenceBeginFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckQwenPersistentRecurrenceEndFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckGenericRawBufferStoreFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int width) const;

    bool CheckMakeRsrcFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckPermFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckRcpFunction(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckSWaitcntFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckSchedGroupBarrierFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
};

AMDGPUIntrinsic::AMDGPUIntrinsic() : NamedModule("amdgpu") {}

void AMDGPUIntrinsic::Initialize() {
    for (const auto &config : amdgpu_mfma::MFMAConfig::GetConfigs()) {
        AddFunction(
            config.name.str(),
            [this,
             config](ast::Call *call_expr, GeneratorContext *gen_ctx,
                     llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreateMfmaFunction(call_expr, gen_ctx, resolved_args,
                                          config);
            },
            [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckGenericMFMAFunction(call_expr, gen_ctx,
                                                resolved_args);
            });
    }

    AddFunction(
        "make_rsrc",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeRsrcFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMakeRsrcFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "perm",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreatePermFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckPermFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "rcp",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRcpFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckRcpFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "s_waitcnt",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSWaitcntFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckSWaitcntFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "raw_buffer_load_x1",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferLoadX1Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferLoadFunction(call_expr, gen_ctx,
                                                     resolved_args, 1);
        });

    AddFunction(
        "raw_buffer_load_x2",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferLoadX2Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferLoadFunction(call_expr, gen_ctx,
                                                     resolved_args, 2);
        });

    AddFunction(
        "raw_buffer_load_x4",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferLoadX4Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferLoadFunction(call_expr, gen_ctx,
                                                     resolved_args, 4);
        });

    AddFunction(
        "raw_buffer_load_x1_lds",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferLoadX1LdsFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckRawBufferLoadX1LdsFunction(call_expr, gen_ctx,
                                                   resolved_args);
        });

    AddFunction(
        "qwen_update_kfrag_load_bf16x4",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateQwenUpdateKFragLoadBF16x4Function(call_expr, gen_ctx,
                                                           resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckQwenUpdateKFragLoadBF16x4Function(call_expr, gen_ctx,
                                                          resolved_args);
        });

    AddFunction(
        "qwen_pred_state_kv_frag_load_bf16x4",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateQwenPredStateKVFragLoadBF16x4Function(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckQwenPredStateKVFragLoadBF16x4Function(
                call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "qwen_k64_pipeline_stage_load",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateQwenK64PipelineStageLoadFunction(call_expr, gen_ctx,
                                                           resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckQwenK64PipelineStageLoadFunction(call_expr, gen_ctx,
                                                          resolved_args);
        });

    // BT64 is intentionally operand-generic: K has four heads and W has
    // eight, but both use the same [2,64,64] bank, BF16x8 packet ownership,
    // opaque stage token and late commit. Keep the K64 spelling above as a
    // compatibility alias for S0; joint recurrence source uses this name.
    AddFunction(
        "qwen_bt64_pipeline_stage_load",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateQwenK64PipelineStageLoadFunction(call_expr, gen_ctx,
                                                           resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckQwenK64PipelineStageLoadFunction(call_expr, gen_ctx,
                                                          resolved_args);
        });

    AddFunction(
        "qwen_k64_pipeline_stage_commit",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateQwenK64PipelineStageCommitFunction(call_expr, gen_ctx,
                                                             resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckQwenK64PipelineStageCommitFunction(call_expr, gen_ctx,
                                                            resolved_args);
        });

    AddFunction(
        "qwen_bt64_pipeline_stage_commit",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateQwenK64PipelineStageCommitFunction(call_expr, gen_ctx,
                                                             resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckQwenK64PipelineStageCommitFunction(call_expr, gen_ctx,
                                                            resolved_args);
        });

    AddFunction(
        "block_dot_bf16_f32",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBlockDotBF16F32Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckBlockDotBF16F32Function(call_expr, gen_ctx,
                                                resolved_args);
        });

    // These helpers still create AMDGPUBlockDotBF16F32Op. They only attach
    // generic operand metadata; the late pass selects the implementation.
    // Keeping one operation identity makes K/H same-source A/B audits
    // possible without introducing a Qwen-specific dot op.
    AddFunction(
        "block_dot_bf16_f32_operand",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBlockDotBF16F32OperandFunction(
                call_expr, gen_ctx, resolved_args, /*transposed=*/false);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckBlockDotBF16F32Function(
                call_expr, gen_ctx, resolved_args,
                /*allow_staged_vdecay=*/false,
                /*allow_preloaded_k=*/false,
                /*allow_operand_mode=*/true);
        });

    AddFunction(
        "block_dot_bf16_f32_operand_transposed",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBlockDotBF16F32OperandFunction(
                call_expr, gen_ctx, resolved_args, /*transposed=*/true);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckBlockDotBF16F32Function(
                call_expr, gen_ctx, resolved_args,
                /*allow_staged_vdecay=*/false,
                /*allow_preloaded_k=*/false,
                /*allow_operand_mode=*/true);
        });

    // Full-scope logical-block constructors still create the same
    // AMDGPUBlockDotBF16F32Op.  They only mark that the third operand is the
    // global logical B block and that the late lowering owns its producer,
    // shared placement and dot-operand construction.  The names are generic
    // block-dot API, not Qwen/chunk-o intrinsics.
    AddFunction(
        "block_dot_bf16_f32_logical",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBlockDotBF16F32LogicalFunction(
                call_expr, gen_ctx, resolved_args, /*transposed=*/false);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckBlockDotBF16F32Function(
                call_expr, gen_ctx, resolved_args,
                /*allow_staged_vdecay=*/false,
                /*allow_preloaded_k=*/false,
                /*allow_operand_mode=*/true);
        });

    AddFunction(
        "block_dot_bf16_f32_logical_transposed",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBlockDotBF16F32LogicalFunction(
                call_expr, gen_ctx, resolved_args, /*transposed=*/true);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckBlockDotBF16F32Function(
                call_expr, gen_ctx, resolved_args,
                /*allow_staged_vdecay=*/false,
                /*allow_preloaded_k=*/false,
                /*allow_operand_mode=*/true);
        });

    AddFunction(
        "block_dot_bf16_f32_precomputed_vdecay",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBlockDotBF16F32PrecomputedVDecayFunction(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckBlockDotBF16F32Function(call_expr, gen_ctx,
                                                resolved_args);
        });

    AddFunction(
        "block_dot_bf16_f32_staged_vdecay",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBlockDotBF16F32StagedVDecayFunction(call_expr, gen_ctx,
                                                             resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckBlockDotBF16F32Function(call_expr, gen_ctx,
                                                resolved_args,
                                                /*allow_staged_vdecay=*/true);
        });

    AddFunction(
        "block_dot_bf16_f32_staged_vdecay_preloaded_k",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBlockDotBF16F32StagedVDecayPreloadedKFunction(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckBlockDotBF16F32Function(
                call_expr, gen_ctx, resolved_args,
                /*allow_staged_vdecay=*/true,
                /*allow_preloaded_k=*/true);
        });

    // This is not another block-dot schedule.  It retains the R4 staged
    // V-decay/preloaded-K producer and MFMA geometry, but makes the output
    // fragment's physical ownership explicit as KxV.  The late lowerer is
    // responsible for swapping only the MFMA operand roles; callers retain
    // the same 14-argument ABI and cannot accidentally select it outside the
    // existing persistent recurrence path.
    AddFunction(
        "block_dot_bf16_f32_staged_vdecay_preloaded_k_state_kv",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBlockDotBF16F32StagedVDecayPreloadedKStateKVFunction(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckBlockDotBF16F32Function(
                call_expr, gen_ctx, resolved_args,
                /*allow_staged_vdecay=*/true,
                /*allow_preloaded_k=*/true);
        });

    AddFunction(
        "qwen_gdn_recurrence_step_bf16_f32",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateQwenGdnRecurrenceStepBF16F32Function(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckQwenGdnRecurrenceStepBF16F32Function(call_expr, gen_ctx,
                                                             resolved_args);
        });

    AddFunction(
        "qwen_persistent_recurrence_begin",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateQwenPersistentRecurrenceBeginFunction(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckQwenPersistentRecurrenceBeginFunction(
                call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "qwen_persistent_recurrence_end",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateQwenPersistentRecurrenceEndFunction(
                call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckQwenPersistentRecurrenceEndFunction(
                call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "raw_buffer_store_x1",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferStoreX1Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferStoreFunction(call_expr, gen_ctx,
                                                      resolved_args, 1);
        });

    AddFunction(
        "raw_buffer_store_x2",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferStoreX2Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferStoreFunction(call_expr, gen_ctx,
                                                      resolved_args, 2);
        });

    AddFunction(
        "raw_buffer_store_x4",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferStoreX4Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferStoreFunction(call_expr, gen_ctx,
                                                      resolved_args, 4);
        });

    AddFunction(
        "sched_group_barrier",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSchedGroupBarrierFunction(call_expr, gen_ctx,
                                                   resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckSchedGroupBarrierFunction(call_expr, gen_ctx,
                                                  resolved_args);
        });
}

void AMDGPUIntrinsic::DeclareModules(mlir::ModuleOp module) {
    if (!module)
        return;

    // Register the bytecode in the intrinsic registry
    auto libraryBytes = GetAmdgpuIntrinsicLibrary();
    auto &registry = utils::EmbeddedFilesystemView::getInstance();
    registry.registerFile(std::string(kAmdgpuIntrinsicLibraryName),
                          libraryBytes);

    intrinsics::GetOrCreateImplementationContainer(module, "amdgpu",
                                                   kAmdgpuIntrinsicLibraryTag);

    auto loadDialects = [](mlir::MLIRContext *ctx) {
        ctx->loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::vector::VectorDialect, mlir::LLVM::LLVMDialect,
                         mlir::ROCDL::ROCDLDialect, cf::AveLangDialect>();
    };

    if (failed(intrinsics::EnsureIntrinsicDeclarations(
            module, kAmdgpuIntrinsicLibraryName, libraryBytes, loadDialects))) {
        module.emitError() << "failed to declare AMDGPU intrinsics";
    }
}

namespace {
mlir::Type GetMfmaElemType(amdgpu_mfma::VectorElemKind kind,
                           mlir::OpBuilder &builder) {
    switch (kind) {
    case amdgpu_mfma::VectorElemKind::I32:
        return builder.getI32Type();
    case amdgpu_mfma::VectorElemKind::F16:
        return builder.getF16Type();
    case amdgpu_mfma::VectorElemKind::F32:
        return builder.getF32Type();
    case amdgpu_mfma::VectorElemKind::BF16:
        return builder.getBF16Type();
    }
    llvm_unreachable("Unsupported MFMA element kind");
}

mlir::Value ConvertToI32(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value value) {
    if (value.getType().isIndex()) {
        return mlir::arith::IndexCastOp::create(builder, location,
                                                builder.getI32Type(), value);
    }

    auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
    if (!intType || intType.getWidth() == 32) {
        return value;
    }
    if (intType.getWidth() < 32) {
        return mlir::arith::ExtUIOp::create(builder, location,
                                            builder.getI32Type(), value);
    }
    return mlir::arith::TruncIOp::create(builder, location,
                                         builder.getI32Type(), value);
}

mlir::Value ConvertToIndex(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value value) {
    if (value.getType().isIndex()) {
        return value;
    }
    auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
    if (!intType) {
        return value;
    }
    return mlir::arith::IndexCastOp::create(builder, location,
                                            builder.getIndexType(), value);
}
} // namespace

mlir::Value AMDGPUIntrinsic::CreateGenericMFMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args,
    const amdgpu_mfma::MFMAConfig &config) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto a = resolved_args[0];
    auto b = resolved_args[1];
    auto c = resolved_args[2];

    auto type_a = GetMfmaElemType(config.aElem, builder);
    auto type_c = GetMfmaElemType(config.cElem, builder);

    int64_t c_elements = config.GetCElementCount();
    auto result_vector_type = mlir::VectorType::get({c_elements}, type_c);

    // Convert MLIR types to string representations for attributes
    std::string type_a_str, type_c_str;
    llvm::raw_string_ostream type_a_stream(type_a_str);
    llvm::raw_string_ostream type_c_stream(type_c_str);

    if (type_a.isF16()) {
        type_a_stream << "f16";
    } else if (type_a.isBF16()) {
        type_a_stream << "bf16";
    } else if (type_a.isF32()) {
        type_a_stream << "f32";
    } else if (type_a.isInteger(32)) {
        type_a_stream << "i32";
    } else if (type_a.isInteger(8)) {
        type_a_stream << "i8";
    } else {
        type_a_stream << type_a;
    }

    if (type_c.isF32()) {
        type_c_stream << "f32";
    } else if (type_c.isF16()) {
        type_c_stream << "f16";
    } else if (type_c.isInteger(32)) {
        type_c_stream << "i32";
    } else {
        type_c_stream << type_c;
    }

    // Create GPUOp AMDGPU MFMA operation with config attributes
    auto mfma_op = cf::AMDGPUMfmaOp::create(
        builder, location, result_vector_type, a, b, c,
        mlir::IntegerAttr::get(builder.getI32Type(), config.m),
        mlir::IntegerAttr::get(builder.getI32Type(), config.n),
        mlir::IntegerAttr::get(builder.getI32Type(), config.k),
        mlir::StringAttr::get(builder.getContext(), type_a_str),
        mlir::StringAttr::get(builder.getContext(), type_c_str));

    return mfma_op.getResult();
}

mlir::Value AMDGPUIntrinsic::CreateMfmaFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args,
    const amdgpu_mfma::MFMAConfig &config) const {
    return CreateGenericMFMAFunction(call_expr, ctx, resolved_args, config);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferLoadX1Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferLoadFunction(call_expr, ctx, resolved_args, 1);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferLoadX2Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferLoadFunction(call_expr, ctx, resolved_args, 2);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferLoadX4Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferLoadFunction(call_expr, ctx, resolved_args, 4);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferLoadX1LdsFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto size = ConstantFolder::FoldIntValue(resolved_args[2]);
    auto aux = ConstantFolder::FoldIntValue(resolved_args[6]);

    if (!size || *size != 4 || !aux || *aux != 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds currently requires compile-time "
               "size=4 and aux=0";
        return nullptr;
    }

    auto ldsPtr = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), resolved_args[1]);
    auto rawOffset = ConvertToIndex(builder, location, resolved_args[5]);
    auto ldsPtrWithOffset =
        mlir::arith::AddIOp::create(builder, location, ldsPtr, rawOffset);

    auto funcName = intrinsics::MakeIntrinsicFuncName(
        "amdgpu", "llvm_amdgcn_raw_buffer_load_lds_u32");
    mlir::func::CallOp::create(
        builder, location, funcName, mlir::TypeRange{},
        mlir::ValueRange{resolved_args[0], ldsPtrWithOffset,
                         ConvertToI32(builder, location, resolved_args[3]),
                         ConvertToI32(builder, location, resolved_args[4])});

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreateQwenUpdateKFragLoadBF16x4Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto resultType = mlir::VectorType::get({4}, builder.getBF16Type());

    auto op = cf::AMDGPUQwenUpdateKFragLoadOp::create(
        builder, location, resultType, resolved_args[0], resolved_args[1],
        resolved_args[2], resolved_args[3], resolved_args[4], resolved_args[5],
        resolved_args[6]);
    return op.getResult();
}

mlir::Value AMDGPUIntrinsic::CreateQwenPredStateKVFragLoadBF16x4Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto resultType = mlir::VectorType::get({4}, builder.getBF16Type());
    auto op = cf::AMDGPUQwenPredStateKVFragLoadOp::create(
        builder, location, resultType, resolved_args[0], resolved_args[1],
        resolved_args[2], resolved_args[3], resolved_args[4]);
    return op.getResult();
}

mlir::Value AMDGPUIntrinsic::CreateQwenK64PipelineStageLoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto op = cf::AMDGPUQwenK64PipelineStageLoadOp::create(
        builder, location, builder.getI64Type(), resolved_args[0],
        resolved_args[1], resolved_args[2], resolved_args[3], resolved_args[4]);
    op->setAttr("avelang.qwen.k64.pipeline.stage_token",
                mlir::UnitAttr::get(builder.getContext()));
    op->setAttr("avelang.qwen.bt64.pipeline.stage_token",
                mlir::UnitAttr::get(builder.getContext()));
    return op.getStageToken();
}

mlir::Value AMDGPUIntrinsic::CreateQwenK64PipelineStageCommitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto op = cf::AMDGPUQwenK64PipelineStageCommitOp::create(
        builder, location, resolved_args[0], resolved_args[1]);
    op->setAttr("avelang.qwen.k64.pipeline.commit",
                mlir::UnitAttr::get(builder.getContext()));
    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreateBlockDotBF16F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto resultType = mlir::VectorType::get({32}, builder.getF32Type());
    auto op = cf::AMDGPUBlockDotBF16F32Op::create(builder, location, resultType,
                                                  resolved_args);
    return op.getResult();
}

mlir::Value AMDGPUIntrinsic::CreateBlockDotBF16F32OperandFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool transposed) const {
    auto result = CreateBlockDotBF16F32Function(call_expr, ctx, resolved_args);
    if (auto *def = result.getDefiningOp()) {
        auto *context = def->getContext();
        def->setAttr("avelang.block_dot.operand_mode",
                     mlir::StringAttr::get(context, "generic_mfma_b"));
        def->setAttr("avelang.block_dot.operand_role",
                     mlir::StringAttr::get(context, "B"));
        def->setAttr("avelang.block_dot.source_role",
                     mlir::StringAttr::get(
                         context, C19LogicalSourceRole(resolved_args, transposed)));
        def->setAttr("avelang.block_dot.transpose",
                     mlir::StringAttr::get(
                         context, transposed ? "rhs_transposed" : "none"));
        def->setAttr("avelang.block_dot.logical_shape",
                     mlir::StringAttr::get(context, "32x32x32"));
        def->setAttr("avelang.block_dot.layout_intent",
                     mlir::StringAttr::get(context, "shared_mfma32_b"));
        def->setAttr("avelang.block_dot.single_accumulator",
                     mlir::UnitAttr::get(context));
        if (auto c16Role = C16RealTileRole()) {
            def->setAttr("c16.real_tile_role",
                         mlir::StringAttr::get(context, *c16Role));
            def->setAttr("avelang.block_dot.source_role",
                         mlir::StringAttr::get(context, *c16Role));
            def->setAttr(
                "avelang.block_dot.physical_contract",
                mlir::StringAttr::get(
                    context,
                    "c13_distributed_c14_shared_mfma32_qhk"));
        } else if (UseC15RealTile()) {
            def->setAttr("c15.real_tile", mlir::StringAttr::get(context, "V"));
            def->setAttr("avelang.block_dot.source_role",
                         mlir::StringAttr::get(context, "V"));
            def->setAttr("avelang.block_dot.logical_shape",
                         mlir::StringAttr::get(context, "64x64"));
            def->setAttr("avelang.block_dot.physical_contract",
                         mlir::StringAttr::get(
                             context,
                             "c13_distributed_c14_rotating_shared_mfma32"));
            if (UseC15ScoreVFull64())
                def->setAttr("c15.scorev_full64", mlir::UnitAttr::get(context));
        }
    }
    return result;
}

mlir::Value AMDGPUIntrinsic::CreateBlockDotBF16F32LogicalFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool transposed) const {
    auto result = CreateBlockDotBF16F32Function(call_expr, ctx, resolved_args);
    if (auto *def = result.getDefiningOp()) {
        auto *context = def->getContext();
        def->setAttr("avelang.block_dot.operand_mode",
                     mlir::StringAttr::get(context, "full_scope"));
        def->setAttr("avelang.block_dot.scope",
                     mlir::StringAttr::get(context, "logical_block"));
        def->setAttr("avelang.block_dot.operand_role",
                     mlir::StringAttr::get(context, "B"));
        def->setAttr("avelang.block_dot.source_role",
                     mlir::StringAttr::get(
                         context, C19LogicalSourceRole(resolved_args, transposed)));
        def->setAttr("avelang.block_dot.transpose",
                     mlir::StringAttr::get(
                         context, transposed ? "rhs_transposed" : "none"));
        def->setAttr("avelang.block_dot.logical_shape",
                     mlir::StringAttr::get(context, "32x32x32"));
        def->setAttr("avelang.block_dot.logical_source_operand",
                     mlir::StringAttr::get(context, "source_block"));
        def->setAttr("avelang.block_dot.lhs_residency",
                     mlir::StringAttr::get(context, "existing_shared"));
        def->setAttr("avelang.block_dot.rhs_producer",
                     mlir::StringAttr::get(context, "global_packet"));
        def->setAttr("avelang.block_dot.reuse_key",
                     mlir::StringAttr::get(context, "lhs_ssa_block"));
        def->setAttr("avelang.block_dot.layout_intent",
                     mlir::StringAttr::get(context, "typed_shared_dot"));
    }
    return result;
}

mlir::Value AMDGPUIntrinsic::CreateBlockDotBF16F32PrecomputedVDecayFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto result = CreateBlockDotBF16F32Function(call_expr, ctx, resolved_args);
    if (auto *def = result.getDefiningOp()) {
        def->setAttr("avelang.block_dot.precomputed_vdecay",
                     mlir::UnitAttr::get(def->getContext()));
    }
    return result;
}

mlir::Value AMDGPUIntrinsic::CreateBlockDotBF16F32StagedVDecayFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto result = CreateBlockDotBF16F32Function(call_expr, ctx, resolved_args);
    if (auto *def = result.getDefiningOp()) {
        def->setAttr("avelang.block_dot.precomputed_vdecay",
                     mlir::UnitAttr::get(def->getContext()));
        def->setAttr("avelang.block_dot.staged_vdecay",
                     mlir::UnitAttr::get(def->getContext()));
    }
    return result;
}

mlir::Value AMDGPUIntrinsic::CreateBlockDotBF16F32StagedVDecayPreloadedKFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto result = CreateBlockDotBF16F32StagedVDecayFunction(
        call_expr, ctx, resolved_args);
    if (auto *def = result.getDefiningOp()) {
        def->setAttr("avelang.block_dot.preloaded_k",
                     mlir::UnitAttr::get(def->getContext()));
    }
    return result;
}

mlir::Value AMDGPUIntrinsic::CreateBlockDotBF16F32StagedVDecayPreloadedKStateKVFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto result = CreateBlockDotBF16F32StagedVDecayPreloadedKFunction(
        call_expr, ctx, resolved_args);
    if (auto *def = result.getDefiningOp()) {
        def->setAttr("avelang.block_dot.state_kv",
                     mlir::UnitAttr::get(def->getContext()));
    }
    return result;
}

mlir::Value AMDGPUIntrinsic::CreateQwenGdnRecurrenceStepBF16F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto resultType = mlir::VectorType::get({32}, builder.getF32Type());
    auto op = cf::AMDGPUQwenGdnRecurrenceStepBF16F32Op::create(
        builder, location, resultType, resolved_args);
    op->setAttr("avelang.qwen_gdn.recurrence_step",
                mlir::UnitAttr::get(builder.getContext()));
    return op.getResult();
}

mlir::Value AMDGPUIntrinsic::CreateQwenPersistentRecurrenceBeginFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto op = cf::AMDGPUQwenPersistentRecurrenceOp::create(
        builder, location, mlir::TypeRange{}, resolved_args);
    // A single explicit yield gives the frontend marker a valid MLIR region.
    // Formation replaces that yield with the complete validated B0 loop.
    auto &body = op.getBody().emplaceBlock();
    mlir::OpBuilder bodyBuilder(&body, body.end());
    cf::AMDGPUQwenPersistentRecurrenceYieldOp::create(bodyBuilder, location);
    op->setAttr("avelang.qwen.persistent_recurrence.frontend_marker",
                mlir::UnitAttr::get(builder.getContext()));
    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreateQwenPersistentRecurrenceEndFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    cf::AMDGPUQwenPersistentRecurrenceEndOp::create(builder, location);
    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferStoreX1Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferStoreFunction(call_expr, ctx, resolved_args,
                                               1);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferStoreX2Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferStoreFunction(call_expr, ctx, resolved_args,
                                               2);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferStoreX4Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferStoreFunction(call_expr, ctx, resolved_args,
                                               4);
}

mlir::Value AMDGPUIntrinsic::CreateMakeRsrcFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto memref = resolved_args[0];
    auto rangeValue = resolved_args[1];

    auto ptrIndex = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), memref);
    auto ptrI64 = mlir::arith::IndexCastOp::create(
        builder, location, builder.getI64Type(), ptrIndex);

    auto shiftAmount =
        mlir::arith::ConstantIntOp::create(builder, location, 32, 64);
    auto ptrHighI64 =
        mlir::arith::ShRUIOp::create(builder, location, ptrI64, shiftAmount);
    auto ptrLowI32 = mlir::arith::TruncIOp::create(
        builder, location, builder.getI32Type(), ptrI64);
    auto ptrHighI32 = mlir::arith::TruncIOp::create(
        builder, location, builder.getI32Type(), ptrHighI64);
    mlir::Value rangeI32 = rangeValue;
    if (rangeI32.getType().isIndex()) {
        rangeI32 = mlir::arith::IndexCastOp::create(
            builder, location, builder.getI32Type(), rangeI32);
    } else if (auto intType =
                   mlir::dyn_cast<mlir::IntegerType>(rangeI32.getType())) {
        if (intType.getWidth() < 32) {
            rangeI32 = mlir::arith::ExtUIOp::create(
                builder, location, builder.getI32Type(), rangeI32);
        } else if (intType.getWidth() > 32) {
            rangeI32 = mlir::arith::TruncIOp::create(
                builder, location, builder.getI32Type(), rangeI32);
        }
    }
    auto configI32 = mlir::arith::ConstantIntOp::create(
        builder, location, kDataFormatU32Config, 32);

    auto rsrcType = mlir::VectorType::get({4}, builder.getI32Type());
    auto zeroAttr = builder.getIntegerAttr(builder.getI32Type(), 0);
    auto zeroRsrcAttr = mlir::DenseElementsAttr::get(rsrcType, zeroAttr);
    mlir::Value rsrc = mlir::arith::ConstantOp::create(builder, location,
                                                       rsrcType, zeroRsrcAttr);

    rsrc = mlir::vector::InsertOp::create(
        builder, location, ptrLowI32, rsrc,
        llvm::SmallVector<mlir::OpFoldResult>{builder.getI64IntegerAttr(0)});
    rsrc = mlir::vector::InsertOp::create(
        builder, location, ptrHighI32, rsrc,
        llvm::SmallVector<mlir::OpFoldResult>{builder.getI64IntegerAttr(1)});
    rsrc = mlir::vector::InsertOp::create(
        builder, location, rangeI32, rsrc,
        llvm::SmallVector<mlir::OpFoldResult>{builder.getI64IntegerAttr(2)});
    rsrc = mlir::vector::InsertOp::create(
        builder, location, configI32, rsrc,
        llvm::SmallVector<mlir::OpFoldResult>{builder.getI64IntegerAttr(3)});

    return rsrc;
}

mlir::Value AMDGPUIntrinsic::CreateRcpFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto funcName =
        intrinsics::MakeIntrinsicFuncName("amdgpu", "llvm_amdgcn_rcp_f32");
    auto callOp = mlir::func::CallOp::create(
        builder, location, funcName, builder.getF32Type(), resolved_args[0]);
    return callOp.getResult(0);
}

mlir::Value AMDGPUIntrinsic::CreateSWaitcntFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto vmcnt = ConstantFolder::FoldIntValue(resolved_args[0]);
    auto expcnt = ConstantFolder::FoldIntValue(resolved_args[1]);
    auto lgkmcnt = ConstantFolder::FoldIntValue(resolved_args[2]);

    auto is_valid = [](std::optional<int64_t> value, int64_t max) {
        return value && *value >= 0 && *value <= max;
    };

    if (!is_valid(vmcnt, 63) || !is_valid(expcnt, 7) ||
        !is_valid(lgkmcnt, 15)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "s_waitcnt(vmcnt, expcnt, lgkmcnt) requires compile-time "
               "integer arguments in ranges vmcnt=[0,63], expcnt=[0,7], "
               "lgkmcnt=[0,15]";
        return nullptr;
    }

    auto vmcntU = static_cast<uint32_t>(*vmcnt);
    auto waitcntValue = ((vmcntU & 48u) << 10) |
                        ((static_cast<uint32_t>(*lgkmcnt) & 15u) << 8) |
                        ((static_cast<uint32_t>(*expcnt) & 7u) << 4) |
                        (vmcntU & 15u);
    auto waitcntAttr =
        mlir::IntegerAttr::get(builder.getI32Type(), waitcntValue);
    mlir::ROCDL::SWaitcntOp::create(builder, location, waitcntAttr);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreatePermFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto funcName =
        intrinsics::MakeIntrinsicFuncName("amdgpu", "llvm_amdgcn_perm");
    auto callOp = mlir::func::CallOp::create(
        builder, location, funcName, builder.getI32Type(), resolved_args);
    return callOp.getResult(0);
}

mlir::Value AMDGPUIntrinsic::CreateSchedGroupBarrierFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto mask = ConstantFolder::FoldIntValue(resolved_args[0]);
    auto size = ConstantFolder::FoldIntValue(resolved_args[1]);
    auto group_id = ConstantFolder::FoldIntValue(resolved_args[2]);

    auto is_valid_u32 = [](std::optional<int64_t> value) {
        return value && *value >= 0 &&
               static_cast<uint64_t>(*value) <=
                   static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    };

    if (!is_valid_u32(mask) || !is_valid_u32(size) || !is_valid_u32(group_id)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "sched_group_barrier(mask, size, group_id) requires compile-"
               "time non-negative integer arguments <= 2^32-1";
        return nullptr;
    }

    mlir::ROCDL::SchedGroupBarrier::create(
        builder, location, static_cast<uint32_t>(*mask),
        static_cast<uint32_t>(*size), static_cast<uint32_t>(*group_id));

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreateGenericRawBufferLoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int width) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto rsrc = resolved_args[0];
    auto vindex = resolved_args[1];
    auto soffset = resolved_args[2];
    auto aux = resolved_args[3];

    mlir::Type result_type;
    if (width == 1) {
        result_type = builder.getI32Type();
    } else if (width == 2) {
        result_type = mlir::VectorType::get({2}, builder.getI32Type());
    } else if (width == 4) {
        result_type = mlir::VectorType::get({4}, builder.getI32Type());
    } else {
        llvm_unreachable("Unsupported raw_buffer_load width");
    }

    // Create GPUOp AMDGPU Raw Buffer Load operation
    auto load_op = cf::AMDGPURawBufferLoadOp::create(
        builder, location, result_type, rsrc, vindex, soffset, aux);

    return load_op.getResult();
}

mlir::Value AMDGPUIntrinsic::CreateGenericRawBufferStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int width) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto vdata = resolved_args[0];
    auto rsrc = resolved_args[1];
    auto vindex = resolved_args[2];
    auto soffset = resolved_args[3];
    auto aux = resolved_args[4];

    auto vdata_type = vdata.getType();

    if (width > 1) {
        auto vector_type = mlir::cast<mlir::VectorType>(vdata_type);
        (void)vector_type;
    }

    // Create GPUOp AMDGPU Raw Buffer Store operation
    cf::AMDGPURawBufferStoreOp::create(builder, location, vdata, rsrc, vindex,
                                       soffset, aux);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool AMDGPUIntrinsic::CheckGenericMFMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mfma operation requires exactly 3 arguments: a, b, c";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mfma operation";
        return false;
    }

    if (!mlir::dyn_cast<mlir::VectorType>(resolved_args[0].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[1].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[2].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "MFMA operands must be vector types";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckGenericRawBufferLoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int width) const {
    if (resolved_args.size() != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x" << width
            << " requires exactly 4 arguments: rsrc, vindex, soffset, aux";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for raw_buffer_load_x" << width;
        return false;
    }

    if (width != 1 && width != 2 && width != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Unsupported width for raw_buffer_load_x" << width
            << " (supported: 1, 2, 4)";
        return false;
    }

    auto rsrc_type = resolved_args[0].getType();
    auto rsrc_vector = mlir::dyn_cast<mlir::VectorType>(rsrc_type);
    if (!rsrc_vector || rsrc_vector.getNumElements() != 4 ||
        !rsrc_vector.getElementType().isInteger(32)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x" << width
            << " expects rsrc to be vector<4xi32>";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckRawBufferLoadX1LdsFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 7) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds requires exactly 7 arguments: "
               "rsrc, lds_ptr, size, vindex, soffset, offset, aux";
        return false;
    }

    for (auto value : resolved_args) {
        if (!value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate operands for raw_buffer_load_x1_lds";
            return false;
        }
    }

    auto rsrcType = resolved_args[0].getType();
    auto rsrcVector = mlir::dyn_cast<mlir::VectorType>(rsrcType);
    if (!rsrcVector || rsrcVector.getNumElements() != 4 ||
        !rsrcVector.getElementType().isInteger(32)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds expects rsrc to be vector<4xi32>";
        return false;
    }

    auto memrefType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[1].getType());
    if (!memrefType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds expects lds_ptr to be a memref";
        return false;
    }

    for (auto operand : {resolved_args[2], resolved_args[3], resolved_args[4],
                         resolved_args[5], resolved_args[6]}) {
        if (!operand.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "raw_buffer_load_x1_lds size/vindex/soffset/offset/aux "
                   "operands must be integer or index types";
            return false;
        }
    }

    auto size = ConstantFolder::FoldIntValue(resolved_args[2]);
    auto aux = ConstantFolder::FoldIntValue(resolved_args[6]);
    if (!size || *size != 4 || !aux || *aux != 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds currently requires compile-time "
               "size=4 and aux=0";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckQwenUpdateKFragLoadBF16x4Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
    };

    if (resolved_args.size() != 7) {
        report("qwen_update_kfrag_load_bf16x4 requires exactly 7 arguments: "
               "shared_k, source_k, thread_id, key_head, token_window_base, "
               "k_col, token_fragment_base");
        return false;
    }
    for (auto value : resolved_args) {
        if (!value) {
            report("qwen_update_kfrag_load_bf16x4 failed to resolve operands");
            return false;
        }
    }

    auto memrefType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!memrefType || memrefType.getRank() != 2 ||
        memrefType.getShape() != llvm::ArrayRef<int64_t>({128, 64}) ||
        !memrefType.getElementType().isBF16()) {
        report("qwen_update_kfrag_load_bf16x4 expects shared BF16 "
               "tensor shape [128,64]");
        return false;
    }
    auto memorySpace = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
        memrefType.getMemorySpace());
    if (!memorySpace ||
        memorySpace.getValue() != mlir::gpu::AddressSpace::Workgroup) {
        report("qwen_update_kfrag_load_bf16x4 expects workgroup/shared "
               "memory");
        return false;
    }
    auto sourceType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[1].getType());
    auto sourceShape =
        sourceType ? sourceType.getShape() : llvm::ArrayRef<int64_t>();
    if (!sourceType || sourceShape.size() != 4 || sourceShape[0] != 1 ||
        sourceShape[1] < 64 || sourceShape[2] != 4 || sourceShape[3] != 128 ||
        !sourceType.getElementType().isBF16()) {
        report("qwen_update_kfrag_load_bf16x4 expects source BF16 tensor "
               "shape [1,T,4,128] with T >= 64");
        return false;
    }
    for (auto value : resolved_args.drop_front(2)) {
        if (!value.getType().isIntOrIndex()) {
            report("qwen_update_kfrag_load_bf16x4 index operands must be "
                   "integer or index values");
            return false;
        }
    }
    if (resolved_args[0] == resolved_args[1]) {
        report("qwen_update_kfrag_load_bf16x4 shared/source tensors must "
               "be distinct");
        return false;
    }
    return true;
}

bool AMDGPUIntrinsic::CheckQwenPredStateKVFragLoadBF16x4Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
    };
    if (resolved_args.size() != 5) {
        report("qwen_pred_state_kv_frag_load_bf16x4 requires state_kv, wave, "
               "k_vector, v_lane, and fragment_offset");
        return false;
    }
    for (auto value : resolved_args) {
        if (!value) {
            report("qwen_pred_state_kv_frag_load_bf16x4 failed to resolve operands");
            return false;
        }
    }
    auto state = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!state || state.getShape() != llvm::ArrayRef<int64_t>({2, 64, 32}) ||
        !state.getElementType().isBF16()) {
        report("qwen_pred_state_kv_frag_load_bf16x4 expects shared BF16 "
               "state_kv[2,64,32]");
        return false;
    }
    auto space = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
        state.getMemorySpace());
    if (!space || space.getValue() != mlir::gpu::AddressSpace::Workgroup) {
        report("qwen_pred_state_kv_frag_load_bf16x4 expects workgroup/shared "
               "state_kv");
        return false;
    }
    for (auto value : resolved_args.drop_front()) {
        if (!value.getType().isIntOrIndex()) {
            report("qwen_pred_state_kv_frag_load_bf16x4 coordinates must be "
                   "integer or index values");
            return false;
        }
    }
    return true;
}

bool AMDGPUIntrinsic::CheckQwenK64PipelineStageLoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
    };
    if (resolved_args.size() != 5) {
        report("qwen_bt64_pipeline_stage_load requires source, thread_id, "
               "chunk_start, head, half");
        return false;
    }
    for (auto value : resolved_args) {
        if (!value) {
        report("qwen_bt64_pipeline_stage_load failed to resolve operands");
            return false;
        }
    }
    auto source = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!source || source.getShape().size() != 4 || source.getShape()[0] != 1 ||
        source.getShape()[1] < 64 ||
        (source.getShape()[2] != 4 && source.getShape()[2] != 8) ||
        source.getShape()[3] != 128 || !source.getElementType().isBF16()) {
        report("qwen_bt64_pipeline_stage_load expects BF16 source [1,T,4|8,128] "
               "with T >= 64");
        return false;
    }
    for (auto value : resolved_args.drop_front()) {
        if (!value.getType().isIntOrIndex()) {
            report("qwen_bt64_pipeline_stage_load index operands must be integer or index");
            return false;
        }
    }
    return true;
}

bool AMDGPUIntrinsic::CheckQwenK64PipelineStageCommitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
    };
    if (resolved_args.size() != 2 || !resolved_args[0] || !resolved_args[1]) {
        report("qwen_bt64_pipeline_stage_commit requires stage_token and shared_wk_bank");
        return false;
    }
    if (!resolved_args[0].getType().isInteger(64)) {
        report("qwen_bt64_pipeline_stage_commit expects an opaque i64 token");
        return false;
    }
    auto bank = mlir::dyn_cast<cf::MemRefType>(resolved_args[1].getType());
    auto memorySpace = bank ? mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
                                  bank.getMemorySpace())
                            : nullptr;
    if (!bank || bank.getShape() != llvm::ArrayRef<int64_t>({2, 64, 64}) ||
        !bank.getElementType().isBF16() || !memorySpace ||
        memorySpace.getValue() != mlir::gpu::AddressSpace::Workgroup) {
        report("qwen_bt64_pipeline_stage_commit expects BF16 workgroup W/K bank [2,64,64]");
        return false;
    }
    return true;
}

bool AMDGPUIntrinsic::CheckBlockDotBF16F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, bool allow_staged_vdecay,
    bool allow_preloaded_k, bool allow_operand_mode) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
    };
    if (resolved_args.size() != 14) {
        report("block_dot_bf16_f32 requires 14 arguments: a_stage, b_stage, "
               "k, v_new, g, tid, chunk_start, value_head, key_head, "
               "value_base, k_half, g_last, acc_low, acc_high");
        return false;
    }
    for (auto value : resolved_args) {
        if (!value) {
            report("block_dot_bf16_f32 failed to resolve operands");
            return false;
        }
    }
    auto isShared = [](mlir::Value value, llvm::ArrayRef<int64_t> shape) {
        auto type = mlir::dyn_cast<cf::MemRefType>(value.getType());
        auto space = type ? mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
                                type.getMemorySpace())
                          : nullptr;
        return type && type.getShape() == shape &&
               type.getElementType().isBF16() && space &&
               space.getValue() == mlir::gpu::AddressSpace::Workgroup;
    };
    const bool isIndependentA = isShared(resolved_args[0], {2, 32, 32});
    const bool isCooperativeA = isShared(resolved_args[0], {1, 32, 32});
    const bool isStagedA =
        allow_staged_vdecay && isShared(resolved_args[0], {1, 32, 64});
    const bool isRegularB = isShared(resolved_args[1], {32, 32});
    const bool isStagedB =
        allow_staged_vdecay && isShared(resolved_args[1], {64, 64});
    const bool isPreloadedB =
        allow_preloaded_k && isShared(resolved_args[1], {2, 64, 64});
    auto isOperandShared = [](mlir::Value value) {
        auto type = mlir::dyn_cast<cf::MemRefType>(value.getType());
        auto space = type ? mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
                                type.getMemorySpace())
                          : nullptr;
        return type && type.getRank() == 2 && type.getShape()[0] >= 32 &&
               type.getShape()[1] >= 32 && type.getElementType().isBF16() &&
               space && space.getValue() == mlir::gpu::AddressSpace::Workgroup;
    };

    // In full-scope logical-block mode the third operand is deliberately the
    // logical global B source.  It may be K=[1,T,4,128] or H=[1,C,8,128,128]
    // and is not required to be the legacy direct-K source.  Keeping this
    // check here makes the representation target-independent while the late
    // planner owns the actual packet and shared placement.
    auto getShape = [](mlir::Type type) -> llvm::ArrayRef<int64_t> {
        if (auto aveType = mlir::dyn_cast<cf::MemRefType>(type))
            return aveType.getShape();
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type))
            return memrefType.getShape();
        return {};
    };
    auto getElementType = [](mlir::Type type) -> mlir::Type {
        if (auto aveType = mlir::dyn_cast<cf::MemRefType>(type))
            return aveType.getElementType();
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type))
            return memrefType.getElementType();
        return {};
    };
    auto isLogicalGlobalBlock = [&](mlir::Value value) {
        auto shape = getShape(value.getType());
        if (shape.empty() || !getElementType(value.getType()).isBF16())
            return false;
        if (shape.size() == 4) {
            return shape[0] == 1 && shape[1] >= 64 && shape[2] == 4 &&
                   shape[3] == 128;
        }
        if (shape.size() == 5) {
            return shape[0] == 1 && shape[1] >= 1 && shape[2] == 8 &&
                   shape[3] == 128 && shape[4] == 128;
        }
        return false;
    };
    const bool sameSource = resolved_args[2] == resolved_args[3];
    auto sourceShape = getShape(resolved_args[2].getType());
    auto sourceElementType = getElementType(resolved_args[2].getType());
    const bool c18VSource =
        UseC18FullPhysicalRegion() && sourceShape.size() == 4 &&
        sourceShape[2] == 8;
    const bool c19QSource =
        UseC19CompatibleFullPhysicalRegion() && sameSource && sourceShape.size() == 4 &&
        sourceShape[2] == 4;
    const bool c19VSource =
        UseC19CompatibleFullPhysicalRegion() && sourceShape.size() == 4 &&
        sourceShape[2] == 8;
    const bool c22VSource =
        UseC22SchedulePreservingPhysical() && sourceShape.size() == 4 &&
        sourceShape[2] == 8;
    if (c19QSource &&
        (sourceShape[0] != 1 || sourceShape[1] < 64 ||
         sourceShape[3] != 128 || !sourceElementType.isBF16())) {
        report("C19 Q logical block-dot source must be BF16 [1,T,4,128]");
        return false;
    }
    if (c18VSource || c19VSource || c22VSource) {
        if (sourceShape.size() != 4 || sourceShape[0] != 1 ||
            sourceShape[1] < 64 || sourceShape[2] != 8 ||
            sourceShape[3] != 128 || !sourceElementType.isBF16()) {
            report("V logical block-dot source must be BF16 [1,T,8,128]");
            return false;
        }
    }
    if (auto c16Role = C16RealTileRole();
        c16Role && !(isOperandShared(resolved_args[0]) &&
                     isOperandShared(resolved_args[1]))) {
        report("C16 Q/H/K role is enabled, but block-dot A/B operands are "
               "not rank-2 BF16 workgroup stages");
        return false;
    }
    if (allow_operand_mode && isOperandShared(resolved_args[0]) &&
        isOperandShared(resolved_args[1])) {
        auto c15Source = mlir::dyn_cast<cf::MemRefType>(resolved_args[2].getType());
        const bool c15RealV =
            UseC15RealTile() && c15Source && c15Source.getRank() == 2 &&
            c15Source.getShape() == llvm::ArrayRef<int64_t>({64, 64}) &&
            c15Source.getElementType().isBF16();
        if (c15RealV)
            return true;
        if (auto c16Role = C16RealTileRole()) {
            const std::array<int64_t, 2> expectedShapeStorage =
                *c16Role == "K" ? std::array<int64_t, 2>{32, 64}
                                 : std::array<int64_t, 2>{64, 32};
            const llvm::ArrayRef<int64_t> expectedShape(expectedShapeStorage);
            const bool c16RealTile =
                c15Source && c15Source.getRank() == 2 &&
                c15Source.getShape() == expectedShape &&
                c15Source.getElementType().isBF16();
            if (c16RealTile)
                return true;
            report("C16 Q/H/K real-tile role requires a BF16 source with "
                   "shape Q/H=[64,32] or K=[32,64]");
            return false;
        }
        if (!isLogicalGlobalBlock(resolved_args[2]) && !c18VSource &&
            !c19VSource && !c19QSource && !c22VSource) {
            report("full/logical block-dot operand mode expects the third "
                   "operand to be a BF16 global K or H logical block");
            return false;
        }
        for (auto value : resolved_args.slice(5, 6)) {
            if (!value.getType().isIntOrIndex()) {
                report("block-dot operand mode indices must be integer or index");
                return false;
            }
        }
        if (!resolved_args[11].getType().isF32()) {
            report("block-dot operand mode g_last must be f32");
            return false;
        }
        for (auto value : resolved_args.drop_front(12)) {
            auto vectorType = mlir::dyn_cast<mlir::VectorType>(value.getType());
            auto memrefType = mlir::dyn_cast<cf::MemRefType>(value.getType());
            auto isVector = vectorType && vectorType.getRank() == 1 &&
                            vectorType.getNumElements() == 16 &&
                            vectorType.getElementType().isF32();
            auto isLocal = memrefType && memrefType.getRank() == 1 &&
                           memrefType.getShape() ==
                               llvm::ArrayRef<int64_t>({16}) &&
                           memrefType.getElementType().isF32();
            if (!isVector && !isLocal) {
                report("block-dot operand mode accumulators must be "
                       "vector<16xf32> or local FP32 [16]");
                return false;
            }
        }
        return true;
    }
    if ((!isIndependentA && !isCooperativeA && !isStagedA) ||
        (!isRegularB && !isStagedB && !isPreloadedB) ||
        (isStagedA != (isStagedB || isPreloadedB))) {
        report("block_dot_bf16_f32 expects workgroup BF16 A=[2,32,32] or "
               "cooperative A=[1,32,32] and B=[32,32]; staged V-decay "
               "requires A=[1,32,64] and B=[64,64] or preloaded K=[2,64,64]");
        return false;
    }
    auto checkTensor = [&](mlir::Value value, llvm::ArrayRef<int64_t> shape,
                           mlir::Type element) {
        auto type = mlir::dyn_cast<cf::MemRefType>(value.getType());
        return type && type.getRank() == static_cast<int>(shape.size()) &&
               type.getShape()[0] == shape[0] &&
               type.getShape()[1] >= shape[1] &&
               type.getShape()[2] == shape[2] &&
               type.getShape()[3] == shape[3] &&
               type.getElementType() == element;
    };
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    if (!c19QSource && !c19VSource &&
        (!checkTensor(resolved_args[2], {1, 64, 4, 128},
                      builder.getBF16Type()) ||
         !checkTensor(resolved_args[3], {1, 64, 8, 128},
                      builder.getBF16Type()))) {
        report("block_dot_bf16_f32 expects direct BF16 K=[1,T,4,128] and "
               "V-new=[1,T,8,128]");
        return false;
    }
    auto gType = mlir::dyn_cast<cf::MemRefType>(resolved_args[4].getType());
    if (!gType || gType.getRank() != 3 || gType.getShape()[0] != 1 ||
        gType.getShape()[1] < 64 || gType.getShape()[2] != 8 ||
        !gType.getElementType().isF32()) {
        report("block_dot_bf16_f32 expects FP32 G=[1,T,8]");
        return false;
    }
    for (auto value : resolved_args.slice(5, 6)) {
        if (!value.getType().isIntOrIndex()) {
            report(
                "block_dot_bf16_f32 index operands must be integer or index");
            return false;
        }
    }
    if (!resolved_args[11].getType().isF32()) {
        report("block_dot_bf16_f32 g_last must be f32");
        return false;
    }
    for (auto value : resolved_args.drop_front(12)) {
        auto vectorType = mlir::dyn_cast<mlir::VectorType>(value.getType());
        auto memrefType = mlir::dyn_cast<cf::MemRefType>(value.getType());
        auto isVector = vectorType && vectorType.getRank() == 1 &&
                        vectorType.getNumElements() == 16 &&
                        vectorType.getElementType().isF32();
        auto isLocal = memrefType && memrefType.getRank() == 1 &&
                       memrefType.getShape() == llvm::ArrayRef<int64_t>({16}) &&
                       memrefType.getElementType().isF32();
        if (!isVector && !isLocal) {
            report("block_dot_bf16_f32 accumulators must be vector<16xf32> "
                   "or local FP32 [16] storage");
            return false;
        }
    }
    return true;
}

bool AMDGPUIntrinsic::CheckQwenGdnRecurrenceStepBF16F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
    };
    if (resolved_args.size() != 22) {
        report("qwen_gdn_recurrence_step_bf16_f32 requires 22 operands");
        return false;
    }
    for (auto value : resolved_args) {
        if (!value) {
            report(
                "qwen_gdn_recurrence_step_bf16_f32 failed to resolve operands");
            return false;
        }
    }
    auto isShared = [](mlir::Value value, llvm::ArrayRef<int64_t> shape,
                       mlir::Type element) {
        auto type = mlir::dyn_cast<cf::MemRefType>(value.getType());
        auto space = type ? mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
                                type.getMemorySpace())
                          : nullptr;
        return type && type.getShape() == shape &&
               type.getElementType() == element && space &&
               space.getValue() == mlir::gpu::AddressSpace::Workgroup;
    };
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    if (!isShared(resolved_args[0], {2, 32, 64}, builder.getBF16Type()) ||
        !isShared(resolved_args[1], {64, 64}, builder.getBF16Type()) ||
        !isShared(resolved_args[2], {2, 32, 32}, builder.getF32Type()) ||
        !isShared(resolved_args[3], {1, 32, 32}, builder.getBF16Type())) {
        report("recurrence step requires state=[2,32,64], phase=[64,64], "
               "pred=[2,32,32] and vdecay=[1,32,32] workgroup buffers");
        return false;
    }
    auto vector16F32 = [&](mlir::Value value) {
        auto type = mlir::dyn_cast<mlir::VectorType>(value.getType());
        return type && type.getRank() == 1 && type.getNumElements() == 16 &&
               type.getElementType().isF32();
    };
    auto local16F32 = [&](mlir::Value value) {
        auto type = mlir::dyn_cast<cf::MemRefType>(value.getType());
        return type && type.getRank() == 1 &&
               type.getShape() == llvm::ArrayRef<int64_t>({16}) &&
               type.getElementType().isF32();
    };
    if ((!vector16F32(resolved_args[19]) && !local16F32(resolved_args[19])) ||
        (!vector16F32(resolved_args[20]) && !local16F32(resolved_args[20]))) {
        report(
            "recurrence step persistent state operands must be vector<16xf32> "
            "or local FP32 [16] storage");
        return false;
    }
    if (!resolved_args[14].getType().isIntOrIndex() ||
        !resolved_args[15].getType().isIntOrIndex() ||
        !resolved_args[16].getType().isIntOrIndex() ||
        !resolved_args[17].getType().isIntOrIndex() ||
        !resolved_args[18].getType().isIntOrIndex()) {
        report("recurrence step thread/chunk/head/index operands must be "
               "integer or index");
        return false;
    }
    if (!resolved_args[21].getType().isInteger(1)) {
        report("recurrence step emit_audit must be i1");
        return false;
    }
    return true;
}

bool AMDGPUIntrinsic::CheckQwenPersistentRecurrenceBeginFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto report = [&](llvm::StringRef message) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
    };
    if (resolved_args.size() != 14) {
        report("qwen_persistent_recurrence_begin requires 14 operands: "
               "K, W, U, G, initial_state, H, pred_f32, pred_bf16, V-new, "
               "V-decay, state_after, final_state, num_chunks, emit_audit");
        return false;
    }
    for (auto value : resolved_args) {
        if (!value) {
            report("qwen_persistent_recurrence_begin failed to resolve operands");
            return false;
        }
    }
    if (!resolved_args[12].getType().isIntOrIndex()) {
        report("qwen_persistent_recurrence_begin num_chunks must be integer or index");
        return false;
    }
    if (!resolved_args[13].getType().isInteger(1)) {
        report("qwen_persistent_recurrence_begin emit_audit must be i1");
        return false;
    }
    return true;
}

bool AMDGPUIntrinsic::CheckQwenPersistentRecurrenceEndFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "qwen_persistent_recurrence_end takes no operands";
        return false;
    }
    return true;
}

bool AMDGPUIntrinsic::CheckGenericRawBufferStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int width) const {
    if (resolved_args.size() != 5) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_store_x" << width
            << " requires exactly 5 arguments: vdata, rsrc, vindex, soffset, "
               "aux";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3] || !resolved_args[4]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for raw_buffer_store_x" << width;
        return false;
    }

    if (width != 1 && width != 2 && width != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Unsupported width for raw_buffer_store_x" << width
            << " (supported: 1, 2, 4)";
        return false;
    }

    auto rsrc_type = resolved_args[1].getType();
    auto rsrc_vector = mlir::dyn_cast<mlir::VectorType>(rsrc_type);
    if (!rsrc_vector || rsrc_vector.getNumElements() != 4 ||
        !rsrc_vector.getElementType().isInteger(32)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_store_x" << width
            << " expects rsrc to be vector<4xi32>";
        return false;
    }

    auto vdata_type = resolved_args[0].getType();
    if (width == 1) {
        if (!mlir::isa<mlir::IntegerType>(vdata_type)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "raw_buffer_store_x1 expects i32 data type";
            return false;
        }
    } else {
        auto vector_type = mlir::dyn_cast<mlir::VectorType>(vdata_type);
        if (!vector_type || vector_type.getNumElements() != width ||
            !mlir::isa<mlir::IntegerType>(vector_type.getElementType())) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "raw_buffer_store_x" << width << " expects vector<" << width
                << "xi32> data type";
            return false;
        }
    }

    return true;
}

bool AMDGPUIntrinsic::CheckMakeRsrcFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 2 || resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_rsrc() requires exactly 2 arguments: tensor, range";
        return false;
    }

    auto tensor = resolved_args[0];
    auto range = resolved_args[1];
    if (!tensor) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve tensor argument for make_rsrc()";
        return false;
    }
    if (!range) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve range argument for make_rsrc()";
        return false;
    }

    auto memrefType = mlir::dyn_cast<cf::MemRefType>(tensor.getType());
    if (!memrefType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_rsrc() expects a tensor argument";
        return false;
    }

    if (!range.getType().isIndex() &&
        !mlir::isa<mlir::IntegerType>(range.getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_rsrc() expects range to be an integer or index value";
        return false;
    }

    auto rangeValue = ConstantFolder::FoldIntValue(range);
    if (rangeValue &&
        (*rangeValue < 0 ||
         static_cast<uint64_t>(*rangeValue) >
             static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_rsrc() requires range to be in [0, 2^32-1]";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckRcpFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 1 || resolved_args.size() != 1 ||
        !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "rcp() requires exactly 1 argument";
        return false;
    }

    if (!resolved_args[0].getType().isF32()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "rcp() expects an f32 argument";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckSWaitcntFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "s_waitcnt requires exactly 3 arguments: vmcnt, expcnt, "
               "lgkmcnt";
        return false;
    }

    for (auto value : resolved_args) {
        if (!value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate operands for s_waitcnt";
            return false;
        }
        if (!value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "s_waitcnt operands must be integer or index types";
            return false;
        }
    }

    auto vmcnt = ConstantFolder::FoldIntValue(resolved_args[0]);
    auto expcnt = ConstantFolder::FoldIntValue(resolved_args[1]);
    auto lgkmcnt = ConstantFolder::FoldIntValue(resolved_args[2]);
    auto is_valid = [](std::optional<int64_t> value, int64_t max) {
        return value && *value >= 0 && *value <= max;
    };
    if (!is_valid(vmcnt, 63) || !is_valid(expcnt, 7) ||
        !is_valid(lgkmcnt, 15)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "s_waitcnt(vmcnt, expcnt, lgkmcnt) requires compile-time "
               "integer arguments in ranges vmcnt=[0,63], expcnt=[0,7], "
               "lgkmcnt=[0,15]";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckPermFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 3 || resolved_args.size() != 3 ||
        !resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "perm() requires exactly 3 arguments";
        return false;
    }

    for (auto arg : resolved_args) {
        auto intType = mlir::dyn_cast<mlir::IntegerType>(arg.getType());
        if (!intType || intType.getWidth() != 32) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "perm() expects 32-bit integer arguments";
            return false;
        }
    }

    return true;
}

bool AMDGPUIntrinsic::CheckSchedGroupBarrierFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 3 || resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "sched_group_barrier() requires exactly 3 arguments: "
               "mask, size, group_id";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for sched_group_barrier()";
        return false;
    }

    return true;
}

// Factory function to create AMDGPU intrinsic module
std::unique_ptr<NamedModule> CreateAMDGPUIntrinsicModule() {
    return std::make_unique<AMDGPUIntrinsic>();
}

} // namespace causalflow::avelang::ir
