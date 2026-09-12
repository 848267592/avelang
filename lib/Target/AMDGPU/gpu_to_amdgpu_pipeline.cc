#include "gpu_to_amdgpu_pipeline.h"
#include "Dialect/AveLang/Transforms/normalize_ave_lang_return_pass.h"
#include "legalize_gpu_shuffle_to_idx_pass.h"
#include "lower_math_to_amdgpu_pass.h"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/AMDGPU/Utils/Chipset.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#define GEN_PASS_DECL_EXPANDSTRIDEDMETADATAPASS
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#undef GEN_PASS_DECL_EXPANDSTRIDEDMETADATAPASS
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>

#include <map>
#include <optional>
#include <sstream>
#include <utility>

namespace causalflow::avelang::target::amdgpu {

using namespace mlir;

namespace {

// Experiment 0.5 uses these snapshots only to bisect where two explicitly
// different AveLang/MLIR branches first converge. They are entirely inert in
// ordinary compilation and intentionally live beside the relevant AMDGPU
// pipeline rather than changing its scheduling or legality.
bool isQwenKFragConvergenceAuditEnabled() {
    return llvm::sys::Process::GetEnv(
               "AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT") ==
           std::optional<std::string>("1");
}

void writeQwenKFragConvergenceSnapshot(mlir::Operation *op,
                                        llvm::StringRef phase) {
    if (!isQwenKFragConvergenceAuditEnabled()) {
        return;
    }
    const auto dumpDir = llvm::sys::Process::GetEnv(
        "AVELANG_QWEN_KFRAG_AB_DUMP_DIR");
    if (!dumpDir) {
        return;
    }
    std::error_code ec = llvm::sys::fs::create_directories(*dumpDir);
    if (ec) {
        llvm::errs() << "[qwen-kfrag-convergence] cannot create " << *dumpDir
                     << ": " << ec.message() << "\n";
        return;
    }
    llvm::SmallString<256> path(*dumpDir);
    llvm::sys::path::append(path, phase + ".mlir");
    std::error_code writeEc;
    llvm::raw_fd_ostream stream(path, writeEc);
    if (writeEc) {
        llvm::errs() << "[qwen-kfrag-convergence] cannot write " << path
                     << ": " << writeEc.message() << "\n";
        return;
    }
    op->print(stream);
}

class QwenKFragModuleConvergenceSnapshotPass
    : public PassWrapper<QwenKFragModuleConvergenceSnapshotPass,
                         OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        QwenKFragModuleConvergenceSnapshotPass)

    explicit QwenKFragModuleConvergenceSnapshotPass(std::string phase)
        : phase_(std::move(phase)) {}

    void runOnOperation() override {
        writeQwenKFragConvergenceSnapshot(getOperation(), phase_);
    }

  private:
    std::string phase_;
};

class QwenKFragGPUConvergenceSnapshotPass
    : public PassWrapper<QwenKFragGPUConvergenceSnapshotPass,
                         OperationPass<gpu::GPUModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        QwenKFragGPUConvergenceSnapshotPass)

    explicit QwenKFragGPUConvergenceSnapshotPass(std::string phase)
        : phase_(std::move(phase)) {}

    void runOnOperation() override {
        writeQwenKFragConvergenceSnapshot(getOperation(), phase_);
    }

  private:
    std::string phase_;
};

std::unique_ptr<Pass>
createQwenKFragModuleConvergenceSnapshotPass(llvm::StringRef phase) {
    return std::make_unique<QwenKFragModuleConvergenceSnapshotPass>(
        phase.str());
}

std::unique_ptr<Pass>
createQwenKFragGPUConvergenceSnapshotPass(llvm::StringRef phase) {
    return std::make_unique<QwenKFragGPUConvergenceSnapshotPass>(
        phase.str());
}

class OverrideRocdlMaxFlatWorkgroupSizePass
    : public PassWrapper<OverrideRocdlMaxFlatWorkgroupSizePass,
                         OperationPass<gpu::GPUModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        OverrideRocdlMaxFlatWorkgroupSizePass)

    OverrideRocdlMaxFlatWorkgroupSizePass(int maxFlatWorkgroupSize)
        : maxFlatWorkgroupSize(maxFlatWorkgroupSize) {}

    void runOnOperation() override {
        if (maxFlatWorkgroupSize <= 0) {
            return;
        }

        gpu::GPUModuleOp gpuModule = getOperation();
        MLIRContext *context = gpuModule.getContext();
        auto *rocdlDialect = context->getOrLoadDialect<ROCDL::ROCDLDialect>();
        auto maxFlatWorkgroupSizeAttr =
            rocdlDialect->getMaxFlatWorkGroupSizeAttrHelper();
        Builder builder(context);
        IntegerAttr attr = builder.getI32IntegerAttr(maxFlatWorkgroupSize);

        gpuModule.walk([&](gpu::GPUFuncOp func) {
            if (func.isKernel()) {
                maxFlatWorkgroupSizeAttr.setAttr(func, attr);
            }
        });
    }

  private:
    int maxFlatWorkgroupSize;
};

static std::unique_ptr<Pass>
createOverrideRocdlMaxFlatWorkgroupSizePass(int maxFlatWorkgroupSize) {
    return std::make_unique<OverrideRocdlMaxFlatWorkgroupSizePass>(
        maxFlatWorkgroupSize);
}

} // namespace

static void
buildCommonPassPipeline(OpPassManager &pm,
                        const AMDGPUToLLVMPipelineOptions &options) {
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addPass(createQwenKFragModuleConvergenceSnapshotPass(
            "amdgpu_00_pre_common"));
    }
    pm.addPass(bufferization::createOneShotBufferizePass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addPass(createQwenKFragModuleConvergenceSnapshotPass(
            "amdgpu_01_post_one_shot_bufferize"));
    }
    pm.addPass(memref::createExpandStridedMetadataPass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addPass(createQwenKFragModuleConvergenceSnapshotPass(
            "amdgpu_02_post_expand_strided_metadata"));
    }
    pm.addPass(
        causalflow::avelang::dialect::createNormalizeAveLangReturnPass());
    pm.addPass(createSCFToControlFlowPass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addPass(createQwenKFragModuleConvergenceSnapshotPass(
            "amdgpu_03_post_scf_to_cf"));
    }
    pm.addPass(affine::createAffineExpandIndexOpsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addPass(createQwenKFragModuleConvergenceSnapshotPass(
            "amdgpu_04_post_common_cleanup"));
    }

    // Reconcile unrealized casts at the end to resolve any remaining type
    // conversion issues
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createCanonicalizerPass());
}

/// Build the GPU pass pipeline for GPU module-specific transformations.
static void buildGpuPassPipeline(OpPassManager &pm,
                                 const AMDGPUToLLVMPipelineOptions &options) {
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addNestedPass<gpu::GPUModuleOp>(
            createQwenKFragGPUConvergenceSnapshotPass(
                "amdgpu_10_pre_gpu_pipeline"));
    }
    // Parse target features to configure GPU passes
    auto parseFeatures =
        [](const std::string &featuresStr) -> std::map<std::string, bool> {
        std::map<std::string, bool> features;
        if (featuresStr.empty())
            return features;

        std::stringstream ss(featuresStr);
        std::string feature;
        while (std::getline(ss, feature, ',')) {
            feature.erase(0, feature.find_first_not_of(" \t"));
            feature.erase(feature.find_last_not_of(" \t") + 1);
            if (feature.empty())
                continue;

            if (feature[0] == '+') {
                features[feature.substr(1)] = true;
            } else if (feature[0] == '-') {
                features[feature.substr(1)] = false;
            }
        }
        return features;
    };

    auto features = parseFeatures(options.target_features);

    // Set wave64 flag based on target features, defaulting to true for backward
    // compatibility
    bool wave64Flag = true; // Default
    if (features.count("wavefrontsize64")) {
        wave64Flag = features["wavefrontsize64"];
    }

    GpuROCDLAttachTargetOptions rocdlOptions;
    rocdlOptions.chip = options.chipset;
    rocdlOptions.triple = options.triple;
    rocdlOptions.optLevel = options.optimization_level;
    rocdlOptions.wave64Flag = wave64Flag;
    pm.addPass(createGpuROCDLAttachTarget(rocdlOptions));
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addNestedPass<gpu::GPUModuleOp>(
            createQwenKFragGPUConvergenceSnapshotPass(
                "amdgpu_11_post_rocdl_attach"));
    }
    if (options.num_warps > 0) {
        const int waveSize = wave64Flag ? 64 : 32;
        pm.addNestedPass<gpu::GPUModuleOp>(
            createOverrideRocdlMaxFlatWorkgroupSizePass(options.num_warps *
                                                        waveSize));
    }
    if (options.optimization_level > 0) {
        pm.addNestedPass<gpu::GPUModuleOp>(createLowerMathToAMDGPUPass());
        if (isQwenKFragConvergenceAuditEnabled()) {
            pm.addNestedPass<gpu::GPUModuleOp>(
                createQwenKFragGPUConvergenceSnapshotPass(
                    "amdgpu_12_post_lower_math"));
        }
    }
    pm.addNestedPass<gpu::GPUModuleOp>(createLegalizeGPUShuffleToIDXPass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addNestedPass<gpu::GPUModuleOp>(
            createQwenKFragGPUConvergenceSnapshotPass(
                "amdgpu_13_post_legalize_shuffle"));
    }

    ConvertGpuOpsToROCDLOpsOptions gpuToRocdlOptions;
    gpuToRocdlOptions.chipset = options.chipset;
    gpuToRocdlOptions.indexBitwidth = kDeriveIndexBitwidthFromDataLayout;
    gpuToRocdlOptions.useBarePtrCallConv =
        options.use_bare_ptr_memref_call_conv;
    gpuToRocdlOptions.runtime = gpu::amd::Runtime::HIP;
    pm.addNestedPass<gpu::GPUModuleOp>(
        createConvertGpuOpsToROCDLOps(std::move(gpuToRocdlOptions)));
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addNestedPass<gpu::GPUModuleOp>(
            createQwenKFragGPUConvergenceSnapshotPass(
                "amdgpu_14_post_gpu_to_rocdl"));
    }

    pm.addNestedPass<gpu::GPUModuleOp>(createConvertAMDGPUToROCDLPass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addNestedPass<gpu::GPUModuleOp>(
            createQwenKFragGPUConvergenceSnapshotPass(
                "amdgpu_15_post_amdgpu_to_rocdl"));
    }

    pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToLLVMPass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addNestedPass<gpu::GPUModuleOp>(
            createQwenKFragGPUConvergenceSnapshotPass(
                "amdgpu_16_post_vector_to_llvm"));
    }
    pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addNestedPass<gpu::GPUModuleOp>(
            createQwenKFragGPUConvergenceSnapshotPass(
                "amdgpu_17_post_gpu_cleanup"));
    }
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToLLVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createArithToLLVMConversionPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertIndexToLLVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createUBToLLVMConversionPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addNestedPass<gpu::GPUModuleOp>(
            createQwenKFragGPUConvergenceSnapshotPass(
                "amdgpu_18_post_scalar_to_llvm"));
    }

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    // Add final canonicalization to clean up after GPU conversion
    pm.addPass(createCanonicalizerPass());
    if (isQwenKFragConvergenceAuditEnabled()) {
        pm.addPass(createQwenKFragModuleConvergenceSnapshotPass(
            "amdgpu_19_post_gpu_pipeline"));
    }
}

void BuildLowerToAMDGPUPassPipeline(
    OpPassManager &pm, const AMDGPUToLLVMPipelineOptions &options) {
    buildCommonPassPipeline(pm, options);
    buildGpuPassPipeline(pm, options);
}

} // namespace causalflow::avelang::target::amdgpu
