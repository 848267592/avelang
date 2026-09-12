//===- c14_static_physical_codegen_test.cc - C14 synthetic codegen -------===//
//
// This is a compile/codegen repro, not a Qwen performance test.  It creates
// four internal block-dot consumers carrying the C13 typed physical recipe,
// then runs the ordinary Avelang -> AMDGPU pipeline.  The test deliberately
// does not launch the generated functions: the stage operands are synthetic
// workgroup tiles and the C14 gate is about representation consumption and
// machine formation, not numerical chunk-o behavior.
//
//===----------------------------------------------------------------------===//

#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Dialect/AveLang/IR/static_physical_layout.h"
#include "IR/ir_context.h"
#include "IR/Intrinsics/intrinsic_support.h"
#include "Target/GPU/lower_to_llvm.h"
#include "amdgpu_backend.h"

#include <gtest/gtest.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>

#include <array>
#include <set>
#include <string>

using namespace causalflow::avelang::dialect;
using namespace causalflow::avelang::dialect::c13;
using namespace causalflow::avelang::target::gpu;
using namespace causalflow::avelang::target::amdgpu;

namespace {

struct C14Artifacts {
    std::string directory;

    void write(llvm::StringRef name, llvm::StringRef contents) const {
        if (directory.empty())
            return;
        llvm::SmallString<256> path(directory);
        llvm::sys::path::append(path, name);
        std::error_code ec;
        llvm::raw_fd_ostream output(path, ec);
        if (!ec)
            output << contents;
    }

    void writeBinary(llvm::StringRef name, llvm::StringRef contents) const {
        if (directory.empty())
            return;
        llvm::SmallString<256> path(directory);
        llvm::sys::path::append(path, name);
        std::error_code ec;
        llvm::raw_fd_ostream output(path, ec,
                                    llvm::sys::fs::OF_None);
        if (!ec)
            output.write(contents.data(), contents.size());
    }
};

mlir::func::FuncOp makeSyntheticFunction(
    mlir::ModuleOp module, mlir::OpBuilder &builder, llvm::StringRef name,
    const PhysicalBlockPlan &physical, mlir::Type stageType,
    mlir::Type accumulatorType, mlir::Type outputType) {
    auto functionType = builder.getFunctionType(
        {stageType, stageType, accumulatorType, outputType}, {});
    // Create detached, then attach exactly once to the module.  Using the
    // builder overload here would insert a later synthetic function into the
    // previous function's entry block and corrupt the module when four paths
    // are assembled.
    auto function = mlir::func::FuncOp::create(builder.getUnknownLoc(), name,
                                               functionType);
    function->setAttr("ave.gpu_func", builder.getI32IntegerAttr(2));
    function->setAttr("c14.synthetic_role",
                      builder.getStringAttr(name));
    auto *entry = function.addEntryBlock();
    builder.setInsertionPointToEnd(entry);

    auto loc = builder.getUnknownLoc();
    auto zero = mlir::arith::ConstantOp::create(
        builder, loc, builder.getF32Type(), builder.getF32FloatAttr(0.0F));
    auto accumulator = mlir::vector::SplatOp::create(
        builder, loc, mlir::cast<mlir::VectorType>(accumulatorType), zero);
    auto indexZero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    auto operation = AMDGPUBlockDotMfmaOperandOp::create(
        builder, loc, accumulatorType, entry->getArgument(0),
        entry->getArgument(1), accumulator.getResult(), indexZero, indexZero,
        indexZero, indexZero);

    // The C14 selector is the only non-typed control bit.  All physical
    // mapping facts are carried by the five typed C13 attributes below.
    operation->setAttr("c14.static_physical", builder.getUnitAttr());
    operation->setAttr(
        "c14.mfma_callee",
        builder.getStringAttr(
            causalflow::avelang::ir::intrinsics::MakeIntrinsicFuncName(
                "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k")));
    operation->setAttr("c13.distributed",
                       physical.distributed.toAttr(module.getContext()));
    operation->setAttr("c13.shared",
                       physical.shared.toAttr(module.getContext()));
    operation->setAttr("c13.mfma",
                       physical.dot.parent.toAttr(module.getContext()));
    operation->setAttr("c13.dot",
                       physical.dot.toAttr(module.getContext()));
    operation->setAttr("c13.transform",
                       physical.transform.toAttr(module.getContext()));
    operation->setAttr("c14.path_kind",
                       builder.getStringAttr(name));
    operation->setAttr("c14.structured_input",
                       builder.getStringAttr(
                           "one_hot,lane_distinct,wave_distinct,packet_distinct,row_column"));

    mlir::vector::StoreOp::create(builder, loc, operation.getResult(),
                                  entry->getArgument(3),
                                  mlir::ValueRange{indexZero});
    mlir::func::ReturnOp::create(builder, loc);
    module.push_back(function);
    return function;
}

TEST(C14StaticPhysicalCodegenTest, SyntheticQHKVPathsReachAMDGPU) {
    auto irContext = causalflow::avelang::ir::IRContext::Create();
    auto *context = irContext->GetMLIRContext();
    context->loadDialect<AveLangDialect, mlir::func::FuncDialect,
                         mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                         mlir::vector::VectorDialect>();
    mlir::OpBuilder builder(context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto plan = ChunkOPhysicalPlan::makeC12T2048WG256();
    std::string error;
    ASSERT_TRUE(plan.verify(&error)) << error;

    // Structured tile verifier: the same ownership/transform/shared/dot
    // algebra is exercised by one-hot, lane, wave, packet and row/column
    // patterns before codegen.  This proves the synthetic path has a
    // deterministic reconstruction oracle without pretending to launch it.
    const auto &v = plan.v;
    std::set<LogicalCoord> recovered;
    for (int64_t wave = 0; wave < 2; ++wave) {
        for (int64_t lane = 0; lane < 64; ++lane) {
            for (int64_t r0 = 0; r0 < 4; ++r0) {
                for (int64_t r1 = 0; r1 < 8; ++r1) {
                    HardwareCoord owner{wave, lane, {r0, r1}};
                    auto logical = v.distributed.mapHardwareToLogical(owner);
                    ASSERT_TRUE(logical.has_value());
                    ASSERT_TRUE(recovered.insert(*logical).second);
                    auto owners = v.distributed.getOwners(*logical);
                    ASSERT_EQ(owners.size(), 1u);
                    auto reconstructed =
                        v.distributed.mapHardwareToLogical(owners.front());
                    ASSERT_TRUE(reconstructed.has_value());
                    EXPECT_EQ(*reconstructed, *logical);
                    auto transformed = v.transform.apply(*logical);
                    ASSERT_TRUE(transformed.has_value());
                    EXPECT_TRUE(v.shared
                                    .getSharedByteOffset(*transformed, {64, 64}, 2)
                                    .has_value());
                    EXPECT_TRUE(v.dot.getOperandRegisterSlot(*logical)
                                    .has_value());

                    const auto pattern = [](int kind, LogicalCoord value,
                                            HardwareCoord patternOwner) -> int64_t {
                        switch (kind) {
                        case 0: // one-hot
                            return value == LogicalCoord{{17, 29}} ? 1 : 0;
                        case 1: // lane-distinct
                            return patternOwner.lane;
                        case 2: // wave-distinct
                            return patternOwner.wave;
                        case 3: // packet/register-distinct
                            return patternOwner.reg[0] + 4 * patternOwner.reg[1];
                        default: // row/column
                            return value[0] * 64 + value[1];
                        }
                    };
                    for (int kind = 0; kind < 5; ++kind)
                        EXPECT_EQ(pattern(kind, *reconstructed, owners.front()),
                                  pattern(kind, *logical, owner));
                }
            }
        }
    }
    EXPECT_EQ(recovered.size(), 64u * 64u);

    auto bf16 = builder.getBF16Type();
    auto f32 = builder.getF32Type();
    auto workgroup = builder.getI64IntegerAttr(3);
    auto stageType = mlir::MemRefType::get(
        {64, 64}, bf16, mlir::MemRefLayoutAttrInterface(), workgroup);
    auto accumulatorType = mlir::VectorType::get({16}, f32);
    auto outputType = mlir::MemRefType::get({16}, f32);
    makeSyntheticFunction(module, builder, "c14_q", plan.q, stageType,
                          accumulatorType, outputType);
    makeSyntheticFunction(module, builder, "c14_h", plan.h, stageType,
                          accumulatorType, outputType);
    makeSyntheticFunction(module, builder, "c14_k", plan.k, stageType,
                          accumulatorType, outputType);
    makeSyntheticFunction(module, builder, "c14_v", plan.v, stageType,
                          accumulatorType, outputType);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module))) << module;

    C14Artifacts artifacts;
    if (auto value = llvm::sys::Process::GetEnv(
            "AVELANG_C14_SYNTHETIC_OUTPUT_DIR")) {
        artifacts.directory = *value;
        std::error_code ec = llvm::sys::fs::create_directories(
            artifacts.directory);
        ASSERT_FALSE(ec) << ec.message();
    }
    std::string preC14;
    llvm::raw_string_ostream preStream(preC14);
    module.print(preStream);
    artifacts.write("pre_c14.mlir", preC14);

    causalflow::avelang::target::gpu::GPUCompilationOptions options;
    options.triple = "amdgcn-amd-amdhsa";
    options.chipset = "gfx942";
    options.optimization_level = 3;
    AMDGPUBackend backend;
    backend.EnsureInitialized();
    llvm::LLVMContext llvmContext;
    causalflow::avelang::target::gpu::LowerToLLVM compiler(irContext.get());
    auto llvmModule = compiler.compile(module, llvmContext, options);
    ASSERT_NE(llvmModule, nullptr);
    std::string postC14;
    llvm::raw_string_ostream postStream(postC14);
    module.print(postStream);
    artifacts.write("post_c14.mlir", postC14);
    std::string llvmText;
    llvm::raw_string_ostream llvmStream(llvmText);
    llvmModule->print(llvmStream, nullptr);
    artifacts.write("post_c14_llvm.ll", llvmText);

    auto assembly = backend.generateAssembly(*llvmModule, options);
    ASSERT_TRUE(static_cast<bool>(assembly))
        << llvm::toString(assembly.takeError());
    artifacts.write("final_c14_isa.s", *assembly);
    EXPECT_NE(assembly->find("mfma"), std::string::npos);
    EXPECT_EQ(assembly->find("ds_bpermute"), std::string::npos);
    EXPECT_EQ(assembly->find("s_div_"), std::string::npos);
    EXPECT_EQ(assembly->find("s_rem_"), std::string::npos);

    auto binary = backend.generateBinary(*llvmModule, options);
    if (binary) {
        artifacts.writeBinary("c14_synthetic.hsaco", *binary);
    } else {
        auto message = llvm::toString(binary.takeError());
        artifacts.write("hsaco_generation_error.txt", message);
        // Assembly is the mandatory codegen proof.  Linker availability is an
        // environment capability and must not turn a valid static lowering
        // test into a false representation failure.
        SUCCEED() << message;
    }
}

} // namespace
