//===- c24_region_pending_packet_codegen_test.cc ------------------------===//
//
// A target-only synthetic proof for C24 pending packet materialization.  It
// constructs one global BF16x8 IssuePacket, three independent current MFMA32
// operations, then a CommitPacket into workgroup LDS.  The test does not
// launch code.  Its gate is final ISA order: load -> multiple MFMA -> wait ->
// LDS store.
//
//===----------------------------------------------------------------------===//

#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Dialect/AveLang/IR/static_physical_layout.h"
#include "IR/ir_context.h"
#include "IR/Intrinsics/intrinsic_support.h"
#include "Target/GPU/lower_to_llvm.h"
#include "amdgpu_backend.h"

#include <gtest/gtest.h>

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

#include <string>

using namespace causalflow::avelang::dialect;
using namespace causalflow::avelang::dialect::c13;
using namespace causalflow::avelang::target::amdgpu;
using namespace causalflow::avelang::target::gpu;

namespace {

struct C24Artifacts {
    std::string directory;

    void write(llvm::StringRef name, llvm::StringRef contents) const {
        if (directory.empty())
            return;
        llvm::SmallString<256> path(directory);
        llvm::sys::path::append(path, name);
        std::error_code ec;
        llvm::raw_fd_ostream stream(path, ec);
        if (!ec)
            stream << contents;
    }
};

AMDGPUBlockDotMfmaOperandOp makeCurrentMfma(
    mlir::OpBuilder &builder, mlir::Location loc, const PhysicalBlockPlan &plan,
    mlir::Value stageA, mlir::Value stageB, mlir::Value accumulator,
    mlir::Value zero) {
    auto current = AMDGPUBlockDotMfmaOperandOp::create(
        builder, loc, accumulator.getType(), stageA, stageB, accumulator, zero,
        zero, zero, zero);
    current->setAttr("c14.static_physical", builder.getUnitAttr());
    current->setAttr(
        "c14.mfma_callee",
        builder.getStringAttr(
            causalflow::avelang::ir::intrinsics::MakeIntrinsicFuncName(
                "amdgpu", "rocdl_mfma_f32_32x32x8bf16_1k")));
    current->setAttr("c13.distributed", plan.distributed.toAttr(builder.getContext()));
    current->setAttr("c13.shared", plan.shared.toAttr(builder.getContext()));
    current->setAttr("c13.mfma", plan.dot.parent.toAttr(builder.getContext()));
    current->setAttr("c13.dot", plan.dot.toAttr(builder.getContext()));
    current->setAttr("c13.transform", plan.transform.toAttr(builder.getContext()));
    current->setAttr("c24.synthetic.current_work",
                     builder.getStringAttr("independent_mfma32"));
    return current;
}

size_t firstOf(llvm::StringRef text, llvm::StringRef first,
               llvm::StringRef second) {
    const auto a = text.find(first);
    const auto b = text.find(second);
    if (a == llvm::StringRef::npos)
        return b;
    if (b == llvm::StringRef::npos)
        return a;
    return std::min(a, b);
}

TEST(C24RegionPendingPacketCodegenTest, SyntheticIssueMfmaCommitSurvives) {
    auto irContext = causalflow::avelang::ir::IRContext::Create();
    auto *context = irContext->GetMLIRContext();
    context->loadDialect<AveLangDialect, mlir::func::FuncDialect,
                         mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                         mlir::vector::VectorDialect>();
    mlir::OpBuilder builder(context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto bf16 = builder.getBF16Type();
    auto f32 = builder.getF32Type();
    auto globalK = mlir::MemRefType::get({1, 64, 4, 128}, bf16);
    auto workgroup = builder.getI64IntegerAttr(3);
    auto stage = mlir::MemRefType::get(
        {64, 64}, bf16, mlir::MemRefLayoutAttrInterface(), workgroup);
    auto output = mlir::MemRefType::get({16}, f32);
    auto accumulatorType = mlir::VectorType::get({16}, f32);
    auto functionType = builder.getFunctionType(
        {globalK, stage, stage, stage, output}, {});
    auto function = mlir::func::FuncOp::create(builder.getUnknownLoc(),
                                                "c24_pending_packet_synthetic",
                                                functionType);
    function->setAttr("ave.gpu_func", builder.getI32IntegerAttr(2));
    auto *entry = function.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    auto loc = builder.getUnknownLoc();
    auto zeroIndex = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    auto trueValue = mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
    auto packetType = mlir::VectorType::get({8}, bf16);
    auto issue = AMDGPURegionPendingPacketIssueOp::create(
        builder, loc, packetType, entry->getArgument(0), trueValue,
        mlir::ValueRange{zeroIndex, zeroIndex, zeroIndex, zeroIndex});
    issue->setAttr("avelang.region_pending_packet.role",
                   builder.getStringAttr("synthetic_next"));
    issue->setAttr("avelang.region_pending_packet.issue_point",
                   builder.getStringAttr("before_three_current_mfma"));

    auto zero = mlir::arith::ConstantOp::create(
        builder, loc, f32, builder.getF32FloatAttr(0.0F));
    auto accumulator = mlir::vector::SplatOp::create(builder, loc,
                                                       accumulatorType, zero);
    auto physical = ChunkOPhysicalPlan::makeC12T2048WG256();
    std::string error;
    ASSERT_TRUE(physical.verify(&error)) << error;
    mlir::Value current = accumulator.getResult();
    for (int i = 0; i < 3; ++i) {
        current = makeCurrentMfma(builder, loc, physical.h, entry->getArgument(1),
                                  entry->getArgument(2), current, zeroIndex)
                      .getResult();
    }
    auto commit = AMDGPURegionPendingPacketCommitOp::create(
        builder, loc, issue.getPacket(), trueValue, entry->getArgument(3),
        mlir::ValueRange{zeroIndex, zeroIndex});
    commit->setAttr("avelang.region_pending_packet.commit_point",
                    builder.getStringAttr("after_three_current_mfma"));
    mlir::vector::StoreOp::create(builder, loc, current, entry->getArgument(4),
                                  mlir::ValueRange{zeroIndex});
    mlir::func::ReturnOp::create(builder, loc);
    module.push_back(function);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module))) << module;

    C24Artifacts artifacts;
    if (auto value = llvm::sys::Process::GetEnv(
            "AVELANG_C24_SYNTHETIC_OUTPUT_DIR")) {
        artifacts.directory = *value;
        std::error_code ec = llvm::sys::fs::create_directories(artifacts.directory);
        ASSERT_FALSE(ec) << ec.message();
    }
    std::string pre;
    llvm::raw_string_ostream preStream(pre);
    module.print(preStream);
    artifacts.write("pre_c24_pending_packet.mlir", pre);

    GPUCompilationOptions options;
    options.triple = "amdgcn-amd-amdhsa";
    options.chipset = "gfx942";
    options.optimization_level = 3;
    AMDGPUBackend backend;
    backend.EnsureInitialized();
    llvm::LLVMContext llvmContext;
    LowerToLLVM compiler(irContext.get());
    auto llvmModule = compiler.compile(module, llvmContext, options);
    ASSERT_NE(llvmModule, nullptr);
    std::string post;
    llvm::raw_string_ostream postStream(post);
    module.print(postStream);
    artifacts.write("post_c24_pending_packet.mlir", post);
    std::string llvmText;
    llvm::raw_string_ostream llvmStream(llvmText);
    llvmModule->print(llvmStream, nullptr);
    artifacts.write("post_c24_pending_packet.ll", llvmText);

    auto assembly = backend.generateAssembly(*llvmModule, options);
    ASSERT_TRUE(static_cast<bool>(assembly))
        << llvm::toString(assembly.takeError());
    artifacts.write("final_c24_pending_packet.s", *assembly);
    const llvm::StringRef isa(*assembly);
    const auto load = firstOf(isa, "global_load", "buffer_load");
    const auto firstMfma = isa.find("v_mfma_f32_32x32x8_bf16");
    const auto secondMfma = isa.find("v_mfma_f32_32x32x8_bf16",
                                     firstMfma + 1);
    const auto wait = isa.find("s_waitcnt", secondMfma + 1);
    const auto store = isa.find("ds_write", wait + 1);
    ASSERT_NE(load, llvm::StringRef::npos);
    ASSERT_NE(firstMfma, llvm::StringRef::npos);
    ASSERT_NE(secondMfma, llvm::StringRef::npos);
    ASSERT_NE(wait, llvm::StringRef::npos);
    ASSERT_NE(store, llvm::StringRef::npos);
    EXPECT_LT(load, firstMfma);
    EXPECT_LT(secondMfma, wait);
    EXPECT_LT(wait, store);
}

} // namespace
