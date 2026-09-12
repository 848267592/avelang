//===- c25_current_ready_pending_codegen_test.cc -------------------------===//
//
// C25 compiler-only feasibility proof.  Two BF16x8 VMEM packets are issued
// in order: the older H packet is made current-ready, while the younger K
// packet remains pending through independent H MFMA work.  The test asserts
// the natural final-ISA dependency order; it never injects a waitcnt value.
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

#include <algorithm>
#include <optional>
#include <regex>
#include <string>

using namespace causalflow::avelang::dialect;
using namespace causalflow::avelang::dialect::c13;
using namespace causalflow::avelang::target::amdgpu;
using namespace causalflow::avelang::target::gpu;

namespace {

struct C25Artifacts {
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
    current->setAttr("c25.synthetic.current_work",
                     builder.getStringAttr("current_h_mfma32"));
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

size_t nthOf(llvm::StringRef text, llvm::StringRef first,
             llvm::StringRef second, unsigned nth) {
    size_t offset = 0;
    for (unsigned i = 0; i <= nth; ++i) {
        const auto current = firstOf(text.drop_front(offset), first, second);
        if (current == llvm::StringRef::npos)
            return current;
        offset += current;
        if (i == nth)
            return offset;
        ++offset;
    }
    return llvm::StringRef::npos;
}

std::optional<unsigned> firstVmcntAtOrAfter(llvm::StringRef isa,
                                             size_t offset,
                                             size_t *position) {
    const std::regex pattern("s_waitcnt[[:space:]]+vmcnt\\(([0-9]+)\\)");
    std::cmatch match;
    const char *begin = isa.data() + offset;
    if (!std::regex_search(begin, isa.data() + isa.size(), match, pattern))
        return std::nullopt;
    if (position)
        *position = offset + static_cast<size_t>(match.position(0));
    return static_cast<unsigned>(std::stoul(match[1].str()));
}

TEST(C25CurrentReadyPendingCodegenTest, NaturalPartialWaitKeepsYoungerPacketPending) {
    auto irContext = causalflow::avelang::ir::IRContext::Create();
    auto *context = irContext->GetMLIRContext();
    context->loadDialect<AveLangDialect, mlir::func::FuncDialect,
                         mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                         mlir::vector::VectorDialect>();
    mlir::OpBuilder builder(context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto bf16 = builder.getBF16Type();
    auto f32 = builder.getF32Type();
    auto global = mlir::MemRefType::get({1, 64, 4, 128}, bf16);
    auto workgroup = builder.getI64IntegerAttr(3);
    auto stage = mlir::MemRefType::get(
        {64, 64}, bf16, mlir::MemRefLayoutAttrInterface(), workgroup);
    auto output = mlir::MemRefType::get({16}, f32);
    auto accumulatorType = mlir::VectorType::get({16}, f32);
    auto functionType = builder.getFunctionType(
        {global, global, stage, stage, stage, output}, {});
    auto function = mlir::func::FuncOp::create(
        builder.getUnknownLoc(), "c25_current_ready_next_pending_synthetic",
        functionType);
    function->setAttr("ave.gpu_func", builder.getI32IntegerAttr(2));
    auto *entry = function.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    auto loc = builder.getUnknownLoc();
    auto zeroIndex = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    auto oneIndex = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
    auto trueValue = mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
    auto packetType = mlir::VectorType::get({8}, bf16);

    // The older H load is intentionally issued first and must become ready
    // before its MFMA consumer.  K is younger and has no consumer until the
    // final CommitPacket below.
    auto hIssue = AMDGPURegionPendingPacketIssueOp::create(
        builder, loc, packetType, entry->getArgument(0), trueValue,
        mlir::ValueRange{zeroIndex, zeroIndex, zeroIndex, zeroIndex});
    hIssue->setAttr("avelang.region_pending_packet.role",
                    builder.getStringAttr("synthetic_current_H"));
    hIssue->setAttr("avelang.region_pending_packet.issue_point",
                    builder.getStringAttr("older_current_issue"));
    auto kIssue = AMDGPURegionPendingPacketIssueOp::create(
        builder, loc, packetType, entry->getArgument(1), trueValue,
        mlir::ValueRange{zeroIndex, oneIndex, zeroIndex, zeroIndex});
    kIssue->setAttr("avelang.region_pending_packet.role",
                    builder.getStringAttr("synthetic_next_K"));
    kIssue->setAttr("avelang.region_pending_packet.issue_point",
                    builder.getStringAttr("younger_next_issue"));

    auto hCommit = AMDGPURegionPendingPacketCommitOp::create(
        builder, loc, hIssue.getPacket(), trueValue, entry->getArgument(2),
        mlir::ValueRange{zeroIndex, zeroIndex});
    hCommit->setAttr("avelang.region_pending_packet.role",
                     builder.getStringAttr("synthetic_current_H"));
    hCommit->setAttr("avelang.region_pending_packet.commit_point",
                     builder.getStringAttr("current_ready_before_mfma"));

    auto zero = mlir::arith::ConstantOp::create(
        builder, loc, f32, builder.getF32FloatAttr(0.0F));
    auto accumulator = mlir::vector::SplatOp::create(builder, loc,
                                                       accumulatorType, zero);
    auto physical = ChunkOPhysicalPlan::makeC12T2048WG256();
    std::string error;
    ASSERT_TRUE(physical.verify(&error)) << error;
    mlir::Value current = accumulator.getResult();
    for (int i = 0; i < 3; ++i) {
        current = makeCurrentMfma(builder, loc, physical.h, entry->getArgument(2),
                                  entry->getArgument(3), current, zeroIndex)
                      .getResult();
    }

    auto kCommit = AMDGPURegionPendingPacketCommitOp::create(
        builder, loc, kIssue.getPacket(), trueValue, entry->getArgument(4),
        mlir::ValueRange{zeroIndex, zeroIndex});
    kCommit->setAttr("avelang.region_pending_packet.role",
                     builder.getStringAttr("synthetic_next_K"));
    kCommit->setAttr("avelang.region_pending_packet.commit_point",
                     builder.getStringAttr("after_current_h_mfma"));
    mlir::vector::StoreOp::create(builder, loc, current, entry->getArgument(5),
                                  mlir::ValueRange{zeroIndex});
    mlir::func::ReturnOp::create(builder, loc);
    module.push_back(function);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module))) << module;

    C25Artifacts artifacts;
    if (auto value = llvm::sys::Process::GetEnv(
            "AVELANG_C25_SYNTHETIC_OUTPUT_DIR")) {
        artifacts.directory = *value;
        std::error_code ec = llvm::sys::fs::create_directories(artifacts.directory);
        ASSERT_FALSE(ec) << ec.message();
    }
    std::string pre;
    llvm::raw_string_ostream preStream(pre);
    module.print(preStream);
    artifacts.write("pre_c25_current_ready_pending.mlir", pre);

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
    artifacts.write("post_c25_current_ready_pending.mlir", post);
    std::string llvmText;
    llvm::raw_string_ostream llvmStream(llvmText);
    llvmModule->print(llvmStream, nullptr);
    artifacts.write("post_c25_current_ready_pending.ll", llvmText);

    auto assembly = backend.generateAssembly(*llvmModule, options);
    ASSERT_TRUE(static_cast<bool>(assembly))
        << llvm::toString(assembly.takeError());
    artifacts.write("final_c25_current_ready_pending.s", *assembly);
    const llvm::StringRef isa(*assembly);
    const auto hLoad = nthOf(isa, "global_load", "buffer_load", 0);
    const auto kLoad = nthOf(isa, "global_load", "buffer_load", 1);
    ASSERT_NE(hLoad, llvm::StringRef::npos);
    ASSERT_NE(kLoad, llvm::StringRef::npos);
    EXPECT_LT(hLoad, kLoad);

    size_t hReadyWait = llvm::StringRef::npos;
    auto hReadyVmcnt = firstVmcntAtOrAfter(isa, kLoad, &hReadyWait);
    ASSERT_TRUE(hReadyVmcnt.has_value());
    // This asserts a naturally derived partial wait, not a preselected
    // waitcnt literal.  The younger K packet is still outstanding.
    EXPECT_GT(*hReadyVmcnt, 0U);
    const auto hStore = isa.find("ds_write", hReadyWait);
    ASSERT_NE(hStore, llvm::StringRef::npos);
    EXPECT_LT(hReadyWait, hStore);
    const auto firstMfma = isa.find("v_mfma_f32_32x32x8_bf16", hStore);
    const auto secondMfma = isa.find("v_mfma_f32_32x32x8_bf16",
                                     firstMfma + 1);
    ASSERT_NE(firstMfma, llvm::StringRef::npos);
    ASSERT_NE(secondMfma, llvm::StringRef::npos);
    EXPECT_LT(hStore, firstMfma);
    size_t kFinalWait = llvm::StringRef::npos;
    auto kFinalVmcnt = firstVmcntAtOrAfter(isa, secondMfma, &kFinalWait);
    ASSERT_TRUE(kFinalVmcnt.has_value());
    EXPECT_LT(secondMfma, kFinalWait);
    const auto kStore = isa.find("ds_write", kFinalWait);
    ASSERT_NE(kStore, llvm::StringRef::npos);
    EXPECT_LT(kFinalWait, kStore);
}

} // namespace
