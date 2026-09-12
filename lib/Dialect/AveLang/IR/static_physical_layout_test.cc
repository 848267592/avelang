//===- static_physical_layout_test.cc - C13 representation tests ----------===//

#include "AveLangDialect.h"
#include "AveLangOps.h"
#include "lower_qwen_block_dot_pass.h"
#include "static_physical_layout.h"

#include <gtest/gtest.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/raw_ostream.h>

#include <set>
#include <tuple>

using namespace causalflow::avelang::dialect::c13;
using namespace causalflow::avelang::dialect;

namespace {

class C13StaticPhysicalLayoutTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;

    void SetUp() override {
        context.loadDialect<causalflow::avelang::dialect::AveLangDialect,
                             mlir::func::FuncDialect>();
    }
};

void expectUniqueSharedMapping(const SharedEncoding &encoding,
                               std::array<int64_t, 2> shape) {
    std::set<int64_t> offsets;
    for (int64_t i = 0; i < shape[0]; ++i) {
        for (int64_t j = 0; j < shape[1]; ++j) {
            auto offset = encoding.getSharedByteOffset({{i, j}}, shape, 2);
            ASSERT_TRUE(offset.has_value());
            EXPECT_TRUE(offsets.insert(*offset).second);
        }
    }
    EXPECT_EQ(offsets.size(),
              static_cast<size_t>(shape[0] * shape[1]));
}

void expectUniqueDotSlots(const DotOperandEncoding &encoding) {
    std::set<std::tuple<int64_t, int64_t, int64_t, int64_t>> slots;
    for (int64_t row = 0; row < 64; ++row) {
        for (int64_t col = 0; col < 64; ++col) {
            auto slot = encoding.getOperandRegisterSlot({{row, col}});
            ASSERT_TRUE(slot.has_value());
            EXPECT_TRUE(slots.insert({slot->rowTile, slot->row, slot->kGroup,
                                      slot->word})
                            .second);
            EXPECT_EQ(slot->opIdx, encoding.opIdx);
        }
    }
    EXPECT_EQ(slots.size(), 64u * 64u);
}

TEST_F(C13StaticPhysicalLayoutTest, C12PlanVerifies) {
    auto plan = ChunkOPhysicalPlan::makeC12T2048WG256();
    std::string error;
    EXPECT_TRUE(plan.verify(&error)) << error;
    EXPECT_EQ(plan.workgroupSize, 256);
    EXPECT_EQ(plan.waveSize, 64);
    EXPECT_EQ(plan.consumers.size(), 5u);
    EXPECT_EQ(plan.lifetimes.size(), 2u);
}

TEST_F(C13StaticPhysicalLayoutTest, DistributedOwnershipIsBijective) {
    auto plan = ChunkOPhysicalPlan::makeC12T2048WG256();
    const std::array<std::pair<const char *, const DistributedEncoding *>, 4>
        blocks = {{{"Q", &plan.q.distributed},
                   {"H", &plan.h.distributed},
                   {"K", &plan.k.distributed},
                   {"V", &plan.v.distributed}}};
    for (const auto &[name, encoding] : blocks) {
        (void)name;
        ASSERT_TRUE(encoding->verify());
        std::set<LogicalCoord> logical;
        for (int64_t wave = 0; wave < encoding->wavesPerCTA[0] *
                                             encoding->wavesPerCTA[1];
             ++wave) {
            for (int64_t lane = 0; lane < 64; ++lane) {
                for (int64_t r0 = 0; r0 < encoding->sizePerThread[0]; ++r0) {
                    for (int64_t r1 = 0; r1 < encoding->sizePerThread[1];
                         ++r1) {
                        auto value = encoding->mapHardwareToLogical(
                            {wave, lane, {r0, r1}});
                        ASSERT_TRUE(value.has_value());
                        EXPECT_TRUE(logical.insert(*value).second);
                        auto owners = encoding->getOwners(*value);
                        ASSERT_EQ(owners.size(), 1u);
                        EXPECT_EQ(owners.front(),
                                  (HardwareCoord{wave, lane, {r0, r1}}));
                    }
                }
            }
        }
        EXPECT_EQ(logical.size(), static_cast<size_t>(
                                      encoding->logicalShape[0] *
                                      encoding->logicalShape[1]));
    }
}

TEST_F(C13StaticPhysicalLayoutTest, SharedAndDotMappingsAreBijective) {
    auto plan = ChunkOPhysicalPlan::makeC12T2048WG256();
    expectUniqueSharedMapping(plan.q.shared, {64, 32});
    expectUniqueSharedMapping(plan.h.shared, {64, 32});
    expectUniqueSharedMapping(plan.k.shared, {32, 64});
    expectUniqueSharedMapping(plan.v.shared, {64, 64});
    expectUniqueSharedMapping(plan.scoreShared, {64, 64});
    expectUniqueDotSlots(plan.q.dot);
    expectUniqueDotSlots(plan.h.dot);
    expectUniqueDotSlots(plan.k.dot);
    expectUniqueDotSlots(plan.v.dot);
}

TEST_F(C13StaticPhysicalLayoutTest, StaticTransformsPreserveElements) {
    auto plan = ChunkOPhysicalPlan::makeC12T2048WG256();
    EXPECT_TRUE(plan.q.transform.verifyPermutation({64, 32}));
    EXPECT_TRUE(plan.h.transform.verifyPermutation({64, 32}));
    EXPECT_TRUE(plan.k.transform.verifyPermutation({32, 64}));
    EXPECT_TRUE(plan.v.transform.verifyPermutation({64, 64}));

    auto transposed = plan.h.transform.apply({{7, 11}});
    ASSERT_TRUE(transposed.has_value());
    EXPECT_EQ(*transposed, (LogicalCoord{{11, 7}}));
}

TEST_F(C13StaticPhysicalLayoutTest, NativeInThreadBasisIsBijective) {
    auto plan = ChunkOPhysicalPlan::makeC12T2048WG256();
    ASSERT_EQ(plan.v.distributed.sizePerThread,
              (std::array<int64_t, 2>{4, 8}));
    ASSERT_EQ(plan.v.distributed.wavesPerCTA,
              (std::array<int64_t, 2>{2, 1}));
    ASSERT_TRUE(plan.v.transform.verify());

    std::set<LogicalCoord> logical;
    for (int64_t wave = 0; wave < 2; ++wave) {
        for (int64_t lane = 0; lane < 64; ++lane) {
            for (int64_t r0 = 0; r0 < 4; ++r0) {
                for (int64_t r1 = 0; r1 < 8; ++r1) {
                    auto value = plan.v.transform.mapBasis(
                        {wave, lane, {r0, r1}}, {64, 64});
                    ASSERT_TRUE(value.has_value());
                    EXPECT_TRUE(logical.insert(*value).second);
                }
            }
        }
    }
    EXPECT_EQ(logical.size(), 64u * 64u);
}

TEST_F(C13StaticPhysicalLayoutTest, AttributeRoundTrip) {
    auto plan = ChunkOPhysicalPlan::makeC12T2048WG256();
    auto qDist = DistributedEncoding::fromAttr(
        plan.q.distributed.toAttr(&context));
    auto qShared = SharedEncoding::fromAttr(plan.q.shared.toAttr(&context));
    auto mfma = MfmaEncoding::fromAttr(plan.mfma.toAttr(&context));
    auto qDot = DotOperandEncoding::fromAttr(plan.q.dot.toAttr(&context));
    auto hTransform = StaticLayoutTransform::fromAttr(
        plan.h.transform.toAttr(&context));
    ASSERT_TRUE(qDist.has_value());
    ASSERT_TRUE(qShared.has_value());
    ASSERT_TRUE(mfma.has_value());
    ASSERT_TRUE(qDot.has_value());
    ASSERT_TRUE(hTransform.has_value());
    EXPECT_EQ(*qDist, plan.q.distributed);
    EXPECT_EQ(*qShared, plan.q.shared);
    EXPECT_EQ(*mfma, plan.mfma);
    EXPECT_EQ(qDot->opIdx, plan.q.dot.opIdx);
    EXPECT_EQ(qDot->kWidth, plan.q.dot.kWidth);
    EXPECT_EQ(*hTransform, plan.h.transform);
}

TEST_F(C13StaticPhysicalLayoutTest, NegativeMappingsAreRejected) {
    auto plan = ChunkOPhysicalPlan::makeC12T2048WG256();

    auto badDistributed = plan.q.distributed;
    badDistributed.order = {0, 0};
    EXPECT_FALSE(badDistributed.verify());

    auto badShape = plan.q.distributed;
    badShape.logicalShape[0] += 1;
    EXPECT_FALSE(badShape.verify());

    auto badShared = plan.q.shared;
    badShared.kind = "unknown_shared";
    EXPECT_FALSE(badShared.verify());

    auto badTransform = plan.h.transform;
    badTransform.permutation = {0, 0};
    EXPECT_FALSE(badTransform.verify());

    auto badDot = plan.q.dot;
    badDot.kWidth = 8;
    EXPECT_FALSE(badDot.verify());
}

TEST_F(C13StaticPhysicalLayoutTest, TypedAttributesSurviveCanonicalizer) {
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    auto functionType = builder.getFunctionType({}, {});
    auto function = mlir::func::FuncOp::create(builder.getUnknownLoc(),
                                                "c13_survival", functionType);
    function->setAttr("c13.distributed",
                      ChunkOPhysicalPlan::makeC12T2048WG256()
                          .q.distributed.toAttr(&context));
    function->setAttr("c13.shared",
                      ChunkOPhysicalPlan::makeC12T2048WG256()
                          .q.shared.toAttr(&context));
    function->setAttr("c13.dot",
                      ChunkOPhysicalPlan::makeC12T2048WG256()
                          .q.dot.toAttr(&context));
    function->setAttr("c13.transform",
                      ChunkOPhysicalPlan::makeC12T2048WG256()
                          .h.transform.toAttr(&context));
    auto *entry = function.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(function);

    mlir::PassManager pm(&context);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.nest<mlir::func::FuncOp>().addPass(createLowerQwenBlockDotPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(module))) << module;
    auto surviving = module.lookupSymbol<mlir::func::FuncOp>("c13_survival");
    ASSERT_TRUE(surviving);
    EXPECT_TRUE(surviving->getAttrOfType<DistributedEncodingAttr>(
        "c13.distributed"));
    EXPECT_TRUE(surviving->getAttrOfType<SharedEncodingAttr>("c13.shared"));
    EXPECT_TRUE(
        surviving->getAttrOfType<DotOperandEncodingAttr>("c13.dot"));
    EXPECT_TRUE(surviving->getAttrOfType<StaticTransformAttr>(
        "c13.transform"));

    std::string printed;
    llvm::raw_string_ostream stream(printed);
    module.print(stream);
    stream.flush();
    auto reparsed = mlir::parseSourceString<mlir::ModuleOp>(printed, &context);
    ASSERT_TRUE(reparsed) << printed;
    auto reparsedFunction =
        reparsed->lookupSymbol<mlir::func::FuncOp>("c13_survival");
    ASSERT_TRUE(reparsedFunction);
    EXPECT_TRUE(reparsedFunction->getAttrOfType<DistributedEncodingAttr>(
        "c13.distributed"));
    EXPECT_TRUE(
        reparsedFunction->getAttrOfType<DotOperandEncodingAttr>("c13.dot"));
}

} // namespace
