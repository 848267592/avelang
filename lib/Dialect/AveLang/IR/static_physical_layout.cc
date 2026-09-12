//===- static_physical_layout.cc - C13 compile-time layout algebra ---------===//

#include "static_physical_layout.h"

#include <mlir/IR/BuiltinAttributes.h>

#include <algorithm>
#include <limits>
#include <sstream>

namespace causalflow::avelang::dialect::c13 {
namespace {

template <size_t N>
bool isPermutation(const std::array<int64_t, N> &order) {
    std::array<bool, N> seen{};
    for (auto value : order) {
        if (value < 0 || value >= static_cast<int64_t>(N) || seen[value])
            return false;
        seen[value] = true;
    }
    return true;
}

bool positive(const std::array<int64_t, 2> &values) {
    return values[0] > 0 && values[1] > 0;
}

void setError(std::string *error, const std::string &message) {
    if (error)
        *error = message;
}

std::array<int64_t, 2> decodeLinear(int64_t linear,
                                    const std::array<int64_t, 2> &extents,
                                    const std::array<int64_t, 2> &order) {
    std::array<int64_t, 2> result{};
    for (auto dimension : order) {
        result[dimension] = linear % extents[dimension];
        linear /= extents[dimension];
    }
    return result;
}

int64_t product(const std::array<int64_t, 2> &values) {
    return values[0] * values[1];
}

template <size_t N>
llvm::ArrayRef<int64_t> toArrayRef(const std::array<int64_t, N> &values) {
    return {values.data(), values.size()};
}

} // namespace

bool DistributedEncoding::verify(std::string *error) const {
    if (!positive(logicalShape) || !positive(sizePerThread) ||
        !positive(threadsPerWave) || !positive(wavesPerCTA)) {
        setError(error, "distributed encoding fields must be positive");
        return false;
    }
    if (!isPermutation(order)) {
        setError(error, "distributed encoding order must be a permutation");
        return false;
    }
    for (size_t dimension = 0; dimension < 2; ++dimension) {
        if (logicalShape[dimension] !=
            sizePerThread[dimension] * threadsPerWave[dimension] *
                wavesPerCTA[dimension]) {
            setError(error, "distributed shape is not fully covered by the "
                           "thread/wave encoding");
            return false;
        }
    }
    if (product(threadsPerWave) != 64) {
        setError(error, "C13 distributed encoding requires wave64");
        return false;
    }
    if (product(wavesPerCTA) != 1 && product(wavesPerCTA) != 2 &&
        product(wavesPerCTA) != 4) {
        setError(error, "C13 supports one, two, or four waves per CTA");
        return false;
    }
    return true;
}

std::optional<LogicalCoord>
DistributedEncoding::mapHardwareToLogical(const HardwareCoord &hardware) const {
    if (!verify() || hardware.wave < 0 || hardware.wave >= product(wavesPerCTA) ||
        hardware.lane < 0 || hardware.lane >= product(threadsPerWave))
        return std::nullopt;
    for (size_t dimension = 0; dimension < 2; ++dimension) {
        if (hardware.reg[dimension] < 0 ||
            hardware.reg[dimension] >= sizePerThread[dimension])
            return std::nullopt;
    }
    auto thread = decodeLinear(hardware.lane, threadsPerWave, order);
    auto wave = decodeLinear(hardware.wave, wavesPerCTA, order);
    LogicalCoord logical;
    for (size_t dimension = 0; dimension < 2; ++dimension) {
        logical.value[dimension] =
            ((wave[dimension] * threadsPerWave[dimension]) +
             thread[dimension]) *
                sizePerThread[dimension] +
            hardware.reg[dimension];
    }
    return logical;
}

std::vector<HardwareCoord>
DistributedEncoding::getOwners(const LogicalCoord &logical) const {
    std::vector<HardwareCoord> owners;
    if (!verify())
        return owners;
    if (logical[0] < 0 || logical[1] < 0 ||
        logical[0] >= logicalShape[0] || logical[1] >= logicalShape[1])
        return owners;
    for (int64_t wave = 0; wave < product(wavesPerCTA); ++wave) {
        for (int64_t lane = 0; lane < product(threadsPerWave); ++lane) {
            for (int64_t reg0 = 0; reg0 < sizePerThread[0]; ++reg0) {
                for (int64_t reg1 = 0; reg1 < sizePerThread[1]; ++reg1) {
                    HardwareCoord hardware{wave, lane, {reg0, reg1}};
                    auto mapped = mapHardwareToLogical(hardware);
                    if (mapped && *mapped == logical)
                        owners.push_back(hardware);
                }
            }
        }
    }
    return owners;
}

std::vector<LogicalCoord> DistributedEncoding::enumerateLogical() const {
    std::vector<LogicalCoord> result;
    if (!verify())
        return result;
    result.reserve(product(logicalShape));
    for (int64_t dim0 = 0; dim0 < logicalShape[0]; ++dim0)
        for (int64_t dim1 = 0; dim1 < logicalShape[1]; ++dim1)
            result.push_back({{dim0, dim1}});
    return result;
}

DistributedEncodingAttr
DistributedEncoding::toAttr(mlir::MLIRContext *context) const {
    return DistributedEncodingAttr::get(
        context, mlir::DenseI64ArrayAttr::get(context, toArrayRef(logicalShape)),
        mlir::DenseI64ArrayAttr::get(context, toArrayRef(sizePerThread)),
        mlir::DenseI64ArrayAttr::get(context, toArrayRef(threadsPerWave)),
        mlir::DenseI64ArrayAttr::get(context, toArrayRef(wavesPerCTA)),
        mlir::DenseI64ArrayAttr::get(context, toArrayRef(order)));
}

std::optional<DistributedEncoding>
DistributedEncoding::fromAttr(DistributedEncodingAttr attr) {
    auto shape = attr.getLogicalShape().asArrayRef();
    auto size = attr.getSizePerThread().asArrayRef();
    auto threads = attr.getThreadsPerWave().asArrayRef();
    auto waves = attr.getWavesPerCTA().asArrayRef();
    auto order = attr.getOrder().asArrayRef();
    if (shape.size() != 2 || size.size() != 2 || threads.size() != 2 ||
        waves.size() != 2 || order.size() != 2)
        return std::nullopt;
    DistributedEncoding result{{shape[0], shape[1]},
                               {size[0], size[1]},
                               {threads[0], threads[1]},
                               {waves[0], waves[1]},
                               {order[0], order[1]}};
    return result.verify() ? std::optional<DistributedEncoding>(result)
                           : std::nullopt;
}

bool SharedEncoding::verify(std::string *error) const {
    if (kind != "swizzled_shared" && kind != "amd_rotating_shared" &&
        kind != "shared") {
        setError(error, "unsupported C13 shared encoding kind");
        return false;
    }
    if (vec <= 0 || perPhase <= 0 || maxPhase <= 0 || !isPermutation(order)) {
        setError(error, "invalid shared encoding parameters");
        return false;
    }
    if (kind == "amd_rotating_shared" && !rotating) {
        setError(error, "amd_rotating_shared must set rotating=true");
        return false;
    }
    return true;
}

std::optional<int64_t> SharedEncoding::getSharedByteOffset(
    const LogicalCoord &logical, const std::array<int64_t, 2> &shape,
    int64_t elementBytes) const {
    if (!verify() || elementBytes <= 0 || logical[0] < 0 || logical[1] < 0 ||
        logical[0] >= shape[0] || logical[1] >= shape[1] ||
        shape[order[1]] % vec != 0)
        return std::nullopt;

    const int64_t outer = logical[order[0]];
    const int64_t inner = logical[order[1]];
    const int64_t innerExtent = shape[order[1]];
    const int64_t groups = innerExtent / vec;
    const int64_t phase = (outer / perPhase) % maxPhase;
    int64_t swizzle = phase % groups;
    if (kind == "amd_rotating_shared") {
        // Triton's sharedToLinearLayoutAMDRotating adds the block phase to
        // the in-phase rotation.  Keep it compile-time parameterized and use
        // only shifts/masks in the target lowering; this is not a runtime
        // layout search.
        const int64_t blockNo = (outer / maxPhase / perPhase) % maxPhase;
        swizzle = (phase ^ blockNo) % groups;
    }
    const int64_t physicalGroup = (inner / vec) ^ swizzle;
    const int64_t physicalInner = physicalGroup * vec + (inner % vec);
    const int64_t elementOffset = outer * innerExtent + physicalInner;
    return elementOffset * elementBytes;
}

bool SharedEncoding::verifyMapping(const std::array<int64_t, 2> &shape,
                                   int64_t elementBytes,
                                   std::string *error) const {
    if (!verify(error) || shape[0] <= 0 || shape[1] <= 0 || elementBytes <= 0) {
        if (!error || error->empty())
            setError(error, "invalid shared mapping shape");
        return false;
    }
    std::vector<int64_t> offsets;
    offsets.reserve(product(shape));
    for (int64_t dim0 = 0; dim0 < shape[0]; ++dim0) {
        for (int64_t dim1 = 0; dim1 < shape[1]; ++dim1) {
            auto offset = getSharedByteOffset({{dim0, dim1}}, shape, elementBytes);
            if (!offset) {
                setError(error, "shared mapping produced an invalid address");
                return false;
            }
            offsets.push_back(*offset);
        }
    }
    std::sort(offsets.begin(), offsets.end());
    for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] == offsets[i - 1]) {
            setError(error, "shared mapping has an aliased byte offset");
            return false;
        }
    }
    if (offsets.empty() || offsets.back() >= product(shape) * elementBytes) {
        setError(error, "shared mapping exceeds its physical tile");
        return false;
    }
    return true;
}

SharedEncodingAttr SharedEncoding::toAttr(mlir::MLIRContext *context) const {
    return SharedEncodingAttr::get(context, kind, vec, perPhase, maxPhase,
                                   mlir::DenseI64ArrayAttr::get(
                                       context, toArrayRef(order)),
                                   rotating);
}

std::optional<SharedEncoding> SharedEncoding::fromAttr(SharedEncodingAttr attr) {
    auto order = attr.getOrder().asArrayRef();
    if (order.size() != 2)
        return std::nullopt;
    SharedEncoding result{attr.getKind().str(), attr.getVec(),
                          attr.getPerPhase(), attr.getMaxPhase(),
                          {order[0], order[1]}, attr.getRotating()};
    return result.verify() ? std::optional<SharedEncoding>(result)
                           : std::nullopt;
}

bool MfmaEncoding::verify(std::string *error) const {
    const bool supportedWarpShape =
        warpsPerCTA == std::array<int64_t, 2>{1, 2} ||
        warpsPerCTA == std::array<int64_t, 2>{2, 1} ||
        warpsPerCTA == std::array<int64_t, 2>{2, 2};
    if (target != "gfx942" || version != 3 ||
        instrShape != std::array<int64_t, 3>{32, 32, 8} ||
        !supportedWarpShape || !isTransposed) {
        setError(error, "C13/C14 requires gfx942 MFMA v3 32x32x8, supported wave64 warp shape, transposed");
        return false;
    }
    return true;
}

MfmaEncodingAttr MfmaEncoding::toAttr(mlir::MLIRContext *context) const {
    return MfmaEncodingAttr::get(
        context, target, version,
        mlir::DenseI64ArrayAttr::get(context, toArrayRef(instrShape)),
                                 mlir::DenseI64ArrayAttr::get(
                                     context, toArrayRef(warpsPerCTA)),
                                 isTransposed);
}

std::optional<MfmaEncoding> MfmaEncoding::fromAttr(MfmaEncodingAttr attr) {
    auto shape = attr.getInstrShape().asArrayRef();
    auto warps = attr.getWarpsPerCTA().asArrayRef();
    if (shape.size() != 3 || warps.size() != 2)
        return std::nullopt;
    MfmaEncoding result{attr.getTarget().str(), attr.getVersion(),
                        {shape[0], shape[1], shape[2]},
                        {warps[0], warps[1]}, attr.getIsTransposed()};
    return result.verify() ? std::optional<MfmaEncoding>(result)
                           : std::nullopt;
}

bool DotOperandEncoding::verify(std::string *error) const {
    if ((opIdx != 0 && opIdx != 1) || kWidth != 4 || !parent.verify(error)) {
        if (error && error->empty())
            *error = "C13 dot operand requires opIdx 0/1 and kWidth=4";
        return false;
    }
    return true;
}

std::optional<DotOperandSlot>
DotOperandEncoding::getOperandRegisterSlot(const LogicalCoord &logical) const {
    if (!verify() || logical[0] < 0 || logical[1] < 0 || logical[0] >= 64 ||
        logical[1] >= 64)
        return std::nullopt;
    DotOperandSlot result;
    result.opIdx = opIdx;
    result.rowTile = logical[0] / 32;
    result.row = logical[0] % 32;
    result.kGroup = logical[1] / kWidth;
    result.word = logical[1] % kWidth;
    return result;
}

DotOperandEncodingAttr
DotOperandEncoding::toAttr(mlir::MLIRContext *context) const {
    return DotOperandEncodingAttr::get(context, opIdx, kWidth,
                                       parent.toAttr(context));
}

std::optional<DotOperandEncoding>
DotOperandEncoding::fromAttr(DotOperandEncodingAttr attr) {
    auto parent = llvm::dyn_cast<MfmaEncodingAttr>(attr.getParent());
    if (!parent)
        return std::nullopt;
    auto mfma = MfmaEncoding::fromAttr(parent);
    if (!mfma)
        return std::nullopt;
    DotOperandEncoding result{attr.getOpIdx(), attr.getKWidth(), *mfma};
    return result.verify() ? std::optional<DotOperandEncoding>(result)
                           : std::nullopt;
}

bool StaticLayoutTransform::verify(std::string *error) const {
    if (kind != "identity" && kind != "fixed_transpose" &&
        kind != "fixed_permutation" && kind != "in_thread_transpose") {
        setError(error, "unsupported C13 static transform kind");
        return false;
    }
    if (!isPermutation(permutation)) {
        setError(error, "static transform must be a rank-2 permutation");
        return false;
    }
    if (kind == "identity" && permutation != std::array<int64_t, 2>{0, 1}) {
        setError(error, "identity transform must use [0, 1]");
        return false;
    }
    if (kind == "fixed_transpose" &&
        permutation != std::array<int64_t, 2>{1, 0}) {
        setError(error, "fixed transpose must use [1, 0]");
        return false;
    }
    if (kind == "in_thread_transpose") {
        if (permutation != std::array<int64_t, 2>{1, 0}) {
            setError(error, "in-thread transpose must use [1, 0]");
            return false;
        }
        for (const auto *basis : {&registerBasis, &laneBasis, &warpBasis}) {
            if (basis->size() % 2 != 0) {
                setError(error, "linear-layout basis must contain coordinate pairs");
                return false;
            }
        }
        if (registerBasis.empty() || laneBasis.empty() || warpBasis.empty()) {
            setError(error, "in-thread transpose requires register/lane/warp basis");
            return false;
        }
    }
    return true;
}

std::optional<LogicalCoord>
StaticLayoutTransform::apply(const LogicalCoord &logical) const {
    if (!verify())
        return std::nullopt;
    return LogicalCoord{{logical[permutation[0]], logical[permutation[1]]}};
}

std::optional<LogicalCoord> StaticLayoutTransform::mapBasis(
    const HardwareCoord &hardware, const std::array<int64_t, 2> &shape) const {
    if (!verify() || registerBasis.empty() || laneBasis.empty() ||
        warpBasis.empty() || shape[0] <= 0 || shape[1] <= 0)
        return std::nullopt;

    auto xorBasis = [](const std::vector<int64_t> &basis, int64_t selector) {
        LogicalCoord result{{0, 0}};
        for (size_t bit = 0; bit * 2 < basis.size(); ++bit) {
            if (((selector >> bit) & 1) == 0)
                continue;
            result.value[0] ^= basis[bit * 2];
            result.value[1] ^= basis[bit * 2 + 1];
        }
        return result;
    };

    // The target linear layout encodes register bits as a scalar register
    // slot.  C13's HardwareCoord keeps a rank-2 source register, so flatten
    // it in row-major order using the source slot's second dimension.
    if (registerStride <= 0)
        return std::nullopt;
    const int64_t registerSelector =
        hardware.reg[0] + hardware.reg[1] * registerStride;
    auto logical = xorBasis(registerBasis, registerSelector);
    auto lane = xorBasis(laneBasis, hardware.lane);
    auto wave = xorBasis(warpBasis, hardware.wave);
    logical.value[0] ^= lane.value[0] ^ wave.value[0];
    logical.value[1] ^= lane.value[1] ^ wave.value[1];
    if (logical[0] < 0 || logical[1] < 0 || logical[0] >= shape[0] ||
        logical[1] >= shape[1])
        return std::nullopt;
    return logical;
}

bool StaticLayoutTransform::verifyPermutation(
    const std::array<int64_t, 2> &shape, std::string *error) const {
    if (!verify(error))
        return false;
    std::vector<LogicalCoord> transformed;
    for (int64_t dim0 = 0; dim0 < shape[0]; ++dim0) {
        for (int64_t dim1 = 0; dim1 < shape[1]; ++dim1) {
            auto value = apply({{dim0, dim1}});
            if (!value) {
                setError(error, "transform failed for a logical coordinate");
                return false;
            }
            transformed.push_back(*value);
        }
    }
    std::sort(transformed.begin(), transformed.end(),
              [](const LogicalCoord &lhs, const LogicalCoord &rhs) {
                  return lhs.value < rhs.value;
              });
    for (size_t i = 1; i < transformed.size(); ++i) {
        if (transformed[i] == transformed[i - 1]) {
            setError(error, "static transform is not one-to-one");
            return false;
        }
    }
    return true;
}

StaticTransformAttr
StaticLayoutTransform::toAttr(mlir::MLIRContext *context) const {
    auto dense = [&](const std::vector<int64_t> &values) {
        return mlir::DenseI64ArrayAttr::get(
            context, llvm::ArrayRef<int64_t>(values.data(), values.size()));
    };
    return StaticTransformAttr::get(context, kind, sourceEncoding,
                                    targetEncoding,
                                    mlir::DenseI64ArrayAttr::get(
                                        context, toArrayRef(permutation)),
                                    dense(registerBasis), dense(laneBasis),
                                    dense(warpBasis), registerStride);
}

std::optional<StaticLayoutTransform>
StaticLayoutTransform::fromAttr(StaticTransformAttr attr) {
    auto permutation = attr.getPermutation().asArrayRef();
    auto registerBasis = attr.getRegisterBasis().asArrayRef();
    auto laneBasis = attr.getLaneBasis().asArrayRef();
    auto warpBasis = attr.getWarpBasis().asArrayRef();
    if (permutation.size() != 2)
        return std::nullopt;
    StaticLayoutTransform result{attr.getKind().str(),
                                 attr.getSourceEncoding().str(),
                                 attr.getTargetEncoding().str(),
                                 {permutation[0], permutation[1]},
                                 {registerBasis.begin(), registerBasis.end()},
                                 {laneBasis.begin(), laneBasis.end()},
                                 {warpBasis.begin(), warpBasis.end()},
                                 attr.getRegisterStride()};
    return result.verify() ? std::optional<StaticLayoutTransform>(result)
                           : std::nullopt;
}

bool PhysicalBlockPlan::verify(std::string *error) const {
    return distributed.verify(error) && shared.verify(error) &&
           shared.verifyMapping(distributed.logicalShape, 2, error) &&
           dot.verify(error) && transform.verify(error) &&
           transform.verifyPermutation(distributed.logicalShape, error);
}

bool ChunkOPhysicalPlan::verify(std::string *error) const {
    if (target != "gfx942" || waveSize != 64 || workgroupSize != 256) {
        setError(error, "C13 C12 validation plan requires gfx942 wave64 WG256");
        return false;
    }
    if (!mfma.verify(error) || !q.verify(error) || !h.verify(error) ||
        !k.verify(error) || !v.verify(error) ||
        !scoreShared.verifyMapping({{64, 64}}, 2, error))
        return false;
    bool qH = false;
    bool qK = false;
    for (const auto &relationship : consumers) {
        if (relationship.producer == "Q" && relationship.consumer == "Q@H" &&
            relationship.samePhysicalSource)
            qH = true;
        if (relationship.producer == "Q" && relationship.consumer == "Q@K" &&
            relationship.samePhysicalSource)
            qK = true;
    }
    if (!qH || !qK) {
        setError(error, "C12 plan must preserve Q dual-consumer identity");
        return false;
    }
    bool sourceBeforeScore = false;
    for (const auto &lifetime : lifetimes) {
        if (lifetime.name == "source_Q_H_K" &&
            lifetime.endPhase == "source_release")
            sourceBeforeScore = true;
    }
    if (!sourceBeforeScore) {
        setError(error, "C12 plan must preserve source-to-score lifetime boundary");
        return false;
    }
    return true;
}

ChunkOPhysicalPlan ChunkOPhysicalPlan::makeC12T2048WG256() {
    ChunkOPhysicalPlan plan;
    plan.mfma = {"gfx942", 3, {32, 32, 8}, {2, 2}, true};
    auto qDistributed = DistributedEncoding{{64, 32}, {1, 8}, {16, 4},
                                             {4, 1}, {1, 0}};
    auto kDistributed = DistributedEncoding{{32, 64}, {8, 1}, {4, 16},
                                             {1, 4}, {0, 1}};
    // The selected native chunk-o TTGIR uses #blocked with
    // sizePerThread=[4,8], threadsPerWarp=[8,8], warpsPerCTA=[2,1],
    // order=[1,0].  Keep the compiler-owned recipe consistent with the
    // in-thread #linear1 basis recovered below; the former [2,8] x [4,1]
    // entry was the older C12 representation, not the selected native map.
    auto vDistributed = DistributedEncoding{{64, 64}, {4, 8}, {8, 8},
                                             {2, 1}, {1, 0}};
    auto qShared = SharedEncoding{"swizzled_shared", 4, 2, 8, {1, 0}, false};
    auto hShared = SharedEncoding{"swizzled_shared", 1, 1, 1, {1, 0}, false};
    auto kShared = SharedEncoding{"swizzled_shared", 4, 2, 8, {0, 1}, false};
    auto vShared = SharedEncoding{"amd_rotating_shared", 4, 1, 16, {0, 1}, true};
    auto qTransform = StaticLayoutTransform{"identity", "#blocked2", "#shared", {0, 1}, {}, {}, {}, 1};
    auto hTransform = StaticLayoutTransform{"fixed_transpose", "#blocked2", "#linear", {1, 0}, {}, {}, {}, 1};
    auto kTransform = StaticLayoutTransform{"identity", "#blocked1", "#shared2", {0, 1}, {}, {}, {}, 1};
    // Exact #linear1 basis recovered from the selected native TTGIR.  The
    // vectors are the GF(2) basis emitted by Triton LinearLayout, not a lane
    // table inferred from source variable names.
    auto vTransform = StaticLayoutTransform{
        "in_thread_transpose", "#blocked", "#linear1", {1, 0},
        {1, 0, 2, 0, 0, 1, 0, 2, 0, 4},
        {0, 8, 0, 16, 0, 32, 4, 0, 8, 0, 16, 0}, {32, 0}, 4};
    plan.q = {"Q", qDistributed, qShared, {0, 4, plan.mfma}, qTransform,
              {"Q@H", "Q@K"}};
    plan.h = {"H", qDistributed, hShared, {1, 4, plan.mfma}, hTransform,
              {"Q@H"}};
    plan.k = {"K", kDistributed, kShared, {1, 4, plan.mfma}, kTransform,
              {"Q@K"}};
    plan.v = {"V", vDistributed, vShared, {1, 4, plan.mfma}, vTransform,
              {"score@V-new"}};
    plan.scoreShared = {"swizzled_shared", 1, 1, 1, {0, 1}, false};
    plan.consumers = {
        {"Q", "Q@H", true}, {"Q", "Q@K", true}, {"H", "Q@H", true},
        {"K", "Q@K", true}, {"V", "score@V-new", true}};
    plan.lifetimes = {
        {"source_Q_H_K", "source_prologue", "source_release", {"Q", "H", "K"}},
        {"score_V", "score_and_causal_g", "output", {"score", "V"}}};
    plan.sharedRegions = {
        {"Q", 8192, "source_Q_H_K"}, {"H", 8192, "source_Q_H_K"},
        {"K", 8192, "source_Q_H_K"}, {"score", 8192, "score_V"},
        {"V", 8192, "score_V"}};
    return plan;
}

} // namespace causalflow::avelang::dialect::c13
