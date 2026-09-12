//===- static_physical_layout.h - C13 compile-time layout algebra ----------===//
//
// This file deliberately contains no MLIR SSA values.  It is the small,
// target-scoped algebra used to prove that a recovered gfx942 physical plan
// can be represented before any lowering to shared memory or MFMA intrinsics.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AveLangAttrs.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace causalflow::avelang::dialect::c13 {

struct LogicalCoord {
    std::array<int64_t, 2> value{};

    int64_t operator[](size_t index) const { return value[index]; }
    bool operator==(const LogicalCoord &) const = default;
    bool operator<(const LogicalCoord &other) const { return value < other.value; }
};

struct HardwareCoord {
    int64_t wave = 0;
    int64_t lane = 0;
    std::array<int64_t, 2> reg{};

    bool operator==(const HardwareCoord &) const = default;
};

struct DotOperandSlot {
    int64_t opIdx = -1;
    int64_t rowTile = -1;
    int64_t row = -1;
    int64_t kGroup = -1;
    int64_t word = -1;

    bool operator==(const DotOperandSlot &) const = default;
};

struct DistributedEncoding {
    std::array<int64_t, 2> logicalShape{};
    std::array<int64_t, 2> sizePerThread{};
    std::array<int64_t, 2> threadsPerWave{};
    std::array<int64_t, 2> wavesPerCTA{};
    std::array<int64_t, 2> order{};

    bool operator==(const DistributedEncoding &) const = default;

    bool verify(std::string *error = nullptr) const;
    std::optional<LogicalCoord>
    mapHardwareToLogical(const HardwareCoord &hardware) const;
    std::vector<HardwareCoord> getOwners(const LogicalCoord &logical) const;
    std::vector<LogicalCoord> enumerateLogical() const;

    DistributedEncodingAttr toAttr(mlir::MLIRContext *context) const;
    static std::optional<DistributedEncoding>
    fromAttr(DistributedEncodingAttr attr);
};

struct SharedEncoding {
    std::string kind;
    int64_t vec = 0;
    int64_t perPhase = 0;
    int64_t maxPhase = 0;
    std::array<int64_t, 2> order{};
    bool rotating = false;

    bool operator==(const SharedEncoding &) const = default;

    bool verify(std::string *error = nullptr) const;

    // The C13 MVP uses the finite phase-xor recipe for the recovered
    // swizzled/rotating encodings.  It is a static bijection, not a runtime
    // layout search.  The recipe is intentionally kept in the representation
    // layer; codegen is a later phase.
    std::optional<int64_t>
    getSharedByteOffset(const LogicalCoord &logical,
                        const std::array<int64_t, 2> &shape,
                        int64_t elementBytes) const;
    bool verifyMapping(const std::array<int64_t, 2> &shape,
                       int64_t elementBytes,
                       std::string *error = nullptr) const;

    SharedEncodingAttr toAttr(mlir::MLIRContext *context) const;
    static std::optional<SharedEncoding> fromAttr(SharedEncodingAttr attr);
};

struct MfmaEncoding {
    std::string target;
    int64_t version = 0;
    std::array<int64_t, 3> instrShape{};
    std::array<int64_t, 2> warpsPerCTA{};
    bool isTransposed = false;

    bool operator==(const MfmaEncoding &) const = default;

    bool verify(std::string *error = nullptr) const;
    MfmaEncodingAttr toAttr(mlir::MLIRContext *context) const;
    static std::optional<MfmaEncoding> fromAttr(MfmaEncodingAttr attr);
};

struct DotOperandEncoding {
    int64_t opIdx = -1;
    int64_t kWidth = 0;
    MfmaEncoding parent;

    bool operator==(const DotOperandEncoding &) const = default;

    bool verify(std::string *error = nullptr) const;
    std::optional<DotOperandSlot>
    getOperandRegisterSlot(const LogicalCoord &logical) const;
    DotOperandEncodingAttr toAttr(mlir::MLIRContext *context) const;
    static std::optional<DotOperandEncoding>
    fromAttr(DotOperandEncodingAttr attr);
};

struct StaticLayoutTransform {
    std::string kind;
    std::string sourceEncoding;
    std::string targetEncoding;
    std::array<int64_t, 2> permutation{};
    // Flattened pairs of logical coordinates.  Each pair is one GF(2) basis
    // vector from Triton's LinearLayout representation.  Empty vectors retain
    // the C13 semantic-only transform form; non-empty vectors are required
    // for an exact in-thread register/lane/warp recipe.
    std::vector<int64_t> registerBasis;
    std::vector<int64_t> laneBasis;
    std::vector<int64_t> warpBasis;
    int64_t registerStride = 1;

    bool operator==(const StaticLayoutTransform &) const = default;

    bool verify(std::string *error = nullptr) const;
    std::optional<LogicalCoord>
    apply(const LogicalCoord &logical) const;
    std::optional<LogicalCoord>
    mapBasis(const HardwareCoord &hardware,
             const std::array<int64_t, 2> &shape) const;
    bool verifyPermutation(const std::array<int64_t, 2> &shape,
                           std::string *error = nullptr) const;
    StaticTransformAttr toAttr(mlir::MLIRContext *context) const;
    static std::optional<StaticLayoutTransform>
    fromAttr(StaticTransformAttr attr);
};

struct PhysicalBlockPlan {
    std::string name;
    DistributedEncoding distributed;
    SharedEncoding shared;
    DotOperandEncoding dot;
    StaticLayoutTransform transform;
    std::vector<std::string> consumers;

    bool verify(std::string *error = nullptr) const;
};

struct ConsumerRelationship {
    std::string producer;
    std::string consumer;
    bool samePhysicalSource = false;
};

struct PhaseLifetime {
    std::string name;
    std::string beginPhase;
    std::string endPhase;
    std::vector<std::string> regions;
};

struct SharedRegion {
    std::string name;
    int64_t bytes = 0;
    std::string phase;
};

struct ChunkOPhysicalPlan {
    std::string target = "gfx942";
    int64_t waveSize = 64;
    int64_t workgroupSize = 256;
    PhysicalBlockPlan q;
    PhysicalBlockPlan h;
    PhysicalBlockPlan k;
    PhysicalBlockPlan v;
    SharedEncoding scoreShared;
    MfmaEncoding mfma;
    std::vector<ConsumerRelationship> consumers;
    std::vector<PhaseLifetime> lifetimes;
    std::vector<SharedRegion> sharedRegions;

    bool verify(std::string *error = nullptr) const;

    // The recovered T2048 WG256 plan.  Unknown native facts are not hidden in
    // this object; this factory only uses facts mechanically present in the
    // C12 TTGIR/JSON artifact.
    static ChunkOPhysicalPlan makeC12T2048WG256();
};

} // namespace causalflow::avelang::dialect::c13
