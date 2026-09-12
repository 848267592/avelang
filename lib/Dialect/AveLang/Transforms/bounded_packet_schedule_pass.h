#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace causalflow::avelang::dialect {

// Performs an opt-in, structurally guarded packet-load/commit fission.  The
// transform keeps the existing shared-memory commit at its consumer point,
// but issues a small raw-buffer packet before an independent MFMA loop.
// AVELANG_STAGE6Z_PACKET_SCHEDULING=bounded_consumer_point enables it.
std::unique_ptr<mlir::Pass> createBoundedPacketSchedulePass();

} // namespace causalflow::avelang::dialect
