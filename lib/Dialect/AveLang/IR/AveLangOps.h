#pragma once

#include "AveLangDialect.h"
#include "AveLangTypes.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// Forward declare the generated classes
namespace causalflow::avelang::dialect {
class ReturnOp;
class MakeIntTupleOp;
class MakeLayoutOp;
class AveLangMemRefAllocaOp;
class AveLangMemRefLoadOp;
class AveLangMemRefLoadVecOp;
class AveLangMemRefStoreOp;
class AveLangMemRefViewOp;
class AveLangMemRefCastOp;
class AveLangMemRefExtractAlignedPointerAsIndexOp;
class AveLangMemRefSubViewOp;
class FullOp;
class EndLifetimeOp;
class NVVMMMAOp;
class NVVMLdMatrixOp;
class NVVMStMatrixOp;
class AMDGPUMfmaOp;
class AMDGPURawBufferLoadOp;
class AMDGPURawBufferStoreOp;
class AMDGPUQwenUpdateKFragLoadOp;
class AMDGPUQwenUpdateKFragLDSLoadOp;
class AMDGPUQwenPredStateKVFragLoadOp;
class AMDGPUBlockDotBF16F32Op;
class AMDGPUBlockDotMfmaOperandOp;
class AMDGPUQwenGdnRecurrenceStepBF16F32Op;
class AMDGPUQwenPersistentRecurrenceOp;
class AMDGPUQwenPersistentRecurrenceYieldOp;
class AMDGPUQwenPersistentRecurrenceEndOp;
class AMDGPUQwenKFragStageLoadOp;
class AMDGPUQwenK64PipelineStageLoadOp;
class AMDGPUQwenK64PipelineStageCommitOp;
} // namespace causalflow::avelang::dialect

// Include the generated declarations
#define GET_OP_CLASSES
#include "AveLangOps.h.inc"

namespace causalflow::avelang::dialect {

// Add custom methods to the generated MakeIntTupleOp class
// These will be added to the TableGen-generated class

} // namespace causalflow::avelang::dialect
