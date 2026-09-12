#include "AveLangDialect.h"
#include "AveLangOps.h"
#include "AveLangTypes.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

// Include the generated definitions
#include "AveLangDialect.cpp.inc"

namespace causalflow::avelang::dialect {

// Define the AveLangDialect's initialize method
void AveLangDialect::initialize() {
    registerAttributes();
    registerTypes();

    addOperations<
#define GET_OP_LIST
#include "AveLangOps.cpp.inc"
        >();
}

} // namespace causalflow::avelang::dialect
