//===- AveLangAttrs.cc - generated static physical layout attributes -------===//

#include "AveLangAttrs.h"

#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <llvm/ADT/TypeSwitch.h>

#define GET_ATTRDEF_CLASSES
#include "AveLangAttrs.cpp.inc"

namespace causalflow::avelang::dialect {

::mlir::Attribute AveLangDialect::parseAttribute(
    ::mlir::DialectAsmParser &parser, ::mlir::Type type) const {
    ::llvm::StringRef mnemonic;
    ::mlir::Attribute attribute;
    if (generatedAttributeParser(parser, &mnemonic, type, attribute).has_value())
        return attribute;
    parser.emitError(parser.getCurrentLocation(), "unknown ave attribute: ")
        << mnemonic;
    return {};
}

void AveLangDialect::printAttribute(
    ::mlir::Attribute attribute, ::mlir::DialectAsmPrinter &printer) const {
    if (failed(generatedAttributePrinter(attribute, printer)))
        printer << "<unknown>";
}

void AveLangDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "AveLangAttrs.cpp.inc"
        >();
}

} // namespace causalflow::avelang::dialect
