//===- AveLangAttrs.td - static physical layout attributes -*- tablegen -*-===//
//
// These attributes carry target-aware physical layout facts without turning
// ordinary AveLang memory layouts into GPU ownership encodings.  They are
// intentionally small: C13 supports the gfx942 wave64/MFMA32 subset needed
// to preserve the recovered chunk-o plan, not an arbitrary layout algebra.
//
//===----------------------------------------------------------------------===//

#ifndef AVE_LANG_ATTRS
#define AVE_LANG_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "AveLangOps.td"

class AveLang_Attr<string name, string mnemonicName>
    : AttrDef<AveLang_Dialect, name> {
    let mnemonic = mnemonicName;
    let assemblyFormat = "`<` struct(params) `>`";
}

def DistributedEncodingAttr
    : AveLang_Attr<"DistributedEncoding", "distributed_encoding"> {
    let summary = "Static register/lane/wave ownership encoding";
    let parameters = (ins
        "mlir::DenseI64ArrayAttr":$logicalShape,
        "mlir::DenseI64ArrayAttr":$sizePerThread,
        "mlir::DenseI64ArrayAttr":$threadsPerWave,
        "mlir::DenseI64ArrayAttr":$wavesPerCTA,
        "mlir::DenseI64ArrayAttr":$order
    );
}

def SharedEncodingAttr
    : AveLang_Attr<"SharedEncoding", "shared_encoding"> {
    let summary = "Static swizzled or rotating shared-memory encoding";
    let parameters = (ins
        StringRefParameter<"kind">:$kind,
        "int64_t":$vec,
        "int64_t":$perPhase,
        "int64_t":$maxPhase,
        "mlir::DenseI64ArrayAttr":$order,
        "bool":$rotating
    );
}

def MfmaEncodingAttr
    : AveLang_Attr<"MfmaEncoding", "mfma_encoding"> {
    let summary = "Static MFMA instruction and warp geometry encoding";
    let parameters = (ins
        StringRefParameter<"target">:$target,
        "int64_t":$version,
        "mlir::DenseI64ArrayAttr":$instrShape,
        "mlir::DenseI64ArrayAttr":$warpsPerCTA,
        "bool":$isTransposed
    );
}

def DotOperandEncodingAttr
    : AveLang_Attr<"DotOperandEncoding", "dot_operand_encoding"> {
    let summary = "Static MFMA dot operand slot encoding";
    let parameters = (ins
        "int64_t":$opIdx,
        "int64_t":$kWidth,
        "mlir::Attribute":$parent
    );
}

def StaticTransformAttr
    : AveLang_Attr<"StaticTransform", "static_transform"> {
    let summary = "Fixed compile-time layout/register transform";
    let parameters = (ins
        StringRefParameter<"kind">:$kind,
        StringRefParameter<"sourceEncoding">:$sourceEncoding,
        StringRefParameter<"targetEncoding">:$targetEncoding,
        "mlir::DenseI64ArrayAttr":$permutation,
        "mlir::DenseI64ArrayAttr":$registerBasis,
        "mlir::DenseI64ArrayAttr":$laneBasis,
        "mlir::DenseI64ArrayAttr":$warpBasis,
        "int64_t":$registerStride
    );
}

#endif // AVE_LANG_ATTRS
