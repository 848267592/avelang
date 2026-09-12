#include "AveLangOps.h"
#include "AveLangDialect.h"
#include "IR/Intrinsics/amdgpu_mfma_signatures.h"
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Ptr/IR/PtrTypes.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <optional>
#include <string>

namespace causalflow::avelang::dialect {
namespace amdgpu_mfma = causalflow::avelang::amdgpu::mfma;

static bool isAveLangMemRefType(mlir::Type type) {
    return mlir::isa<MemRefType>(type);
}

namespace {

enum class VectorElemKind {
    I32,
    F16,
    F32,
    BF16,
};

struct VectorTypePattern {
    VectorElemKind elem;
    int64_t elements;
};

mlir::LogicalResult EnsureRank1Vector(mlir::Operation *op, mlir::VectorType vec,
                                      llvm::StringRef label) {
    if (vec.getRank() != 1) {
        op->emitOpError() << label << " must be a 1-D vector";
        return mlir::failure();
    }
    return mlir::success();
}

bool MatchesVectorType(mlir::VectorType vec, const VectorTypePattern &pattern) {
    if (vec.getRank() != 1 || vec.getNumElements() != pattern.elements) {
        return false;
    }

    auto elemType = vec.getElementType();
    switch (pattern.elem) {
    case VectorElemKind::I32:
        return elemType.isInteger(32);
    case VectorElemKind::F16:
        return elemType.isF16();
    case VectorElemKind::F32:
        return elemType.isF32();
    case VectorElemKind::BF16:
        return elemType.isBF16();
    }
    return false;
}

bool MatchesAnyVectorType(mlir::VectorType vec,
                          llvm::ArrayRef<VectorTypePattern> patterns) {
    for (const auto &pattern : patterns) {
        if (MatchesVectorType(vec, pattern)) {
            return true;
        }
    }
    return false;
}

struct NvvMmaSignature {
    llvm::StringRef name;
    llvm::ArrayRef<VectorTypePattern> aTypes;
    llvm::ArrayRef<VectorTypePattern> bTypes;
    llvm::ArrayRef<VectorTypePattern> cTypes;
    llvm::ArrayRef<VectorTypePattern> rTypes;
};

static const VectorTypePattern kNvvmMma16x8x16A[] = {
    {VectorElemKind::I32, 4},
    {VectorElemKind::F16, 8},
};
static const VectorTypePattern kNvvmMma16x8x16B[] = {
    {VectorElemKind::I32, 2},
    {VectorElemKind::F16, 4},
};
static const VectorTypePattern kNvvmMma16x8x16C[] = {
    {VectorElemKind::F16, 4},
};
static const VectorTypePattern kNvvmMma16x8x16R[] = {
    {VectorElemKind::F16, 4},
};

static const VectorTypePattern kNvvmMma16x8x8A[] = {
    {VectorElemKind::I32, 2},
    {VectorElemKind::F16, 4},
};
static const VectorTypePattern kNvvmMma16x8x8B[] = {
    {VectorElemKind::I32, 1},
    {VectorElemKind::F16, 2},
};
static const VectorTypePattern kNvvmMma16x8x8C[] = {
    {VectorElemKind::F32, 4},
};
static const VectorTypePattern kNvvmMma16x8x8R[] = {
    {VectorElemKind::F32, 4},
};

static const NvvMmaSignature kNvvmMmaSignatures[] = {
    {
        "mma_16x8x16_f16_f16",
        llvm::ArrayRef(kNvvmMma16x8x16A),
        llvm::ArrayRef(kNvvmMma16x8x16B),
        llvm::ArrayRef(kNvvmMma16x8x16C),
        llvm::ArrayRef(kNvvmMma16x8x16R),
    },
    {
        "mma_16x8x8_f16_f32",
        llvm::ArrayRef(kNvvmMma16x8x8A),
        llvm::ArrayRef(kNvvmMma16x8x8B),
        llvm::ArrayRef(kNvvmMma16x8x8C),
        llvm::ArrayRef(kNvvmMma16x8x8R),
    },
};

static std::string BuildAmdgpuMfmaSignatureList() {
    std::string list;
    for (const auto &cfg : amdgpu_mfma::MFMAConfig::GetConfigs()) {
        if (!list.empty()) {
            list += ", ";
        }
        list += cfg.name.str();
    }
    return list;
}

static std::optional<int64_t> getElementByteSize(mlir::Type elementType) {
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(elementType)) {
        auto scalarType = vectorType.getElementType();
        if (scalarType.isIndex()) {
            return 8 * vectorType.getNumElements();
        }
        if (!scalarType.isIntOrFloat()) {
            return std::nullopt;
        }
        int64_t bitWidth = scalarType.getIntOrFloatBitWidth();
        int64_t elementBytes = (bitWidth + 7) / 8;
        if (elementBytes <= 0) {
            return std::nullopt;
        }
        return elementBytes * vectorType.getNumElements();
    }

    if (elementType.isIndex()) {
        return 8;
    }

    if (!elementType.isIntOrFloat()) {
        return std::nullopt;
    }

    int64_t bitWidth = elementType.getIntOrFloatBitWidth();
    int64_t elementBytes = (bitWidth + 7) / 8;
    if (elementBytes <= 0) {
        return std::nullopt;
    }
    return elementBytes;
}

static std::optional<int64_t> getStaticTotalByteSize(MemRefType memrefType) {
    auto elementBytes = getElementByteSize(memrefType.getElementType());
    if (!elementBytes) {
        return std::nullopt;
    }

    int64_t elementCount = 1;
    for (auto dim : memrefType.getShape()) {
        if (mlir::ShapedType::isDynamic(dim) || dim < 0) {
            return std::nullopt;
        }
        elementCount *= dim;
    }

    return elementCount * (*elementBytes);
}

static int64_t countTupleElements(mlir::Value value) {
    if (auto tupleOp = value.getDefiningOp<MakeIntTupleOp>()) {
        int64_t count = 0;
        for (auto elem : tupleOp.getElements()) {
            count += countTupleElements(elem);
        }
        return count;
    }
    return 1;
}

} // namespace

// Custom build method
void MakeIntTupleOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state,
                           mlir::ValueRange elements) {
    // Set the result type to NoneType (as a placeholder for tuple type)
    state.addTypes(builder.getNoneType());

    // Add all elements as operands
    state.addOperands(elements);

    // Add an attribute to mark this as an ave-lang tuple
    state.addAttribute("is_tuple", builder.getBoolAttr(true));
}

// Custom verify method
mlir::LogicalResult MakeIntTupleOp::verify() {
    // Verify that all operands are integer or index types, or nested tuples
    for (auto operand : getOperands()) {
        auto type = operand.getType();

        // Allow integer types, index types, or None types (for nested tuples)
        if (!type.isIntOrIndex() && !mlir::isa<mlir::NoneType>(type)) {
            return emitOpError(
                "all elements must be integers, indices, or tuples");
        }
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AveLangMemRefLoadOp
//===----------------------------------------------------------------------===//

void AveLangMemRefLoadOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(mlir::MemoryEffects::Read::get(),
                         mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// AveLangMemRefLoadVecOp
//===----------------------------------------------------------------------===//

void AveLangMemRefLoadVecOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(mlir::MemoryEffects::Read::get(),
                         mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// AveLangMemRefStoreOp
//===----------------------------------------------------------------------===//

void AveLangMemRefStoreOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
                         mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// MakeLayoutOp
//===----------------------------------------------------------------------===//

void MakeLayoutOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value dims, mlir::Value stride) {
    state.addOperands({dims, stride});
    // FIXME: Use OpaqueType to represent the layout type
    auto layoutType =
        mlir::OpaqueType::get(builder.getStringAttr("ave"), "layout");
    state.addTypes(layoutType);
}

mlir::LogicalResult MakeLayoutOp::verify() {
    // Verify that dims and stride are tuples (MakeIntTupleOp)
    if (!getDims().getDefiningOp<MakeIntTupleOp>()) {
        return emitOpError("dims must be a tuple created by make_int_tuple");
    }
    if (!getStride().getDefiningOp<MakeIntTupleOp>()) {
        return emitOpError("stride must be a tuple created by make_int_tuple");
    }

    auto dimsTuple = getDims().getDefiningOp<MakeIntTupleOp>();
    auto strideTuple = getStride().getDefiningOp<MakeIntTupleOp>();

    // Verify that dims and stride have the same number of elements
    if (dimsTuple.getNumElements() != strideTuple.getNumElements()) {
        return emitOpError(
            "dims and stride must have the same number of elements");
    }

    // Verify that dims and stride are not empty
    if (dimsTuple.getNumElements() == 0) {
        return emitOpError("dims and stride cannot be empty");
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AveLangMemRefCastOp
//===----------------------------------------------------------------------===//

void AveLangMemRefCastOp::build(mlir::OpBuilder &builder,
                                mlir::OperationState &state, mlir::Value source,
                                mlir::Type resultType) {
    state.addOperands({source});
    state.addTypes(resultType);
}

void AveLangMemRefCastOp::build(mlir::OpBuilder &builder,
                                mlir::OperationState &state, mlir::Value source,
                                mlir::Value layout, mlir::Type resultType) {
    state.addOperands({source, layout});
    state.addTypes(resultType);
}

mlir::LogicalResult AveLangMemRefCastOp::verify() {
    auto sourceType = getSource().getType();
    auto sourceMemref = mlir::dyn_cast<MemRefType>(sourceType);
    auto sourceBuiltinMemref = mlir::dyn_cast<mlir::MemRefType>(sourceType);
    auto sourcePtr = mlir::dyn_cast<mlir::ptr::PtrType>(sourceType);
    auto resultType = getResult().getType();
    auto resultMemref = mlir::dyn_cast<MemRefType>(resultType);
    auto resultBuiltinMemref = mlir::dyn_cast<mlir::MemRefType>(resultType);
    if ((!sourceMemref && !sourceBuiltinMemref && !sourcePtr) ||
        (!resultMemref && !resultBuiltinMemref)) {
        return emitOpError("source must be a memref or ptr type; result must "
                           "be a memref type");
    }

    auto resultRank =
        resultMemref ? resultMemref.getRank() : resultBuiltinMemref.getRank();

    bool hasLayout = false;
    if (auto layoutValue = getLayout()) {
        hasLayout = true;
        if (!layoutValue.getDefiningOp<MakeLayoutOp>()) {
            return emitOpError("layout must be a value created by make_layout");
        }

        auto layoutOp = layoutValue.getDefiningOp<MakeLayoutOp>();
        auto dimsValue = layoutOp.getDims();
        auto strideValue = layoutOp.getStride();

        auto dimsTuple = dimsValue.getDefiningOp<MakeIntTupleOp>();
        auto strideTuple = strideValue.getDefiningOp<MakeIntTupleOp>();

        if (!dimsTuple || !strideTuple) {
            return emitOpError(
                "layout must contain valid dims and stride tuples");
        }

        int64_t dimsCount = countTupleElements(dimsValue);
        int64_t strideCount = countTupleElements(strideValue);
        if (dimsCount != strideCount) {
            return emitOpError(
                "layout dims and stride must have the same number of elements");
        }

        if (dimsCount != resultRank) {
            return emitOpError("layout dims must match the result memref rank");
        }
    }

    if (!hasLayout && sourceMemref && resultMemref) {
        auto sourceBytes = getStaticTotalByteSize(sourceMemref);
        auto resultBytes = getStaticTotalByteSize(resultMemref);
        if (!sourceBytes || !resultBytes) {
            return emitOpError(
                "layout is required when casting dynamic shapes or unsupported "
                "element types");
        }
        if (*sourceBytes != *resultBytes) {
            return emitOpError("The source memref and target type must have "
                               "the same total byte size");
        }
    }

    if (!hasLayout && sourcePtr) {
        if (resultMemref) {
            auto resultBytes = getStaticTotalByteSize(resultMemref);
            if (!resultBytes) {
                return emitOpError(
                    "layout is required when casting from ptr with dynamic "
                    "shapes or unsupported element types");
            }
        }
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// GPU Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// NVVMMMAOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult NVVMMMAOp::verify() {
    auto aVector = mlir::dyn_cast<mlir::VectorType>(getA().getType());
    auto bVector = mlir::dyn_cast<mlir::VectorType>(getB().getType());
    auto cVector = mlir::dyn_cast<mlir::VectorType>(getC().getType());
    auto resultVector = mlir::dyn_cast<mlir::VectorType>(getResult().getType());

    if (!aVector || !bVector || !cVector) {
        return emitOpError("all operands must be vector types");
    }
    if (!resultVector) {
        return emitOpError("result must be a vector type");
    }

    if (mlir::failed(EnsureRank1Vector(getOperation(), aVector, "A")) ||
        mlir::failed(EnsureRank1Vector(getOperation(), bVector, "B")) ||
        mlir::failed(EnsureRank1Vector(getOperation(), cVector, "C")) ||
        mlir::failed(
            EnsureRank1Vector(getOperation(), resultVector, "result"))) {
        return mlir::failure();
    }

    for (const auto &sig : kNvvmMmaSignatures) {
        if (MatchesAnyVectorType(aVector, sig.aTypes) &&
            MatchesAnyVectorType(bVector, sig.bTypes) &&
            MatchesAnyVectorType(cVector, sig.cTypes) &&
            MatchesAnyVectorType(resultVector, sig.rTypes)) {
            return mlir::success();
        }
    }

    return emitOpError(
        "operand types do not match any supported NVVM MMA signature; "
        "supported: mma_16x8x16_f16_f16, mma_16x8x8_f16_f32");
}

//===----------------------------------------------------------------------===//
// NVVMLdMatrixOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult NVVMLdMatrixOp::verify() {
    // Verify memref operand
    if (!isAveLangMemRefType(getMemref().getType())) {
        return emitOpError("memref operand must be an ave-lang memref type");
    }

    // Verify matrix shape is valid
    auto shape = getMatrixShape();
    if (shape != "m8n8" && shape != "m16n16") {
        return emitOpError("unsupported matrix shape: " + shape);
    }

    // Verify matrix num is valid (1, 2, 4)
    auto num = getMatrixNum();
    if (num != 1 && num != 2 && num != 4) {
        return emitOpError("matrix num must be 1, 2, or 4");
    }

    // Verify bit width (8, 16)
    auto bitWidth = getMatrixBitWidth();
    if (bitWidth != 8 && bitWidth != 16) {
        return emitOpError("matrix bit width must be 8 or 16");
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// NVVMStMatrixOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult NVVMStMatrixOp::verify() {
    // Verify memref operand
    if (!isAveLangMemRefType(getMemref().getType())) {
        return emitOpError("memref operand must be an ave-lang memref type");
    }

    // Verify matrix shape is valid
    auto shape = getMatrixShape();
    if (shape != "m8n8" && shape != "m16n16") {
        return emitOpError("unsupported matrix shape: " + shape);
    }

    // Verify matrix num is valid (1, 2, 4)
    auto num = getMatrixNum();
    if (num != 1 && num != 2 && num != 4) {
        return emitOpError("matrix num must be 1, 2, or 4");
    }

    // Verify bit width (8, 16)
    auto bitWidth = getMatrixBitWidth();
    if (bitWidth != 8 && bitWidth != 16) {
        return emitOpError("matrix bit width must be 8 or 16");
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPUMfmaOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPUMfmaOp::verify() {
    auto aVector = mlir::dyn_cast<mlir::VectorType>(getA().getType());
    auto bVector = mlir::dyn_cast<mlir::VectorType>(getB().getType());
    auto cVector = mlir::dyn_cast<mlir::VectorType>(getC().getType());
    auto resultVector = mlir::dyn_cast<mlir::VectorType>(getResult().getType());

    if (!aVector || !bVector || !cVector) {
        return emitOpError("all operands must be vector types");
    }
    if (!resultVector) {
        return emitOpError("result must be a vector type");
    }

    if (mlir::failed(EnsureRank1Vector(getOperation(), aVector, "A")) ||
        mlir::failed(EnsureRank1Vector(getOperation(), bVector, "B")) ||
        mlir::failed(EnsureRank1Vector(getOperation(), cVector, "C")) ||
        mlir::failed(
            EnsureRank1Vector(getOperation(), resultVector, "result"))) {
        return mlir::failure();
    }

    auto typeAName = getTypeAAttr().getValue();
    auto typeCName = getTypeCAttr().getValue();

    const amdgpu_mfma::MFMAConfig *config = amdgpu_mfma::MFMAConfig::Find(
        getMAttr().getInt(), getNAttr().getInt(), getKAttr().getInt(),
        typeAName, typeCName);

    if (!config) {
        return emitOpError("unsupported MFMA configuration; supported: " +
                           BuildAmdgpuMfmaSignatureList());
    }

    if (!config->MatchesAType(aVector) || !config->MatchesBType(bVector) ||
        !config->MatchesCType(cVector) || !config->MatchesCType(resultVector)) {
        return emitOpError("operand types do not match MFMA signature")
               << " " << config->name;
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPURawBufferLoadOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPURawBufferLoadOp::verify() {
    // Verify result type (can be scalar or vector)
    auto resultType = getResult().getType();
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(resultType)) {
        if (vectorType.getRank() != 1) {
            return emitOpError("result vector must be 1-dimensional");
        }
        auto numElements = vectorType.getNumElements();
        if (numElements != 1 && numElements != 2 && numElements != 4) {
            return emitOpError("result vector must have 1, 2 or 4 elements");
        }
    }

    // All operands should be integer types for buffer addressing
    for (auto operand : {getVindex(), getSoffset(), getAux()}) {
        if (!operand.getType().isIntOrIndex()) {
            return emitOpError(
                "buffer operands must be integer or index types");
        }
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPURawBufferStoreOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPURawBufferStoreOp::verify() {
    // Verify data type (can be scalar or vector)
    auto dataType = getData().getType();
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(dataType)) {
        if (vectorType.getRank() != 1) {
            return emitOpError("data vector must be 1-dimensional");
        }
        auto numElements = vectorType.getNumElements();
        if (numElements != 1 && numElements != 2 && numElements != 4) {
            return emitOpError("data vector must have 1, 2 or 4 elements");
        }
    }

    // All other operands should be integer types for buffer addressing
    for (auto operand : {getVindex(), getSoffset(), getAux()}) {
        if (!operand.getType().isIntOrIndex()) {
            return emitOpError(
                "buffer operands must be integer or index types");
        }
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPUQwenUpdateKFragLoadOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPUQwenUpdateKFragLoadOp::verify() {
    auto resultType = mlir::dyn_cast<mlir::VectorType>(getResult().getType());

    auto getShape = [](mlir::Type type) -> llvm::ArrayRef<int64_t> {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getShape();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getShape();
        }
        return {};
    };
    auto getElementType = [](mlir::Type type) -> mlir::Type {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getElementType();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getElementType();
        }
        return {};
    };
    auto getMemorySpace = [](mlir::Type type) -> mlir::Attribute {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getMemorySpace();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getMemorySpace();
        }
        return {};
    };

    if (getShape(getSharedK().getType()) !=
            llvm::ArrayRef<int64_t>({128, 64}) ||
        !getElementType(getSharedK().getType()).isBF16()) {
        return emitOpError("shared K must be BF16 [128,64]");
    }
    auto sharedSpace = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
        getMemorySpace(getSharedK().getType()));
    if (!sharedSpace ||
        sharedSpace.getValue() != mlir::gpu::AddressSpace::Workgroup) {
        return emitOpError("shared K must use workgroup memory");
    }
    auto sourceShape = getShape(getSourceK().getType());
    if (sourceShape.size() != 4 || sourceShape[0] != 1 || sourceShape[1] < 64 ||
        sourceShape[2] != 4 || sourceShape[3] != 128 ||
        !getElementType(getSourceK().getType()).isBF16()) {
        return emitOpError("source K must be BF16 [1,T,4,128] with T >= 64");
    }
    if (!resultType || resultType.getRank() != 1 ||
        resultType.getNumElements() != 4 ||
        !resultType.getElementType().isBF16()) {
        return emitOpError("result must be vector<4xbf16>");
    }
    for (auto value : {getThreadId(), getKeyHead(), getTokenWindowBase(),
                       getKColumn(), getTokenFragmentBase()}) {
        if (!value.getType().isIntOrIndex()) {
            return emitOpError("index operands must be integer or index");
        }
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPUQwenUpdateKFragLDSLoadOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPUQwenUpdateKFragLDSLoadOp::verify() {
    auto resultType = mlir::dyn_cast<mlir::VectorType>(getResult().getType());
    if (!resultType || resultType.getRank() != 1 ||
        resultType.getNumElements() != 4 ||
        !resultType.getElementType().isBF16()) {
        return emitOpError("result must be vector<4xbf16>");
    }
    for (auto value :
         {getSharedBase(), getKColumn(), getTokenFragmentLocalBase()}) {
        if (!value.getType().isIntOrIndex()) {
            return emitOpError("address operands must be integer or index");
        }
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPUBlockDotBF16F32Op
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPUBlockDotBF16F32Op::verify() {
    auto getShape = [](mlir::Type type) -> llvm::ArrayRef<int64_t> {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getShape();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getShape();
        }
        return {};
    };
    auto getElementType = [](mlir::Type type) -> mlir::Type {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getElementType();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getElementType();
        }
        return {};
    };
    auto getMemorySpace = [](mlir::Type type) -> mlir::Attribute {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getMemorySpace();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getMemorySpace();
        }
        return {};
    };
    auto isWorkgroupBF16 = [&](mlir::Value value,
                               llvm::ArrayRef<int64_t> shape) {
        auto space = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
            getMemorySpace(value.getType()));
        return getShape(value.getType()) == shape &&
               getElementType(value.getType()).isBF16() && space &&
               space.getValue() == mlir::gpu::AddressSpace::Workgroup;
    };
    if (getOperation()->hasAttr("avelang.block_dot.operand_mode")) {
        auto isOperandBlock = [&](mlir::Value value) {
            auto shape = getShape(value.getType());
            auto space = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
                getMemorySpace(value.getType()));
            return shape.size() == 2 && shape[0] >= 32 && shape[1] >= 32 &&
                   getElementType(value.getType()).isBF16() && space &&
                   space.getValue() == mlir::gpu::AddressSpace::Workgroup;
        };
        if (!isOperandBlock(getAStage()) || !isOperandBlock(getBStage())) {
            return emitOpError("operand mode expects workgroup BF16 rank-2 "
                               "logical A/B blocks with at least 32 elements");
        }
        auto resultType = mlir::dyn_cast<mlir::VectorType>(getResult().getType());
        if (!resultType || resultType.getRank() != 1 ||
            resultType.getNumElements() != 32 ||
            !resultType.getElementType().isF32()) {
            return emitOpError("operand mode result must be vector<32xf32>");
        }
        for (auto value : {getThreadId(), getChunkStart(), getValueHead(),
                           getKeyHead(), getValueBase(), getKHalf()}) {
            if (!value.getType().isIntOrIndex()) {
                return emitOpError("operand mode index operands must be integer "
                                   "or index");
            }
        }
        if (!getGLast().getType().isF32()) {
            return emitOpError("operand mode g_last must be f32");
        }
        for (auto accumulator : {getAccLow(), getAccHigh()}) {
            auto type = accumulator.getType();
            auto vectorType = mlir::dyn_cast<mlir::VectorType>(type);
            const bool isVector = vectorType && vectorType.getRank() == 1 &&
                                  vectorType.getNumElements() == 16 &&
                                  vectorType.getElementType().isF32();
            const bool isMemRef = getShape(type) ==
                                      llvm::ArrayRef<int64_t>({16}) &&
                                  getElementType(type).isF32();
            if (!isVector && !isMemRef) {
                return emitOpError("operand mode accumulators must be "
                                   "vector<16xf32> or local F32 [16]");
            }
        }
        return mlir::success();
    }
    const bool isIndependentA = isWorkgroupBF16(getAStage(), {2, 32, 32});
    const bool isCooperativeA = isWorkgroupBF16(getAStage(), {1, 32, 32});
    const bool isStagedA = isWorkgroupBF16(getAStage(), {1, 32, 64});
    const bool isRegularB = isWorkgroupBF16(getBStage(), {32, 32});
    const bool isStagedB = isWorkgroupBF16(getBStage(), {64, 64});
    const bool isPreloadedB =
        getOperation()->hasAttr("avelang.block_dot.preloaded_k") &&
        isWorkgroupBF16(getBStage(), {2, 64, 64});
    if ((!isIndependentA && !isCooperativeA && !isStagedA) ||
        (!isRegularB && !isStagedB && !isPreloadedB) ||
        (isStagedA != (isStagedB || isPreloadedB))) {
        return emitOpError("expects workgroup BF16 A=[2,32,32] or cooperative "
                           "A=[1,32,32] and B=[32,32] staging; staged V-decay "
                           "uses A=[1,32,64] with B=[64,64] or preloaded K=[2,64,64]");
    }
    auto kShape = getShape(getSourceK().getType());
    auto vShape = getShape(getSourceVNew().getType());
    auto gShape = getShape(getSourceG().getType());
    const auto sourceRole = getOperation()->getAttrOfType<mlir::StringAttr>(
        "avelang.block_dot.source_role");
    const bool c18VSource = sourceRole && sourceRole.getValue() == "V" &&
                            getSourceK() == getSourceVNew();
    const bool c19QSource = sourceRole && sourceRole.getValue() == "Q" &&
                            getSourceK() == getSourceVNew();
    if ((!c18VSource && !c19QSource &&
         (kShape.size() != 4 || kShape[0] != 1 || kShape[1] < 64 ||
          kShape[2] != 4 || kShape[3] != 128)) ||
        !getElementType(getSourceK().getType()).isBF16()) {
        return emitOpError("source K must be BF16 [1,T,4,128] with T >= 64");
    }
    if (!c19QSource &&
        (vShape.size() != 4 || vShape[0] != 1 || vShape[1] < 64 ||
        vShape[2] != 8 || vShape[3] != 128 ||
        !getElementType(getSourceVNew().getType()).isBF16())) {
        return emitOpError(
            "source V-new must be BF16 [1,T,8,128] with T >= 64");
    }
    if (gShape.size() != 3 || gShape[0] != 1 || gShape[1] < 64 ||
        gShape[2] != 8 || !getElementType(getSourceG().getType()).isF32()) {
        return emitOpError("source G must be FP32 [1,T,8] with T >= 64");
    }
    auto resultType = mlir::dyn_cast<mlir::VectorType>(getResult().getType());
    if (!resultType || resultType.getRank() != 1 ||
        resultType.getNumElements() != 32 ||
        !resultType.getElementType().isF32()) {
        return emitOpError("result must be vector<32xf32>");
    }
    for (auto value : {getThreadId(), getChunkStart(), getValueHead(),
                       getKeyHead(), getValueBase(), getKHalf()}) {
        if (!value.getType().isIntOrIndex()) {
            return emitOpError("index operands must be integer or index");
        }
    }
    if (!getGLast().getType().isF32()) {
        return emitOpError("g_last must be f32");
    }
    for (auto accumulator : {getAccLow(), getAccHigh()}) {
        auto type = accumulator.getType();
        auto isVector = [&](mlir::Type valueType) {
            auto vectorType = mlir::dyn_cast<mlir::VectorType>(valueType);
            return vectorType && vectorType.getRank() == 1 &&
                   vectorType.getNumElements() == 16 &&
                   vectorType.getElementType().isF32();
        };
        auto isMemRef = [&](mlir::Type valueType) {
            return getShape(valueType) == llvm::ArrayRef<int64_t>({16}) &&
                   getElementType(valueType).isF32();
        };
        if (!isVector(type) && !isMemRef(type)) {
            return emitOpError("persistent accumulators must be vector<16xf32> "
                               "or F32 [16] local storage");
        }
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPUQwenGdnRecurrenceStepBF16F32Op
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPUQwenGdnRecurrenceStepBF16F32Op::verify() {
    auto getShape = [](mlir::Type type) -> llvm::ArrayRef<int64_t> {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getShape();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getShape();
        }
        return {};
    };
    auto getElementType = [](mlir::Type type) -> mlir::Type {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getElementType();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getElementType();
        }
        return {};
    };
    auto getMemorySpace = [](mlir::Type type) -> mlir::Attribute {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getMemorySpace();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getMemorySpace();
        }
        return {};
    };
    auto isWorkgroup = [&](mlir::Value value, llvm::ArrayRef<int64_t> shape,
                           mlir::Type element) {
        auto space = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
            getMemorySpace(value.getType()));
        return getShape(value.getType()) == shape &&
               getElementType(value.getType()) == element && space &&
               space.getValue() == mlir::gpu::AddressSpace::Workgroup;
    };
    auto *context = getContext();
    if (!isWorkgroup(getStateStage(), {2, 32, 64},
                     mlir::BFloat16Type::get(context))) {
        return emitOpError("stateStage must be workgroup BF16 [2,32,64]");
    }
    if (!isWorkgroup(getPhaseStage(), {64, 64},
                     mlir::BFloat16Type::get(context)) ||
        !isWorkgroup(getPredPartial(), {2, 32, 32},
                     mlir::Float32Type::get(context)) ||
        !isWorkgroup(getVdecayStage(), {1, 32, 32},
                     mlir::BFloat16Type::get(context))) {
        return emitOpError("requires phase=[64,64] BF16, pred=[2,32,32] F32 "
                           "and vdecay=[1,32,32] BF16 workgroup buffers");
    }
    auto vector16F32 = [](mlir::Type type) {
        auto vectorType = mlir::dyn_cast<mlir::VectorType>(type);
        return vectorType && vectorType.getRank() == 1 &&
               vectorType.getNumElements() == 16 &&
               vectorType.getElementType().isF32();
    };
    auto local16F32 = [&](mlir::Type type) {
        return getShape(type) == llvm::ArrayRef<int64_t>({16}) &&
               getElementType(type).isF32();
    };
    auto vector32F32 = [](mlir::Type type) {
        auto vectorType = mlir::dyn_cast<mlir::VectorType>(type);
        return vectorType && vectorType.getRank() == 1 &&
               vectorType.getNumElements() == 32 &&
               vectorType.getElementType().isF32();
    };
    if ((!vector16F32(getStateLow().getType()) &&
         !local16F32(getStateLow().getType())) ||
        (!vector16F32(getStateHigh().getType()) &&
         !local16F32(getStateHigh().getType())) ||
        !vector32F32(getResult().getType())) {
        return emitOpError("state operands must be vector<16xf32> or local "
                           "FP32 [16], and result must be vector<32xf32>");
    }
    for (auto value : {getThreadId(), getChunkIndex(), getValueHead(),
                       getKeyHead(), getValueBase()}) {
        if (!value.getType().isIntOrIndex()) {
            return emitOpError(
                "thread/chunk/head/index operands must be integer "
                "or index");
        }
    }
    if (!getEmitAudit().getType().isInteger(1)) {
        return emitOpError("emitAudit must be i1");
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPUQwenPersistentRecurrenceOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPUQwenPersistentRecurrenceOp::verify() {
    if (!getNumChunks().getType().isIntOrIndex()) {
        return emitOpError("num_chunks must be integer or index");
    }
    if (!getEmitAudit().getType().isInteger(1)) {
        return emitOpError("emit_audit must be i1");
    }
    const bool frontendMarker = getOperation()->hasAttr(
        "avelang.qwen.persistent_recurrence.frontend_marker");
    if (getBody().empty() || !llvm::hasSingleElement(getBody())) {
        return emitOpError("must have exactly one body block");
    }
    auto &body = getBody().front();
    if (body.empty() ||
        !mlir::isa<AMDGPUQwenPersistentRecurrenceYieldOp>(body.back())) {
        return emitOpError("body must end in persistent recurrence yield");
    }
    if (frontendMarker && !llvm::hasSingleElement(body)) {
        return emitOpError("frontend marker must contain only its yield");
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPUQwenKFragStageLoadOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPUQwenKFragStageLoadOp::verify() {
    auto getShape = [](mlir::Type type) -> llvm::ArrayRef<int64_t> {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getShape();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getShape();
        }
        return {};
    };
    auto getElementType = [](mlir::Type type) -> mlir::Type {
        if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
            return aveType.getElementType();
        }
        if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
            return memrefType.getElementType();
        }
        return {};
    };
    auto sourceShape = getShape(getSourceK().getType());
    if (sourceShape.size() != 4 || sourceShape[0] != 1 || sourceShape[1] < 64 ||
        sourceShape[2] != 4 || sourceShape[3] != 128 ||
        !getElementType(getSourceK().getType()).isBF16()) {
        return emitOpError("source K must be BF16 [1,T,4,128] with T >= 64");
    }
    if (!getResult().getType().isBF16()) {
        return emitOpError("result must be BF16");
    }
    for (auto value : {getSourceToken(), getKeyHead(), getKColumn()}) {
        if (!value.getType().isIntOrIndex()) {
            return emitOpError("index operands must be integer or index");
        }
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPUQwenK64PipelineStageLoadOp / CommitOp
//===----------------------------------------------------------------------===//

namespace {

llvm::ArrayRef<int64_t> getMemRefShape(mlir::Type type) {
    if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
        return aveType.getShape();
    }
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        return memrefType.getShape();
    }
    return {};
}

mlir::Type getMemRefElementType(mlir::Type type) {
    if (auto aveType = mlir::dyn_cast<MemRefType>(type)) {
        return aveType.getElementType();
    }
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        return memrefType.getElementType();
    }
    return {};
}

} // namespace

mlir::LogicalResult AMDGPUQwenK64PipelineStageLoadOp::verify() {
    auto sourceShape = getMemRefShape(getSourceK().getType());
    if (sourceShape.size() != 4 || sourceShape[0] != 1 || sourceShape[1] < 64 ||
        (sourceShape[2] != 4 && sourceShape[2] != 8) || sourceShape[3] != 128 ||
        !getMemRefElementType(getSourceK().getType()).isBF16()) {
        return emitOpError("source operand must be BF16 [1,T,4|8,128] with T >= 64");
    }
    if (!getStageToken().getType().isInteger(64)) {
        return emitOpError("opaque stage token must be i64");
    }
    for (auto value : {getThreadId(), getChunkStart(), getKeyHead(), getKHalf()}) {
        if (!value.getType().isIntOrIndex()) {
            return emitOpError("thread/chunk/head/K-half operands must be integer or index");
        }
    }
    return mlir::success();
}

mlir::LogicalResult AMDGPUQwenK64PipelineStageCommitOp::verify() {
    // The Python/JIT expression layer may insert a trivial integer wrapper
    // around this opaque token. Its producer is therefore checked after GPU
    // outlining by the late lowering pass, where the wrapper can be unwrapped
    // without making the token frontend-addressable.
    if (!getStageToken().getType().isInteger(64)) {
        return emitOpError("opaque stage token must be i64");
    }
    auto shape = getMemRefShape(getSharedKBank().getType());
    if (shape != llvm::ArrayRef<int64_t>({2, 64, 64}) ||
        !getMemRefElementType(getSharedKBank().getType()).isBF16()) {
        return emitOpError("shared W/K bank must be BF16 [2,64,64]");
    }
    // After AveLang-to-memref, workgroup space may be represented as the
    // target's normalized integer address-space attribute. The source-facing
    // intrinsic checker enforces workgroup memory before this conversion; do
    // not reject the equivalent late representation here.
    return mlir::success();
}

mlir::LogicalResult AMDGPUQwenK64CoreIssueOp::verify() {
    auto sourceShape = getMemRefShape(getSourceK().getType());
    if (sourceShape.size() != 4 || sourceShape[0] != 1 || sourceShape[1] < 64 ||
        (sourceShape[2] != 4 && sourceShape[2] != 8) || sourceShape[3] != 128 ||
        !getMemRefElementType(getSourceK().getType()).isBF16()) {
        return emitOpError("source operand must be BF16 [1,T,4|8,128] with T >= 64");
    }
    auto packetType = mlir::dyn_cast<mlir::VectorType>(getPacket().getType());
    if (!packetType || packetType.getRank() != 1 ||
        packetType.getNumElements() != 8 || !packetType.getElementType().isBF16()) {
        return emitOpError("packet must be vector<8xbf16>");
    }
    auto packetStart = getOperation()->getAttrOfType<mlir::IntegerAttr>(
        "avelang.qwen.core_staggered.packet_start");
    if (!packetStart || packetStart.getInt() < 0 || packetStart.getInt() >= 4) {
        return emitOpError("requires core_staggered.packet_start in [0,4)");
    }
    for (auto value : {getThreadId(), getChunkStart(), getKeyHead(), getKHalf()}) {
        if (!value.getType().isIntOrIndex()) {
            return emitOpError("thread/chunk/head/K-half operands must be integer or index");
        }
    }
    return mlir::success();
}

mlir::LogicalResult AMDGPUQwenK64CoreCommitOp::verify() {
    auto packetType = mlir::dyn_cast<mlir::VectorType>(getPacket().getType());
    if (!packetType || packetType.getRank() != 1 ||
        packetType.getNumElements() != 8 || !packetType.getElementType().isBF16()) {
        return emitOpError("packet must be vector<8xbf16>");
    }
    auto shape = getMemRefShape(getSharedKBank().getType());
    if (shape != llvm::ArrayRef<int64_t>({2, 64, 64}) ||
        !getMemRefElementType(getSharedKBank().getType()).isBF16()) {
        return emitOpError("shared W/K bank must be BF16 [2,64,64]");
    }
    auto packetStart = getOperation()->getAttrOfType<mlir::IntegerAttr>(
        "avelang.qwen.core_staggered.packet_start");
    if (!packetStart || packetStart.getInt() < 0 || packetStart.getInt() >= 4) {
        return emitOpError("requires core_staggered.packet_start in [0,4)");
    }
    for (auto value : {getThreadId(), getKHalf()}) {
        if (!value.getType().isIntOrIndex()) {
            return emitOpError("thread/K-half operands must be integer or index");
        }
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPURegionPendingPacketIssueOp / CommitOp
//===----------------------------------------------------------------------===//

namespace {

mlir::LogicalResult verifyPendingPacketType(mlir::Operation *operation,
                                            mlir::Type type) {
    auto packet = mlir::dyn_cast<mlir::VectorType>(type);
    if (!packet || packet.getRank() != 1 || packet.getNumElements() != 8 ||
        !packet.getElementType().isBF16()) {
        return operation->emitOpError("packet must be vector<8xbf16>");
    }
    return mlir::success();
}

mlir::LogicalResult verifyPendingPacketIndices(mlir::Operation *operation,
                                               mlir::Type memrefType,
                                               mlir::OperandRange indices) {
    const auto shape = getMemRefShape(memrefType);
    if (shape.empty() || !getMemRefElementType(memrefType).isBF16()) {
        return operation->emitOpError("memory operand must be a BF16 memref");
    }
    if (indices.size() != shape.size()) {
        return operation->emitOpError("index count must match the memory rank");
    }
    for (auto index : indices) {
        if (!index.getType().isIntOrIndex()) {
            return operation->emitOpError("packet indices must be integer or index");
        }
    }
    return mlir::success();
}

} // namespace

mlir::LogicalResult AMDGPURegionPendingPacketIssueOp::verify() {
    if (mlir::failed(verifyPendingPacketType(getOperation(), getPacket().getType()))) {
        return mlir::failure();
    }
    if (!getPredicate().getType().isInteger(1)) {
        return emitOpError("predicate must be i1");
    }
    return verifyPendingPacketIndices(getOperation(), getSource().getType(),
                                      getIndices());
}

mlir::LogicalResult AMDGPURegionPendingPacketCommitOp::verify() {
    if (mlir::failed(verifyPendingPacketType(getOperation(), getPacket().getType()))) {
        return mlir::failure();
    }
    if (!getPredicate().getType().isInteger(1)) {
        return emitOpError("predicate must be i1");
    }
    auto issue = mlir::dyn_cast_or_null<AMDGPURegionPendingPacketIssueOp>(
        getPacket().getDefiningOp());
    if (!issue || issue->getBlock() != getOperation()->getBlock()) {
        return emitOpError(
            "requires a direct same-block region_pending_packet_issue producer");
    }
    return verifyPendingPacketIndices(getOperation(), getDestination().getType(),
                                      getIndices());
}

} // namespace causalflow::avelang::dialect

// Include the generated definitions
#define GET_OP_CLASSES
#include "AveLangOps.cpp.inc"

// Manual implementations for missing generic build methods
namespace causalflow::avelang::dialect {

// NVVMMMAOp build method
void NVVMMMAOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::ValueRange operands, mlir::TypeRange resultTypes,
                      mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 3u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    assert(resultTypes.size() == 1u && "mismatched number of return types");
    state.addTypes(resultTypes);
}

// NVVMLdMatrixOp build method - This is already generated, but let's ensure
// it's correct (This method already exists in the generated code)

// NVVMStMatrixOp build method - This is already generated, but let's ensure
// it's correct (This method already exists in the generated code)

// AMDGPUMfmaOp build method
void AMDGPUMfmaOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::ValueRange operands, mlir::TypeRange resultTypes,
                         mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 3u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    assert(resultTypes.size() == 1u && "mismatched number of return types");
    state.addTypes(resultTypes);
}

// NVVMLdMatrixOp build method
void NVVMLdMatrixOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state,
                           mlir::ValueRange operands,
                           mlir::TypeRange resultTypes,
                           mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 1u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    assert(resultTypes.size() == 1u && "mismatched number of return types");
    state.addTypes(resultTypes);
}

// NVVMStMatrixOp build method
void NVVMStMatrixOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state,
                           mlir::ValueRange operands,
                           mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 2u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    // No result types to add (void operation)
}

// AMDGPURawBufferLoadOp build method
void AMDGPURawBufferLoadOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state,
    mlir::ValueRange operands, mlir::TypeRange resultTypes,
    mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 4u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    assert(resultTypes.size() == 1u && "mismatched number of return types");
    state.addTypes(resultTypes);
}

// AMDGPURawBufferStoreOp build method
void AMDGPURawBufferStoreOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state,
    mlir::ValueRange operands,
    mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 5u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    // No result types to add (void operation)
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

// FullOp build method
void FullOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value shape, mlir::Value value,
                   mlir::Type resultType) {
    state.addOperands({shape, value});
    state.addTypes(resultType);
}

// FullOp verify method
mlir::LogicalResult FullOp::verify() {
    // Verify that the result type is a memref
    auto resultInfo = mlir::dyn_cast<MemRefType>(getResult().getType());
    if (!resultInfo) {
        return emitOpError("result must be an ave-lang memref type");
    }

    // Verify that the value type matches the memref element type
    if (auto result_element_type =
            mlir::dyn_cast<mlir::VectorType>(resultInfo.getElementType())) {
        auto valueType = getValue().getType();
        if (valueType != result_element_type &&
            valueType != result_element_type.getElementType()) {
            return emitOpError(
                "fill value type must match memref element type");
        }
    } else if (getValue().getType() != resultInfo.getElementType()) {
        return emitOpError("fill value type must match memref element type");
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// EndLifetimeOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult EndLifetimeOp::verify() {
    if (getValues().empty()) {
        return emitOpError("requires at least one operand");
    }
    return mlir::success();
}

void EndLifetimeOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
                         mlir::SideEffects::DefaultResource::get());
}

} // namespace causalflow::avelang::dialect
