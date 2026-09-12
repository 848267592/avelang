//===- AveLangOps.td - AveLang dialect ops ----------------*- tablegen -*-===//
//
// This file defines the operations of the AveLang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AVE_LANG_OPS
#define AVE_LANG_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AveLang Dialect
//===----------------------------------------------------------------------===//

def AveLang_Dialect : Dialect {
    let name = "ave";
    let cppNamespace = "::causalflow::avelang::dialect";
    let summary = "AveLang dialect";
    let description = [{
        The AveLang dialect provides operations for avelang
        computations, including tuple operations and specialized kernels.
    }];

    let extraClassDeclaration = [{
        void registerAttributes();
        void registerTypes();

        ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                         ::mlir::Type type) const override;
        void printAttribute(::mlir::Attribute attr,
                            ::mlir::DialectAsmPrinter &printer) const override;
        ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
        void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
    }];
}

//===----------------------------------------------------------------------===//
// Base operation classes
//===----------------------------------------------------------------------===//

class AveLang_Op<string mnemonic, list<Trait> traits = []> :
    Op<AveLang_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AveLang Operations
//===----------------------------------------------------------------------===//

def ReturnOp : AveLang_Op<"return"> {
    let summary = "Return from an ave-lang function";
    let description = [{
        The `ave.return` operation represents an early return before
        control-flow normalization rewrites functions to a single-exit form.
    }];

    let arguments = (ins Variadic<AnyType>:$operands);
    let results = (outs);
}

def MakeIntTupleOp : AveLang_Op<"make_int_tuple", [Pure]> {
    let summary = "Create an integer tuple from elements";
    let description = [{
        The `ave.make_int_tuple` operation creates a tuple containing
        integer or index values, or nested tuples. This replaces the 
        ParseNestedTuple functionality from builtin_module.cc.
        
        Example:
        ```mlir
        %tuple = ave.make_int_tuple %0, %1, %2 : i32, i32, i32
        %nested = ave.make_int_tuple %tuple, %3 : none, i32
        ```
    }];
    
    let arguments = (ins Variadic<AnyType>:$elements);
    let results = (outs AnyType:$result);
    
    let assemblyFormat = [{
        $elements attr-dict `:` type($elements) `->` type($result)
    }];
    
    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$elements)>
    ];
    
    let hasVerifier = 1;
    
    let extraClassDeclaration = [{
        // Helper methods for tuple access
        size_t getNumElements() { return getElements().size(); }
        mlir::Value getElement(size_t index) { 
            assert(index < getNumElements());
            return getElements()[index]; 
        }
    }];
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

include "AveLangMemRefOps.td"

def MakeLayoutOp : AveLang_Op<"make_layout", [Pure]> {
    let summary = "Create a layout from dims and stride tuples";
    let description = [{
        The `ave.make_layout` operation creates a Layout value from
        dims and stride integer tuples. The Layout represents memory layout
        information that can be used in tensor operations.

        Example:
        ```mlir
        %dims = ave.make_int_tuple %c4, %c4 : i32, i32 -> none
        %stride = ave.make_int_tuple %c4, %c1 : i32, i32 -> none
        %layout = ave.make_layout %dims, %stride : none, none -> none
        ```
    }];

    let arguments = (ins
        AnyType:$dims,
        AnyType:$stride
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $dims `,` $stride attr-dict `:`
        functional-type(operands, results)
    }];

    let builders = [
        OpBuilder<(ins "mlir::Value":$dims, "mlir::Value":$stride)>
    ];

    let hasVerifier = 1;
}

def FullOp : AveLang_Op<"full", []> {
    let summary = "Create a memref filled with a constant value";
    let description = [{
        The `ave.full` operation creates an ave-lang memref filled with a constant scalar value.
        The memref is flattened to a 1D buffer, filled using vector broadcast and transfer write,
        then reshaped to the target shape.

        Example:
        ```mlir
        %result = ave.full %shape, %value : i32 ->
            !ave.memref<!ave.layout<dims = [4, 4], strides = []>, i32>
        ```
    }];

    let arguments = (ins
        AnyType:$shape,
        AnyType:$value
    );
    let results = (outs AveLang_MemRef:$result);

    let assemblyFormat = [{
        $shape `,` $value attr-dict `:`
        type($shape) `,` type($value) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::Value":$shape, "mlir::Value":$value,
                      "mlir::Type":$resultType)>
    ];

    let hasVerifier = 1;
}

def EndLifetimeOp : AveLang_Op<"end_lifetime", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Mark values as dead after this point";
    let description = [{
        The `ave.end_lifetime` operation is a compiler-facing marker used to
        express that its operands are no longer needed after this point.  It is
        intended as a narrow phase/lifetime boundary for kernels where lowering
        otherwise keeps producer-side temporaries live into a later phase.

        The operation has no numerical semantics and returns no values.  It is
        deliberately not marked Pure so early DCE/CSE do not silently erase it.
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs);

    let assemblyFormat = [{
        $values attr-dict `:` type($values)
    }];

    let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// GPU Operations
//===----------------------------------------------------------------------===//

// Base class for GPU operations
class AveLang_GPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<AveLang_Dialect, "gpu." # mnemonic, traits>;

//===----------------------------------------------------------------------===//
// NVVM Operations
//===----------------------------------------------------------------------===//

def NVVMMMAOp : AveLang_GPU_Op<"nvvm_mma", []> {
    let summary = "NVVM Matrix Multiply-Accumulate operation";
    let description = [{
        The `ave.gpu.nvvm_mma` operation performs matrix multiplication and accumulation
        using NVVM MMA intrinsics.
    }];

    let arguments = (ins
        LLVM_AnyVector:$a,
        LLVM_AnyVector:$b,
        LLVM_AnyVector:$c
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $a `,` $b `,` $c attr-dict `:`
        type($a) `,` type($b) `,` type($c) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

def NVVMLdMatrixOp : AveLang_GPU_Op<"nvvm_ldmatrix", []> {
    let summary = "NVVM Load Matrix operation";
    let description = [{
        The `ave.gpu.nvvm_ldmatrix` operation loads a matrix from shared memory
        using NVVM LdMatrix intrinsics.
    }];

    let arguments = (ins
        AveLang_MemRef:$memref,
        StrAttr:$matrix_shape,
        I32Attr:$matrix_num,
        I32Attr:$matrix_bit_width
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $memref attr-dict `:` type($memref) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

def NVVMStMatrixOp : AveLang_GPU_Op<"nvvm_stmatrix", []> {
    let summary = "NVVM Store Matrix operation";
    let description = [{
        The `ave.gpu.nvvm_stmatrix` operation stores a matrix to shared memory
        using NVVM StMatrix intrinsics.
    }];

    let arguments = (ins
        AnyType:$value,
        AveLang_MemRef:$memref,
        StrAttr:$matrix_shape,
        I32Attr:$matrix_num,
        I32Attr:$matrix_bit_width
    );
    let results = (outs);

    let assemblyFormat = [{
        $value `,` $memref attr-dict `:`
        type($value) `,` type($memref)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AMDGPU Operations
//===----------------------------------------------------------------------===//

def AMDGPUMfmaOp : AveLang_GPU_Op<"amdgpu_mfma", []> {
    let summary = "AMDGPU MFMA operation";
    let description = [{
        The `ave.gpu.amdgpu_mfma` operation performs matrix multiplication and accumulation
        using AMDGPU MFMA intrinsics. The operation is configured with matrix dimensions (m, n, k)
        and data types (type_a, type_c) to determine the appropriate intrinsic to use.
    }];

    let arguments = (ins
        LLVM_AnyVector:$a,
        LLVM_AnyVector:$b,
        LLVM_AnyVector:$c,
        I32Attr:$m,
        I32Attr:$n,
        I32Attr:$k,
        StrAttr:$type_a,
        StrAttr:$type_c
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $a `,` $b `,` $c attr-dict `:`
        type($a) `,` type($b) `,` type($c) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

def AMDGPURawBufferLoadOp : AveLang_GPU_Op<"amdgpu_raw_buffer_load", []> {
    let summary = "AMDGPU Raw Buffer Load operation";
    let description = [{
        The `ave.gpu.amdgpu_raw_buffer_load` operation loads data from a raw buffer
        using AMDGPU raw buffer intrinsics.
    }];

    let arguments = (ins
        AnyType:$rsrc,
        AnyType:$vindex,
        AnyType:$soffset,
        AnyType:$aux
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $rsrc `,` $vindex `,` $soffset `,` $aux attr-dict `:`
        type($rsrc) `,` type($vindex) `,` type($soffset) `,` type($aux) `->` type($result)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

def AMDGPURawBufferStoreOp : AveLang_GPU_Op<"amdgpu_raw_buffer_store", []> {
    let summary = "AMDGPU Raw Buffer Store operation";
    let description = [{
        The `ave.gpu.amdgpu_raw_buffer_store` operation stores data to a raw buffer
        using AMDGPU raw buffer intrinsics.
    }];

    let arguments = (ins
        AnyType:$data,
        AnyType:$rsrc,
        AnyType:$vindex,
        AnyType:$soffset,
        AnyType:$aux
    );
    let results = (outs);

    let assemblyFormat = [{
        $data `,` $rsrc `,` $vindex `,` $soffset `,` $aux attr-dict `:`
        type($data) `,` type($rsrc) `,` type($vindex) `,` type($soffset) `,` type($aux)
    }];

    let builders = [
        OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::ArrayRef<mlir::NamedAttribute>":$attributes)>
    ];

    let hasVerifier = 1;
}

def AMDGPUQwenUpdateKFragLoadOp : AveLang_GPU_Op<
    "amdgpu_qwen_update_kfrag_load", []> {
    let summary = "Persistent Qwen update MFMA B-fragment load";
    let description = [{
        This opt-in operation marks the fixed Qwen L6 producer-consumer
        pattern.  It remains in the IR through AveLang-to-memref conversion so
        a dedicated pass can replace broad [128,64] K staging with a compact
        [128,16] physical tile before LLVM register allocation.

        The first implementation is deliberately guarded to BF16 Qwen L6
        geometry and a vector<4xbf16> result consumed as the B operand of an
        mfma_16x16x16_bf16_f32 call.
    }];

    let arguments = (ins
        MemRefLike:$sharedK,
        MemRefLike:$sourceK,
        AnyType:$threadId,
        AnyType:$keyHead,
        AnyType:$tokenWindowBase,
        AnyType:$kColumn,
        AnyType:$tokenFragmentBase
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $sharedK `,` $sourceK `,` $threadId `,` $keyHead `,`
        $tokenWindowBase `,` $kColumn `,` $tokenFragmentBase attr-dict `:`
        type($sharedK) `,` type($sourceK) `,` type($threadId) `,`
        type($keyHead) `,` type($tokenWindowBase) `,` type($kColumn) `,`
        type($tokenFragmentBase) `->` type($result)
    }];

    let hasVerifier = 1;
}

def AMDGPUQwenUpdateKFragLDSLoadOp : AveLang_GPU_Op<
    "amdgpu_qwen_update_kfrag_lds_load", []> {
    let summary = "Late-lowered Qwen update MFMA B-fragment LDS load";
    let description = [{
        This is the post-rewrite form of the Qwen update K-fragment consumer.
        Unlike `amdgpu_qwen_update_kfrag_load`, it intentionally carries only
        the workgroup base address and the two fixed-layout coordinates needed
        for one packed vector<4xbf16> LDS read.  A late AMDGPU pass lowers it
        directly to an addrspace(3) 8-byte LLVM load, avoiding a generic
        vector.load/memref GEP chain in the update MFMA16 B operand path.
    }];

    let arguments = (ins
        AnyType:$sharedBase,
        AnyType:$kColumn,
        AnyType:$tokenFragmentLocalBase
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $sharedBase `,` $kColumn `,` $tokenFragmentLocalBase attr-dict `:`
        type($sharedBase) `,` type($kColumn) `,`
        type($tokenFragmentLocalBase) `->` type($result)
    }];

    let hasVerifier = 1;
}

def AMDGPUQwenPredStateKVFragLoadOp : AveLang_GPU_Op<
    "amdgpu_qwen_pred_state_kv_frag_load", []> {
    let summary = "Qwen pred MFMA B fragment from a KxV persistent-state tile";
    let description = [{
        This narrow first-class fragment operation is the producer-side
        mechanism probe for the State-KV pred M/N exchange.  It represents a
        BF16x4 B fragment read at fixed V from a workgroup [wave,K,V] state
        snapshot.  Its late lowering must materialize the four K-strided
        elements directly into vector SSA; it must not introduce a transient
        VxK shared view or a private gather buffer.
    }];

    let arguments = (ins
        MemRefLike:$stateKV,
        AnyType:$wave,
        AnyType:$kVector,
        AnyType:$vLane,
        AnyType:$fragmentOffset
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $stateKV `,` $wave `,` $kVector `,` $vLane `,` $fragmentOffset
        attr-dict `:` type($stateKV) `,` type($wave) `,` type($kVector) `,`
        type($vLane) `,` type($fragmentOffset) `->` type($result)
    }];
}

def AMDGPUBlockDotBF16F32Op : AveLang_GPU_Op<
    "amdgpu_block_dot_bf16_f32", []> {
    let summary = "BF16 block dot with FP32 accumulation";
    let description = [{
        This experimental operation keeps the direct-K64 recurrence-update
        block-dot semantic intact until after GPU outlining.  It represents
        the two 32x32 output tiles for one K[64,64] half and one wave-local
        V[32,32] block.  A=[2,32,32] denotes two independent V32 wave
        slices, while A=[1,32,32] denotes the experimental BV32 cooperative
        mode where both waves consume the same V32 staging tile and own
        separate K64 persistent-state halves.  The result is a packed vector<32xf32>: elements
        0..15 are the low K32 tile and elements 16..31 are the high K32 tile.

        The generic and gfx942-specialized lowerings deliberately share this
        same source op, operand ABI, shared allocations, CTA ownership and
        MFMA geometry.  They differ only in late scheduling/materialization
        of the typed B operand.

        In operand mode, the first two operands are logical workgroup BF16
        A/B blocks.  The `operand_role`, `transpose` and `layout_intent`
        attributes describe the block without embedding Qwen-specific
        pointers or phase indices.  In full-scope logical-block mode the
        third operand is the target-independent global logical B source,
        while the first operand may be an already-resident shared A block and
        the second operand is the shared destination selected by the planner.
        This preserves the operation ABI while allowing one lowering to own
        producer packet ownership, shared placement and the dot consumer.
        The remaining legacy operands stay in the operation for
        backward-compatible construction.
    }];

    let arguments = (ins
        MemRefLike:$aStage,
        MemRefLike:$bStage,
        MemRefLike:$sourceK,
        MemRefLike:$sourceVNew,
        MemRefLike:$sourceG,
        AnyType:$threadId,
        AnyType:$chunkStart,
        AnyType:$valueHead,
        AnyType:$keyHead,
        AnyType:$valueBase,
        AnyType:$kHalf,
        AnyType:$gLast,
        AnyType:$accLow,
        AnyType:$accHigh
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $aStage `,` $bStage `,` $sourceK `,` $sourceVNew `,` $sourceG `,`
        $threadId `,` $chunkStart `,` $valueHead `,` $keyHead `,`
        $valueBase `,` $kHalf `,` $gLast `,` $accLow `,` $accHigh attr-dict `:`
        type($aStage) `,` type($bStage) `,` type($sourceK) `,`
        type($sourceVNew) `,` type($sourceG) `,` type($threadId) `,`
        type($chunkStart) `,` type($valueHead) `,` type($keyHead) `,`
        type($valueBase) `,` type($kHalf) `,` type($gLast) `,` type($accLow) `,`
        type($accHigh) `->` type($result)
    }];

    let hasVerifier = 1;
}

// Internal target-aware representation used by block_dot lowering.  This is
// deliberately not a user-facing Qwen/chunk-o operation: it keeps the
// planner-selected shared rows and packed operand identity intact until the
// GPU-module late lowering materializes the MFMA32 consumer.
def AMDGPUBlockDotMfmaOperandOp : AveLang_GPU_Op<
    "amdgpu_block_dot_mfma_operand", []> {
    let summary = "Planned packed MFMA operand consumer for block_dot";
    let description = [{
        Carries one logical block-dot MFMA consumer after producer/shared
        planning.  The row and operand-word values are explicit SSA operands,
        while role, transpose, logical shape, physical encoding and target
        instruction shape remain generic attributes.  A later GPU-module
        lowering turns this into packed LDS reads and the existing MFMA
        intrinsic call.  It is an internal representation only and must not
        become a public source primitive.
    }];

    let arguments = (ins
        MemRefLike:$aStage,
        MemRefLike:$bStage,
        LLVM_AnyVector:$accumulator,
        AnyType:$aRow,
        AnyType:$bRow,
        AnyType:$operandWord,
        AnyType:$kStage
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $aStage `,` $bStage `,` $accumulator `,` $aRow `,` $bRow `,`
        $operandWord `,` $kStage attr-dict `:` type($aStage) `,`
        type($bStage) `,` type($accumulator) `,` type($aRow) `,`
        type($bRow) `,` type($operandWord) `,` type($kStage) `->`
        type($result)
    }];
}

def AMDGPUQwenGdnRecurrenceStepBF16F32Op : AveLang_GPU_Op<
    "amdgpu_qwen_gdn_recurrence_step_bf16_f32", []> {
    let summary = "Late-lowered Qwen BT64 Direct-K64 recurrence step";
    let description = [{
        Experimental-only semantic operation for one Qwen GDN BT64 recurrence
        step. The source owns the outer device-side chunk loop and persistent
        FP32 state. This operation owns the internal stream32 schedule:

          pred partial -> BF16 V-new -> BF16 V-decay -> Direct-K64 update.

        It remains intact through AveLang-to-memref lowering and is expanded by
        a dedicated late pass before GPU outlining. The operation deliberately
        exposes all workgroup buffers, ownership indices, ABI outputs and
        persistent state operands. It does not hide a whole-kernel schedule,
        select an LDS layout, or alter MFMA geometry.

        `phaseStage` is a single physical BF16 [64,64] workgroup allocation.
        The stream32 lowering uses it first as [2,32,64] pred-W storage and,
        after the correction barrier, as [64,32] update-K storage. `vdecay`
        contains only the current [V32,T32] phase. The result packs next
        state low/high vectors, each with sixteen FP32 MFMA accumulator lanes.
    }];

    let arguments = (ins
        MemRefLike:$stateStage,
        MemRefLike:$phaseStage,
        MemRefLike:$predPartial,
        MemRefLike:$vdecayStage,
        MemRefLike:$sourceK,
        MemRefLike:$sourceW,
        MemRefLike:$sourceU,
        MemRefLike:$sourceG,
        MemRefLike:$outputH,
        MemRefLike:$auditPredF32,
        MemRefLike:$auditPredBF16,
        MemRefLike:$outputVNew,
        MemRefLike:$auditVDecay,
        MemRefLike:$auditStateAfter,
        AnyType:$threadId,
        AnyType:$chunkIndex,
        AnyType:$valueHead,
        AnyType:$keyHead,
        AnyType:$valueBase,
        AnyType:$stateLow,
        AnyType:$stateHigh,
        AnyType:$emitAudit
    );
    let results = (outs LLVM_AnyVector:$result);

    let assemblyFormat = [{
        $stateStage `,` $phaseStage `,` $predPartial `,` $vdecayStage `,`
        $sourceK `,` $sourceW `,` $sourceU `,` $sourceG `,` $outputH `,`
        $auditPredF32 `,` $auditPredBF16 `,` $outputVNew `,` $auditVDecay `,`
        $auditStateAfter `,` $threadId `,` $chunkIndex `,` $valueHead `,`
        $keyHead `,` $valueBase `,` $stateLow `,` $stateHigh `,` $emitAudit
        attr-dict `:` type($stateStage) `,` type($phaseStage) `,`
        type($predPartial) `,` type($vdecayStage) `,` type($sourceK) `,`
        type($sourceW) `,` type($sourceU) `,` type($sourceG) `,`
        type($outputH) `,` type($auditPredF32) `,` type($auditPredBF16) `,`
        type($outputVNew) `,` type($auditVDecay) `,` type($auditStateAfter) `,`
        type($threadId) `,` type($chunkIndex) `,` type($valueHead) `,`
        type($keyHead) `,` type($valueBase) `,` type($stateLow) `,`
        type($stateHigh) `,` type($emitAudit) `->` type($result)
    }];

    let hasVerifier = 1;
}

def AMDGPUQwenPersistentRecurrenceYieldOp : AveLang_GPU_Op<
    "amdgpu_qwen_persistent_recurrence_yield", [Terminator]> {
    let summary = "Terminate a persistent recurrence semantic region";
    let arguments = (ins);
    let results = (outs);
}

def AMDGPUQwenPersistentRecurrenceOp : AveLang_GPU_Op<
    "amdgpu_qwen_persistent_recurrence",
    [SingleBlockImplicitTerminator<"AMDGPUQwenPersistentRecurrenceYieldOp">]> {
    let summary = "Qwen full device-side persistent recurrence semantic region";
    let description = [{
        Experimental-only semantic container for a complete Qwen GDN
        device-side recurrence.  Unlike
        `amdgpu_qwen_gdn_recurrence_step_bf16_f32`, this operation owns the
        outer BT64 chunk loop together with its loop-carried FP32 state,
        pre-update BF16 H snapshot, pred, BF16 V-new boundary, V-decay and
        update.  The body region is formed from the already validated B0
        source loop after AveLang-to-memref lowering.  This preserves the
        exact legacy B0 schedule while making the entire recurrence visible to
        a future target schedule planner.

        The operation deliberately does not encode a Qwen ISA sequence, a
        fixed LDS mapping, or an external code object.  Those are target
        schedule-layer decisions attached as an attribute by the formation
        pass.  `legacy_b0` lowering inlines the body region unchanged before
        block-dot lowering, so it is a semantic-preservation baseline rather
        than a new performance schedule.
    }];

    let arguments = (ins
        MemRefLike:$sourceK,
        MemRefLike:$sourceW,
        MemRefLike:$sourceU,
        MemRefLike:$sourceG,
        MemRefLike:$initialState,
        MemRefLike:$outputH,
        MemRefLike:$auditPredF32,
        MemRefLike:$auditPredBF16,
        MemRefLike:$outputVNew,
        MemRefLike:$auditVDecay,
        MemRefLike:$auditStateAfter,
        MemRefLike:$finalState,
        AnyType:$numChunks,
        AnyType:$emitAudit
    );
    let results = (outs);
    let regions = (region AnyRegion:$body);
    let hasVerifier = 1;
}

def AMDGPUQwenPersistentRecurrenceEndOp : AveLang_GPU_Op<
    "amdgpu_qwen_persistent_recurrence_end", []> {
    let summary = "Temporary frontend delimiter for persistent recurrence formation";
    let description = [{
        This experimental delimiter is consumed by the persistent recurrence
        formation pass.  It is never allowed to reach target lowering.
    }];
    let arguments = (ins);
    let results = (outs);
}

def AMDGPUQwenKFragStageLoadOp : AveLang_GPU_Op<
    "amdgpu_qwen_kfrag_stage_load", []> {
    let summary = "Late-expanded Qwen compact-K producer global load";
    let description = [{
        This guarded, internal operation represents one BF16 source-K load
        feeding the compact [128,64] workgroup staging tile. It deliberately
        retains source-K indexing as an opaque recipe until after GPU outlining
        so the final 64-bit global-address tuple is constructed next to this
        load and its immediately following LDS store.

        It is not a new mathematical operation and does not fuse a load with a
        store. The late pass expands it to the same scalar BF16 memref load as
        the baseline producer. The only experimental purpose is to constrain
        when the full global address is materialized.
    }];

    let arguments = (ins
        MemRefLike:$sourceK,
        AnyType:$sourceToken,
        AnyType:$keyHead,
        AnyType:$kColumn
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $sourceK `,` $sourceToken `,` $keyHead `,` $kColumn attr-dict `:`
        type($sourceK) `,` type($sourceToken) `,` type($keyHead) `,`
        type($kColumn) `->` type($result)
    }];

    let hasVerifier = 1;
}

def AMDGPUQwenK64PipelineStageLoadOp : AveLang_GPU_Op<
    "amdgpu_qwen_k64_pipeline_stage_load", []> {
    let summary = "Opaque gfx942 Qwen BT64 operand distributed pipeline-stage load";
    let description = [{
        This experimental-only operation issues no frontend-visible tensor
        value.  It records one BF16 [64,64] W or K global tile and its
        lane-owned packet recipe as an opaque compiler-owned stage token. The token may
        only be consumed by amdgpu_qwen_k64_pipeline_stage_commit.

        A late GPU-module pass selects either immediate lowering (load and
        commit together) or distributed lookahead lowering (BF16x8 loads at
        this op and LDS stores at commit).  Keeping the token opaque prevents
        generic vector/memref passes and AveLang source from materializing a
        W/K next-tile local array.
    }];

    let arguments = (ins
        MemRefLike:$sourceK,
        AnyType:$threadId,
        AnyType:$chunkStart,
        AnyType:$keyHead,
        AnyType:$kHalf
    );
    let results = (outs AnyType:$stageToken);

    let assemblyFormat = [{
        $sourceK `,` $threadId `,` $chunkStart `,` $keyHead `,` $kHalf
        attr-dict `:` type($sourceK) `,` type($threadId) `,`
        type($chunkStart) `,` type($keyHead) `,` type($kHalf) `->`
        type($stageToken)
    }];

    let hasVerifier = 1;
}

def AMDGPUQwenK64PipelineStageCommitOp : AveLang_GPU_Op<
    "amdgpu_qwen_k64_pipeline_stage_commit", []> {
    let summary = "Commit an opaque gfx942 Qwen BT64 operand pipeline stage to LDS";
    let description = [{
        Consumes exactly one amdgpu_qwen_k64_pipeline_stage_load token and
        commits its distributed BF16x8 packets into a BF16 [2,64,64]
        workgroup W/K bank. The corresponding current-bank consumer must have
        completed before this operation.  The late pass owns packet lifetime,
        vector load placement, waitcnt selection, and LDS stores.
    }];

    let arguments = (ins AnyType:$stageToken, MemRefLike:$sharedKBank);

    let assemblyFormat = [{
        $stageToken `,` $sharedKBank attr-dict `:` type($stageToken) `,`
        type($sharedKBank)
    }];

    let hasVerifier = 1;
}

// A core-last-use issue is deliberately a real BF16x8 SSA value, rather than
// the historical i64 stage token.  It is used when a next packet is issued
// from inside the (single) recurrence core and has to remain register-live
// until the existing tail commit boundary.  In particular it must never be
// materialized through a private packet ring merely to cross a compiler
// scheduling boundary.
def AMDGPUQwenK64CoreIssueOp : AveLang_GPU_Op<
    "amdgpu_qwen_k64_core_issue", []> {
    let summary = "Issue one Qwen BF16x8 next packet at a core last-use boundary";
    let description = [{
        Represents a single lane-owned BF16x8 global W/K packet issued after
        its current-core consumer group.  The result remains vector SSA until
        amdgpu_qwen_k64_core_commit at the pre-existing safe recurrence-tail
        LDS boundary.  packet_start and consumer-group metadata describe the
        planner-selected physical schedule; they do not encode lane addresses.
    }];

    let arguments = (ins
        MemRefLike:$sourceK,
        AnyType:$threadId,
        AnyType:$chunkStart,
        AnyType:$keyHead,
        AnyType:$kHalf
    );
    let results = (outs LLVM_AnyVector:$packet);

    let assemblyFormat = [{
        $sourceK `,` $threadId `,` $chunkStart `,` $keyHead `,` $kHalf
        attr-dict `:` type($sourceK) `,` type($threadId) `,`
        type($chunkStart) `,` type($keyHead) `,` type($kHalf) `->`
        type($packet)
    }];

    let hasVerifier = 1;
}

def AMDGPUQwenK64CoreCommitOp : AveLang_GPU_Op<
    "amdgpu_qwen_k64_core_commit", []> {
    let summary = "Commit a register-live Qwen BF16x8 packet at recurrence tail";
    let description = [{
        Commits one vector result from amdgpu_qwen_k64_core_issue into the
        existing R4 W/K LDS bank.  This op is intentionally placed only after
        the already-validated cross-wave tail barrier; it neither allocates a
        second LDS bank nor inserts a per-group barrier.
    }];

    let arguments = (ins
        LLVM_AnyVector:$packet,
        AnyType:$threadId,
        AnyType:$kHalf,
        MemRefLike:$sharedKBank
    );

    let assemblyFormat = [{
        $packet `,` $threadId `,` $kHalf `,` $sharedKBank attr-dict `:`
        type($packet) `,` type($threadId) `,` type($kHalf) `,`
        type($sharedKBank)
    }];

    let hasVerifier = 1;
}

// A pending packet is a compiler-internal SSA edge used by target pipeline
// materialization.  Unlike the historical opaque stage token, its result is
// the actual vector payload.  The region planner owns the issue/commit pair;
// no frontend API may create it and no lowering is allowed to materialize a
// private packet ring merely to cross a logical consumer boundary.
def AMDGPURegionPendingPacketIssueOp : AveLang_GPU_Op<
    "amdgpu_region_pending_packet_issue", []> {
    let summary = "Issue a target-owned BF16 packet for a later region commit";
    let description = [{
        Internal-only GPU pipeline representation.  `source` and `indices`
        describe one typed global packet; `predicate` determines whether the
        lane owns the packet.  The vector result remains SSA-live until the
        unique amdgpu_region_pending_packet_commit.  The operation deliberately
        has no Qwen frontend spelling and carries no tile-specific layout.
    }];

    let arguments = (ins
        MemRefLike:$source,
        AnyType:$predicate,
        Variadic<AnyType>:$indices
    );
    let results = (outs LLVM_AnyVector:$packet);

    let hasVerifier = 1;
}

def AMDGPURegionPendingPacketCommitOp : AveLang_GPU_Op<
    "amdgpu_region_pending_packet_commit", []> {
    let summary = "Commit a target-owned pending packet into shared memory";
    let description = [{
        Internal-only counterpart to amdgpu_region_pending_packet_issue.  It
        consumes exactly one same-block pending packet, waits only when the
        packet becomes an LDS store dependency, and preserves the existing
        region barrier after publication.  `destination` and `indices` are
        intentionally generic so the representation is not tied to a Qwen
        shape or a source-facing intrinsic.
    }];

    let arguments = (ins
        LLVM_AnyVector:$packet,
        AnyType:$predicate,
        MemRefLike:$destination,
        Variadic<AnyType>:$indices
    );

    let hasVerifier = 1;
}

#endif // AVE_LANG_OPS
