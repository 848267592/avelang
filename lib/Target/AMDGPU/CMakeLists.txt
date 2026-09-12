add_library(ave-lang-target-amdgpu
  gpu_to_amdgpu_pipeline.cc
  legalize_gpu_shuffle_to_idx_pass.cc
  lower_math_to_amdgpu_pass.cc
  amdgpu_backend.cc
  rocm_installation_detector.cc
  amdgpu_toolchain.cc
)
ave_lang_disable_rtti_for_llvm(ave-lang-target-amdgpu)

target_link_libraries(ave-lang-target-amdgpu
  PRIVATE
  MLIRArithToLLVM
  MLIRArithDialect
  MLIRVectorToLLVMPass
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRIndexToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
  MLIRGPUTransforms
  MLIRBufferizationTransforms
  MLIRGPUDialect
  MLIRAMDGPUDialect
  MLIRAMDGPUToROCDL
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRSCFToControlFlow
  MLIRUBToLLVM
  MLIRROCDLDialect
  MLIRGPUToROCDLTransforms
  MLIRMemRefTransforms
  ave-lang-mlir
  ave-lang-dialect-transforms
  LLVMAMDGPUCodeGen
  LLVMAMDGPUInfo
  LLVMAMDGPUDesc
  LLVMAsmPrinter
  LLVMBitWriter
  LLVMCodeGen
  LLVMTarget
  LLVMCore
  LLVMSupport
  clangDriver
)

target_compile_definitions(ave-lang-target-amdgpu PRIVATE ${LLVM_DEFINITIONS})

# Add AMDGPU codegen test
add_executable(amdgpu_codegen_test amdgpu_codegen_test.cc)
ave_lang_disable_rtti_for_llvm(amdgpu_codegen_test)
target_link_libraries(amdgpu_codegen_test
  ave-lang-target-amdgpu
  ave-lang-driver
  ave-lang-basic
  ave-lang-ast
  hip::host
  GTest::gtest_main
)

add_test(amdgpu_codegen_test amdgpu_codegen_test)

# C14 is a synthetic static-physical codegen proof.  It intentionally does
# not launch a Qwen kernel or run a performance benchmark.
add_executable(c14_static_physical_codegen_test
  c14_static_physical_codegen_test.cc)
ave_lang_disable_rtti_for_llvm(c14_static_physical_codegen_test)
target_link_libraries(c14_static_physical_codegen_test
  ave-lang-target-amdgpu
  ave-lang-dialect-transforms
  ave-lang-driver
  ave-lang-basic
  ave-lang-ast
  GTest::gtest_main)
add_test(c14_static_physical_codegen_test c14_static_physical_codegen_test)

# C24 is a compiler-only proof that a target-owned pending packet can remain
# SSA-live across independent MFMA work and is committed to LDS only at its
# first true consumer.  It does not launch a Qwen kernel.
add_executable(c24_region_pending_packet_codegen_test
  c24_region_pending_packet_codegen_test.cc)
ave_lang_disable_rtti_for_llvm(c24_region_pending_packet_codegen_test)
target_link_libraries(c24_region_pending_packet_codegen_test
  ave-lang-target-amdgpu
  ave-lang-dialect-transforms
  ave-lang-driver
  ave-lang-basic
  ave-lang-ast
  GTest::gtest_main)
add_test(c24_region_pending_packet_codegen_test
  c24_region_pending_packet_codegen_test)

# C25 asks the target waitcnt analysis to make an older BF16x8 packet ready
# while a younger packet remains pending through current MFMA work.  It is a
# compiler-only feasibility proof and does not launch a Qwen kernel.
add_executable(c25_current_ready_pending_codegen_test
  c25_current_ready_pending_codegen_test.cc)
ave_lang_disable_rtti_for_llvm(c25_current_ready_pending_codegen_test)
target_link_libraries(c25_current_ready_pending_codegen_test
  ave-lang-target-amdgpu
  ave-lang-dialect-transforms
  ave-lang-driver
  ave-lang-basic
  ave-lang-ast
  GTest::gtest_main)
add_test(c25_current_ready_pending_codegen_test
  c25_current_ready_pending_codegen_test)
