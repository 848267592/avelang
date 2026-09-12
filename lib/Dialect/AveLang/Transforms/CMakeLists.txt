add_library(ave-lang-dialect-transforms
  allocation_op_interface_impl.cc
  bounded_packet_schedule_pass.cc
  hoist_alloca_pass.cc
  normalize_ave_lang_return_pass.cc
  qwen_kfrag_producer_consumer_rewrite_pass.cc
  lower_ave_lang_to_memref_pass.cc
  lower_gpuop_to_intrinsics_pass.cc
  lower_qwen_block_dot_pass.cc
  lower_qwen_gdn_recurrence_step_pass.cc
  qwen_modulo_software_pipeline_pass.cc
  qwen_persistent_recurrence_pass.cc
  lower_qwen_kfrag_lds_pass.cc
  lower_qwen_k64_pipeline_stage_pass.cc
)
if(COMMAND ave_lang_disable_rtti_for_llvm)
  ave_lang_disable_rtti_for_llvm(ave-lang-dialect-transforms)
endif()

target_include_directories(ave-lang-dialect-transforms PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/lib
    ${CMAKE_SOURCE_DIR}/lib/Dialect/AveLang/IR
    ${CMAKE_BINARY_DIR}/lib/Dialect/AveLang/IR
)

target_link_libraries(ave-lang-dialect-transforms
  PUBLIC
    ave-lang-dialect
    ave-lang-mlir
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTransforms
    MLIRFuncTransforms
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRNVVMDialect
    MLIRROCDLDialect
    MLIRGPUDialect
    MLIRMemRefDialect
    MLIRMathDialect
    MLIRPtrDialect
    MLIRBufferizationDialect
)
