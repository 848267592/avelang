add_library(ave-lang-driver
  driver.cc
)
ave_lang_disable_rtti_for_llvm(ave-lang-driver)

target_link_libraries(ave-lang-driver
  PRIVATE
  ave-lang-frontend
  ave-lang-mlir
  ave-lang-target-gpu
  ave-lang-basic
  ave-lang-ast
  LLVMCore
  LLVMSupport
)

target_compile_definitions(ave-lang-driver PRIVATE ${LLVM_DEFINITIONS})
