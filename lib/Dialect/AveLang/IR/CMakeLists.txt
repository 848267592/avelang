# Generate MLIR dialect files from TableGen
set(LLVM_TARGET_DEFINITIONS AveLangOps.td)
mlir_tablegen(AveLangOps.h.inc -gen-op-decls)
mlir_tablegen(AveLangOps.cpp.inc -gen-op-defs)
mlir_tablegen(AveLangDialect.h.inc -gen-dialect-decls -dialect=ave)
mlir_tablegen(AveLangDialect.cpp.inc -gen-dialect-defs -dialect=ave)

# Generate type definitions from TableGen
set(LLVM_TARGET_DEFINITIONS AveLangTypes.td)
mlir_tablegen(AveLangTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AveLangTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(AveLangOpsIncGen)
add_public_tablegen_target(AveLangTypesIncGen)

add_library(ave-lang-dialect
    AveLangDialect.cc
    AveLangOps.cc
    AveLangTypes.cc
)
ave_lang_disable_rtti_for_llvm(ave-lang-dialect)

target_include_directories(ave-lang-dialect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_dependencies(ave-lang-dialect AveLangOpsIncGen AveLangTypesIncGen)

target_link_libraries(ave-lang-dialect
    MLIRIR
    MLIRLLVMDialect
    MLIRPtrDialect
    MLIRSupport
)
