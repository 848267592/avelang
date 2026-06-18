add_library(ave-lang-basic
  diagnostic.cc
)
ave_lang_disable_rtti_for_llvm(ave-lang-basic)

llvm_map_components_to_libnames(llvm_libs support)
target_link_libraries(ave-lang-basic 
  ${llvm_libs} 
  clangBasic
)
