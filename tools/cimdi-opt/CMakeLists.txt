# 使用MLIR的帮助函数来创建一个工具可执行文件
add_mlir_tool(cimdi-opt
  cimdi-opt.cpp
)

# 将我们的工具链接到我们的方言库和必要的MLIR库
target_link_libraries(cimdi-opt PRIVATE
  MLIRCimdi
  MLIROptLib
  
  # Core MLIR dialects
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRQuantDialect
  
  # Support libraries
  MLIRSupport
  LLVMSupport
  
  # Pass libraries  
  MLIRPass
  MLIRTransforms

  # **[新添加]** 链接我们的Pass库
  MLIRLinalgToCimdi
)

# 添加include目录来找到生成的.inc文件
target_include_directories(cimdi-opt PRIVATE
  ${CMAKE_BINARY_DIR}/lib/Dialect/Cimdi
)