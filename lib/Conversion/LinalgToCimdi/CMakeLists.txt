add_mlir_conversion_library(MLIRLinalgToCimdi
  LinalgToCimdi.cpp

  # 依赖我们的Cimdi方言库和Linalg方言
  DEPENDS
  MLIRCimdi
  MLIRLinalgDialect

  LINK_LIBS PUBLIC
  MLIRCimdi
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  MLIRLinalgTransforms # **[新]**
  MLIRSCFDialect      # **[修复]** 正确的库名
  MLIRAffineDialect   # **[修复]** 正确的库名
  MLIRArithDialect    # **[修复]** 正确的库名
  MLIRFuncDialect     # **[修复]** 正确的库名

)

# 添加include目录来找到生成的.inc文件
target_include_directories(obj.MLIRLinalgToCimdi PRIVATE 
  ${CMAKE_BINARY_DIR}/lib/Dialect/Cimdi
)
target_include_directories(MLIRLinalgToCimdi PRIVATE 
  ${CMAKE_BINARY_DIR}/lib/Dialect/Cimdi
)