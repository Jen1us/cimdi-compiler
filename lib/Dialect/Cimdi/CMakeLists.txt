set(LLVM_TARGET_DEFINITIONS ../../../include/mlir/Dialect/Cimdi/CimdiDialect.td)
mlir_tablegen(CimdiDialect.h.inc -gen-dialect-decls -I${MLIR_MAIN_SRC_DIR}/include)
mlir_tablegen(CimdiDialect.cpp.inc -gen-dialect-defs -I${MLIR_MAIN_SRC_DIR}/include)
add_public_tablegen_target(MLIRCimdiIncGen)

set(LLVM_TARGET_DEFINITIONS ../../../include/mlir/Dialect/Cimdi/CimdiOps.td)
mlir_tablegen(CimdiOps.h.inc -gen-op-decls -I${MLIR_MAIN_SRC_DIR}/include)
mlir_tablegen(CimdiOps.cpp.inc -gen-op-defs -I${MLIR_MAIN_SRC_DIR}/include)
add_public_tablegen_target(MLIRCimdiOpsIncGen)

add_mlir_dialect_library(MLIRCimdi
  CimdiDialect.cpp

  DEPENDS
  MLIRCimdiIncGen
  MLIRCimdiOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

target_include_directories(obj.MLIRCimdi PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(MLIRCimdi PRIVATE ${CMAKE_CURRENT_BINARY_DIR})