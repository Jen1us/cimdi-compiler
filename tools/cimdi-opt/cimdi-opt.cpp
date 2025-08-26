#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// 引入我们自己 Cimdi 方言的头文件
#include "mlir/Dialect/Cimdi/CimdiDialect.h"

// 引入常用的标准方言
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/IR/Quant.h"


// **[新添加]** 声明我们将要调用的Pass注册函数
// 这个函数的实体在 LinalgToCimdi.cpp 中
void registerLinalgToCimdiPass();

int main(int argc, char **argv) {
  // **[新添加]** 注册我们的自定义Pass
  registerLinalgToCimdiPass();

  // 创建一个方言注册表
  mlir::DialectRegistry registry;

  // 注册常用的MLIR标准方言
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::quant::QuantDialect>();

  // **最关键的一步: 将我们自定义的Cimdi方言注册进去**
  registry.insert<mlir::cimdi::CimdiDialect>();

  // 运行 MLIR 的优化器主程序，传入我们的注册表
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CIMDI Optimizer", registry));
}