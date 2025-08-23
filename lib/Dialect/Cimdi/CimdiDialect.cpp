#include "mlir/Dialect/Cimdi/CimdiDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::cimdi;

// 这行代码会引入由 CimdiDialect.td 生成的方言类实现
#define GET_DIALECT_CLASSES
#include "CimdiDialect.cpp.inc"

// 先引入操作的头文件声明
#define GET_OP_CLASSES
#include "CimdiOps.h.inc"

// 然后引入操作的实现
#define GET_OP_CLASSES
#include "CimdiOps.cpp.inc"

// 方言的初始化函数
void CimdiDialect::initialize() {
    // 在这里，我们将所有自动生成的Op添加到方言中，
    // 让MLIR的上下文(Context)能够认识它们。
    addOperations<
#define GET_OP_LIST
#include "CimdiOps.cpp.inc"
    >();
    
    // TODO: 在这里注册我们的自定义类型，比如I4
}