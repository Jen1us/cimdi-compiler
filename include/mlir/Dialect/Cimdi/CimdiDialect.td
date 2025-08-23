#ifndef CIMDI_DIALECT_TD
#define CIMDI_DIALECT_TD

include "mlir/IR/DialectBase.td"

// 定义 'cimdi' 方言本身
def Cimdi_Dialect : Dialect {
  // 方言在文本IR中的名字 (e.g., cimdi.op_name)
  let name = "cimdi";
  
  // 方言在C++代码中的命名空间
  let cppNamespace = "::mlir::cimdi";
  
  // 方言的简短描述
  let summary = "A dialect for the CIM-Digital hybrid accelerator.";

  // 如果方言有自定义类型，在这里声明
  // let hasCustomTypes = 1;
}

#endif // CIMDI_DIALECT_TD