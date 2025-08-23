#ifndef CIMDI_OPS_TD
#define CIMDI_OPS_TD

include "mlir/Dialect/Cimdi/CimdiDialect.td" // 引入我们刚创建的方言定义
include "mlir/IR/OpBase.td"

// ============================================================================
// 定义基础Operation类
// ============================================================================

// Cimdi方言中所有操作的基类
class Cimdi_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cimdi_Dialect, mnemonic, traits>;

// ============================================================================
// 我们将在这里添加所有 cimdi 操作的定义
// ============================================================================

//===----------------------------------------------------------------------===//
// CIM SA Matmul Operation
//===----------------------------------------------------------------------===//
def CimdiCIMSAMatmulOp : Cimdi_Op<"cim_sa_matmul"> {

  let summary = "CIM SA matrix multiplication for FP16 activation and INT4 weight";

  let description = [{
    Represents the entire custom computation flow for FP16 x INT4 mixed-precision
    matrix multiplication on the CIM chiplet. This includes pre-processing, 
    analog MAC, digital correction, and final dequantization.
  }];

  let arguments = (ins
    AnyRankedTensor:$activation,
    AnyRankedTensor:$weight,
    F32:$scale,
    F32:$zero_point
  );

  let results = (outs
    AnyRankedTensor:$result
  );
}

//===----------------------------------------------------------------------===//
// PE Matmul Operation
//===----------------------------------------------------------------------===//
def CimdiPEMatmulOp : Cimdi_Op<"pe_matmul"> {

  let summary = "PE array matrix multiplication for FP16 tensors";

  let description = [{
    Represents a single matrix multiplication executed on a PE array of the
    digital chiplet. This operation is the target for tiling large,
    floating-point `linalg.matmul` operations.
    
    It performs the computation `result = A * B + C_initial`.
  }];

  // 定义输入参数
  let arguments = (ins
    AnyRankedTensor:$A,
    AnyRankedTensor:$B,
    AnyRankedTensor:$C_initial // 用于累加的初始C矩阵
  );

  // 定义输出结果
  let results = (outs
    AnyRankedTensor:$result
  );
}


#endif // CIMDI_OPS_TD