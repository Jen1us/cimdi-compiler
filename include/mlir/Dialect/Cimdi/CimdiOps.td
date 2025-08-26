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

  let summary = "CIM SA matrix multiplication with FP16 activation and INT4/INT8 quantized weight";

  let description = [{
    Represents SA (Systolic Array) native quantized matrix multiplication on CIM chiplet.
    This operation takes:
    - FP16 activation input
    - Original INT4/INT8 quantized weight (bypassing software dequantization)
    - Scale factor tensor for hardware dequantization
    - Zero point tensor for hardware dequantization
    
    The hardware performs:
    1. FP16 activation decomposition (sign, exponent, mantissa)
    2. Native INT4 weight processing in RRAM cells
    3. Analog MAC computation in CIM
    4. Digital correction and hardware dequantization
    5. Direct FP16 output generation
    
    This replaces the PyTorch software dequantization chain:
    sitofp(weight) -> subf(zero_point) -> mulf(scale) -> matmul
    
    With SA hardware native computation:
    sa_matmul(fp16_activation, int4_weight, scale, zero_point) -> fp16_result
  }];

  let arguments = (ins
    AnyRankedTensor:$activation,         // FP16 激活
    RankedTensorOf<[AnyInteger]>:$weight,  // INT4/INT8 量化权重
    AnyRankedTensor:$scale,              // FP32 缩放因子
    RankedTensorOf<[AnyInteger]>:$zero_point  // INT8 原始零点（不反量化）
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

//===----------------------------------------------------------------------===//
// CIM SA Matmul Operation
//===----------------------------------------------------------------------===//
def CIM_SA_MatmulOp : Cimdi_Op<"cim.sa.matmul", []> {

  let summary = "CIM SA custom mixed-precision matrix multiplication.";

  let description = [{
    Represents the entire custom computation flow for FP16 activation x INT4 weight
    mixed-precision matrix multiplication on the CIM chiplet.
  }];

  // 定义操作的输入
  let arguments = (ins
    RankedTensorOf<[F16]>:$activation, // FP16 激活
    RankedTensorOf<[AnyInteger]>:$weight,       // INT4 权重
    F32:$scale,                     // 用于反量化的 scale
    F32:$zero_point                 // 用于反量化的 zero point
  );

  // 定义操作的输出
  let results = (outs
    RankedTensorOf<[F16]>:$result
  );
}


#endif // CIMDI_OPS_TD