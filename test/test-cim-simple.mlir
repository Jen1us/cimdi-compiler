// 简化的量化测试文件
func.func @test_quantized_matmul(%activation: tensor<128x256xf16>) -> tensor<128x512xf16> {
  // 1. 常量权重
  %weight_fp16 = arith.constant dense<1.0> : tensor<256x512xf16>
  
  // 2. 量化激活 (FP16 -> INT8)  
  %act_quant = quant.qcast %activation : tensor<128x256xf16> to tensor<128x256x!quant.uniform<i8:f16, 0.05:0>>
  
  // 3. 量化权重 (FP16 -> INT4)
  %weight_quant = quant.qcast %weight_fp16 : tensor<256x512xf16> to tensor<256x512x!quant.uniform<i4:f16, 0.02:0>>
  
  // 4. 初始化输出
  %init = arith.constant dense<0> : tensor<128x512xi32>
  
  // 5. 量化矩阵乘法
  %result_quant = linalg.matmul 
    ins(%act_quant, %weight_quant : tensor<128x256x!quant.uniform<i8:f16, 0.05:0>>, tensor<256x512x!quant.uniform<i4:f16, 0.02:0>>) 
    outs(%init : tensor<128x512xi32>) -> tensor<128x512xi32>
  
  // 6. 反量化结果
  %result_fp16 = quant.dcast %result_quant : tensor<128x512xi32> to tensor<128x512xf16>
  
  return %result_fp16 : tensor<128x512xf16>
}