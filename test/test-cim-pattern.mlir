// 符合我们pattern的量化测试: dcast(matmul(qcast(a), qcast(w)))
func.func @test_quantized_pattern(%activation: tensor<128x256xf16>) -> tensor<128x512xf16> {
  // 1. 常量权重
  %weight_fp16 = arith.constant dense<1.0> : tensor<256x512xf16>
  
  // 2. 量化激活和权重
  %act_quant = quant.qcast %activation : tensor<128x256xf16> to tensor<128x256x!quant.uniform<i8:f16, 0.05:0>>
  %weight_quant = quant.qcast %weight_fp16 : tensor<256x512xf16> to tensor<256x512x!quant.uniform<i4:f16, 0.02:0>>
  
  // 3. 初始化输出矩阵 - 应该匹配matmul操作的输出类型
  %init = arith.constant dense<0.0> : tensor<128x512xf16>
  
  // 4. 在量化值上执行矩阵乘法（实际上linalg期望FP输出）
  %result_matmul = linalg.matmul 
    ins(%act_quant, %weight_quant : tensor<128x256x!quant.uniform<i8:f16, 0.05:0>>, tensor<256x512x!quant.uniform<i4:f16, 0.02:0>>) 
    outs(%init : tensor<128x512xf16>) -> tensor<128x512xf16>
  
  // 5. 反量化 - 这是我们要匹配的最终操作
  %result_dquant = quant.dcast %result_matmul : tensor<128x512xf16> to tensor<128x512xf16>
  
  return %result_dquant : tensor<128x512xf16>
}