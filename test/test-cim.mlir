// test-cim.mlir
#quant_type_w = !quant.uniform<i4:f16, 0.02:0>
#quant_type_a = !quant.uniform<i8:f16, 0.05:0>
#quant_type_r = !quant.uniform<i32:f16, 0.001:0>

func.func @main(%activation_fp16: tensor<128x256xf16>) -> tensor<128x512xf16> {
  // 一个浮点的常量权重
  %weight_fp16 = arith.constant dense<1.0> : tensor<256x512xf16>

  // --- 这就是我们想要匹配的特征模式 ---
  // 1. 将浮点 activation 量化成整数
  %act_quant = "quant.qcast"(%activation_fp16) : (tensor<128x256xf16>) -> tensor<128x256x#quant_type_a>

  // 2. 将浮点常量 weight 量化成整数
  %weight_quant = "quant.qcast"(%weight_fp16) : (tensor<256x512xf16>) -> tensor<256x512x#quant_type_w>

  // 3. 在整数上执行 matmul
  %init_acc = arith.constant dense<0> : tensor<128x512xi32>
  %result_quant = linalg.matmul ins(%act_quant, %weight_quant : tensor<128x256x#quant_type_a>, tensor<256x512x#quant_type_w>) outs(%init_acc : tensor<128x512xi32>) -> tensor<128x512xi32>

  // 4. 将整数结果反量化回浮点
  %result_fp16 = "quant.dcast"(%result_quant) : (tensor<128x512xi32>) -> tensor<128x512xf16>
  // --- 模式结束 ---

  return %result_fp16 : tensor<128x512xf16>
}