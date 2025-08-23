// test.mlir
func.func @main(%A: tensor<128x128xf16>, %B: tensor<128x128xf16>, %C: tensor<128x128xf16>) -> tensor<128x128xf16> {
  // 一个简单的 linalg.matmul，尺寸与我们的PE硬件匹配
  %result = linalg.matmul ins(%A, %B : tensor<128x128xf16>, tensor<128x128xf16>) outs(%C : tensor<128x128xf16>) -> tensor<128x128xf16>
  return %result : tensor<128x128xf16>
}