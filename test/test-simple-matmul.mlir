func.func @simple_matmul(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %init = arith.constant dense<0.0> : tensor<4x4xf32>
  %result = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}