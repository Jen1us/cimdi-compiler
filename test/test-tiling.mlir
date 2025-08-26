  // test-tiling.mlir
  func.func @main(%A: tensor<256x384xf16>, %B: tensor<384x512xf16>, %C: tensor<256x512xf16>) -> tensor<256x512xf16> {
    %result = linalg.matmul ins(%A, %B : tensor<256x384xf16>, tensor<384x512xf16>) outs(%C : tensor<256x512xf16>) -> tensor<256x512xf16>
    return %result : tensor<256x512xf16>
  }