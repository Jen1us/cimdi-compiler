// 基于实际PyTorch量化模式的测试
func.func @test_real_quantized_pattern(%activation: tensor<10x128xf32>) -> tensor<10x8xf32> {
  // 1. INT8量化权重和参数
  %quantized_weight = arith.constant dense<1> : tensor<8x128xi8>
  %scale = arith.constant dense<0.02> : tensor<8x1xf32>  
  %zero_point = arith.constant dense<0> : tensor<8x1xi8>
  
  // 2. 将量化权重转换为f32
  %weight_f32_temp = tensor.empty() : tensor<8x128xf32>
  %weight_f32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%quantized_weight : tensor<8x128xi8>) outs(%weight_f32_temp : tensor<8x128xf32>) {
  ^bb0(%in: i8, %out: f32):
    %cast = arith.sitofp %in : i8 to f32
    linalg.yield %cast : f32
  } -> tensor<8x128xf32>
  
  // 3. 将零点转换为f32
  %zp_f32_temp = tensor.empty() : tensor<8x1xf32>
  %zp_f32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%zero_point : tensor<8x1xi8>) outs(%zp_f32_temp : tensor<8x1xf32>) {
  ^bb0(%in: i8, %out: f32):
    %cast = arith.sitofp %in : i8 to f32
    linalg.yield %cast : f32
  } -> tensor<8x1xf32>
  
  // 4. 反量化：减去零点
  %dequant1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%weight_f32, %zp_f32 : tensor<8x128xf32>, tensor<8x1xf32>) outs(%weight_f32_temp : tensor<8x128xf32>) {
  ^bb0(%in: f32, %zp: f32, %out: f32):
    %sub = arith.subf %in, %zp : f32
    linalg.yield %sub : f32
  } -> tensor<8x128xf32>
  
  // 5. 反量化：乘以缩放因子
  %dequant_weight = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%dequant1, %scale : tensor<8x128xf32>, tensor<8x1xf32>) outs(%weight_f32_temp : tensor<8x128xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %mul = arith.mulf %in, %s : f32
    linalg.yield %mul : f32
  } -> tensor<8x128xf32>
  
  // 6. 转置权重用于矩阵乘法
  %weight_transposed_temp = tensor.empty() : tensor<128x8xf32>
  %weight_transposed = linalg.transpose ins(%dequant_weight : tensor<8x128xf32>) 
                                      outs(%weight_transposed_temp : tensor<128x8xf32>) 
                                      permutation = [1, 0]
  
  // 7. 初始化输出
  %init = arith.constant dense<0.0> : tensor<10x8xf32>
  
  // 8. 矩阵乘法
  %result = linalg.matmul ins(%activation, %weight_transposed : tensor<10x128xf32>, tensor<128x8xf32>) 
                         outs(%init : tensor<10x8xf32>) -> tensor<10x8xf32>
  
  return %result : tensor<10x8xf32>
}