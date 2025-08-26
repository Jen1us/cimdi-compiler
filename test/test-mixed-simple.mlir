// 测试PE和SA两种pattern的混合场景
func.func @test_mixed_matmul_patterns(%input: tensor<4x8xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  
  // ========================================
  // 场景1: SA量化模式 - 应该匹配SA pattern
  // ========================================
  
  // 1. SA量化权重和参数（常量）
  %sa_quantized_weight = arith.constant dense<2> : tensor<4x8xi8>
  %sa_scale = arith.constant dense<0.1> : tensor<4x1xf32>  
  %sa_zero_point = arith.constant dense<1> : tensor<4x1xi8>
  
  // 2. SA反量化链条：sitofp → subf → mulf → transpose → matmul
  %sa_weight_f32_temp = tensor.empty() : tensor<4x8xf32>
  %sa_weight_f32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_quantized_weight : tensor<4x8xi8>) outs(%sa_weight_f32_temp : tensor<4x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %sitofp_result = arith.sitofp %in : i8 to f32
    linalg.yield %sitofp_result : f32
  } -> tensor<4x8xf32>
  
  %sa_zp_f32_temp = tensor.empty() : tensor<4x1xf32>
  %sa_zp_f32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_zero_point : tensor<4x1xi8>) outs(%sa_zp_f32_temp : tensor<4x1xf32>) {
  ^bb0(%in: i8, %out: f32):
    %sitofp_zp = arith.sitofp %in : i8 to f32
    linalg.yield %sitofp_zp : f32
  } -> tensor<4x1xf32>
  
  %sa_dequant1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_weight_f32, %sa_zp_f32 : tensor<4x8xf32>, tensor<4x1xf32>) outs(%sa_weight_f32_temp : tensor<4x8xf32>) {
  ^bb0(%in: f32, %zp: f32, %out: f32):
    %subf_result = arith.subf %in, %zp : f32
    linalg.yield %subf_result : f32
  } -> tensor<4x8xf32>
  
  %sa_dequant_weight = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_dequant1, %sa_scale : tensor<4x8xf32>, tensor<4x1xf32>) outs(%sa_weight_f32_temp : tensor<4x8xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %mulf_result = arith.mulf %in, %s : f32
    linalg.yield %mulf_result : f32
  } -> tensor<4x8xf32>
  
  %sa_weight_transposed_temp = tensor.empty() : tensor<8x4xf32>
  %sa_weight_transposed = linalg.transpose ins(%sa_dequant_weight : tensor<4x8xf32>) 
                                         outs(%sa_weight_transposed_temp : tensor<8x4xf32>) 
                                         permutation = [1, 0]
  
  %sa_init = arith.constant dense<0.0> : tensor<4x4xf32>
  %sa_result = linalg.matmul ins(%input, %sa_weight_transposed : tensor<4x8xf32>, tensor<8x4xf32>) 
                            outs(%sa_init : tensor<4x4xf32>) -> tensor<4x4xf32>
  
  // ========================================  
  // 场景2: PE普通模式 - 应该匹配PE pattern  
  // ========================================
  
  // 普通的矩阵乘法，权重是动态计算的（非常量），应该走PE路径
  %pe_weight = arith.constant dense<1.5> : tensor<8x4xf32>  // 简单常量，但没有量化链条
  %pe_init = arith.constant dense<0.0> : tensor<4x4xf32>
  %pe_result = linalg.matmul ins(%input, %pe_weight : tensor<4x8xf32>, tensor<8x4xf32>) 
                            outs(%pe_init : tensor<4x4xf32>) -> tensor<4x4xf32>
  
  return %sa_result, %pe_result : tensor<4x4xf32>, tensor<4x4xf32>
}