// 测试PE和SA两种pattern的混合场景
func.func @test_mixed_matmul_patterns(%input: tensor<10x128xf32>) -> tensor<10x16xf32> {
  
  // ========================================
  // 场景1: SA量化模式 - 应该匹配SA pattern
  // ========================================
  
  // 1. SA量化权重和参数（常量）
  %sa_quantized_weight = arith.constant dense<2> : tensor<8x128xi8>
  %sa_scale = arith.constant dense<0.05> : tensor<8x1xf32>  
  %sa_zero_point = arith.constant dense<1> : tensor<8x1xi8>
  
  // 2. SA反量化链条：sitofp → subf → mulf → transpose → matmul
  %sa_weight_f32_temp = tensor.empty() : tensor<8x128xf32>
  %sa_weight_f32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_quantized_weight : tensor<8x128xi8>) outs(%sa_weight_f32_temp : tensor<8x128xf32>) {
  ^bb0(%in: i8, %out: f32):
    %sitofp_result = arith.sitofp %in : i8 to f32
    linalg.yield %sitofp_result : f32
  } -> tensor<8x128xf32>
  
  %sa_zp_f32_temp = tensor.empty() : tensor<8x1xf32>
  %sa_zp_f32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_zero_point : tensor<8x1xi8>) outs(%sa_zp_f32_temp : tensor<8x1xf32>) {
  ^bb0(%in: i8, %out: f32):
    %sitofp_zp = arith.sitofp %in : i8 to f32
    linalg.yield %sitofp_zp : f32
  } -> tensor<8x1xf32>
  
  %sa_dequant1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_weight_f32, %sa_zp_f32 : tensor<8x128xf32>, tensor<8x1xf32>) outs(%sa_weight_f32_temp : tensor<8x128xf32>) {
  ^bb0(%in: f32, %zp: f32, %out: f32):
    %subf_result = arith.subf %in, %zp : f32
    linalg.yield %subf_result : f32
  } -> tensor<8x128xf32>
  
  %sa_dequant_weight = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_dequant1, %sa_scale : tensor<8x128xf32>, tensor<8x1xf32>) outs(%sa_weight_f32_temp : tensor<8x128xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %mulf_result = arith.mulf %in, %s : f32
    linalg.yield %mulf_result : f32
  } -> tensor<8x128xf32>
  
  %sa_weight_transposed_temp = tensor.empty() : tensor<128x8xf32>
  %sa_weight_transposed = linalg.transpose ins(%sa_dequant_weight : tensor<8x128xf32>) 
                                         outs(%sa_weight_transposed_temp : tensor<128x8xf32>) 
                                         permutation = [1, 0]
  
  %sa_init = arith.constant dense<0.0> : tensor<10x8xf32>
  %sa_result = linalg.matmul ins(%input, %sa_weight_transposed : tensor<10x128xf32>, tensor<128x8xf32>) 
                            outs(%sa_init : tensor<10x8xf32>) -> tensor<10x8xf32>
  
  // ========================================  
  // 场景2: PE普通模式 - 应该匹配PE pattern
  // ========================================
  
  // 创建一个动态计算的权重矩阵（非常量），应该走PE路径
  %pe_dynamic_input = tensor.empty() : tensor<128x8xf32>
  %pe_weight = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<10x128xf32>) outs(%pe_dynamic_input : tensor<128x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %processed = arith.mulf %in, %in : f32  // 动态计算权重
    linalg.yield %processed : f32
  } -> tensor<128x8xf32>
  
  %pe_init = arith.constant dense<1.0> : tensor<10x8xf32>  
  %pe_result = linalg.matmul ins(%input, %pe_weight : tensor<10x128xf32>, tensor<128x8xf32>) 
                            outs(%pe_init : tensor<10x8xf32>) -> tensor<10x8xf32>
  
  // ========================================
  // 组合结果 
  // ========================================
  
  %combined_temp = tensor.empty() : tensor<10x16xf32>
  %combined = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1 + 8)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_result, %pe_result : tensor<10x8xf32>, tensor<10x8xf32>) outs(%combined_temp : tensor<10x16xf32>) {
  ^bb0(%sa_in: f32, %pe_in: f32, %out: f32):
    %sum = arith.addf %sa_in, %pe_in : f32
    linalg.yield %sum : f32
  } -> tensor<10x16xf32>
  
  return %combined : tensor<10x16xf32>
}