// test-comprehensive-tiling.mlir - 综合测试SA Tiling和PE Tiling功能
func.func @test_comprehensive_tiling(%input_large: tensor<512x256xf32>, %input_pe: tensor<256x384xf32>) -> (tensor<512x384xf32>, tensor<256x256xf32>, tensor<64x64xf32>) {
  
  // ========================================
  // 场景1: SA Tiling - 大矩阵量化计算 (应该触发SA tiling)
  // 512x256 x 256x384 -> 512x384 
  // ========================================
  
  // SA量化权重和参数（常量）- 大尺寸权重矩阵 
  %sa_quantized_weight = arith.constant dense<4> : tensor<384x256xi8>
  %sa_scale = arith.constant dense<0.025> : tensor<384x1xf32>  
  %sa_zero_point = arith.constant dense<3> : tensor<384x1xi8>
  
  // SA反量化链条：sitofp → subf → mulf → transpose → matmul
  %sa_weight_f32_temp = tensor.empty() : tensor<384x256xf32>
  %sa_weight_f32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_quantized_weight : tensor<384x256xi8>) outs(%sa_weight_f32_temp : tensor<384x256xf32>) {
  ^bb0(%in: i8, %out: f32):
    %sitofp_result = arith.sitofp %in : i8 to f32
    linalg.yield %sitofp_result : f32
  } -> tensor<384x256xf32>
  
  %sa_zp_f32_temp = tensor.empty() : tensor<384x1xf32>
  %sa_zp_f32 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_zero_point : tensor<384x1xi8>) outs(%sa_zp_f32_temp : tensor<384x1xf32>) {
  ^bb0(%in: i8, %out: f32):
    %sitofp_zp = arith.sitofp %in : i8 to f32
    linalg.yield %sitofp_zp : f32
  } -> tensor<384x1xf32>
  
  %sa_dequant1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_weight_f32, %sa_zp_f32 : tensor<384x256xf32>, tensor<384x1xf32>) outs(%sa_weight_f32_temp : tensor<384x256xf32>) {
  ^bb0(%in: f32, %zp: f32, %out: f32):
    %subf_result = arith.subf %in, %zp : f32
    linalg.yield %subf_result : f32
  } -> tensor<384x256xf32>
  
  %sa_dequant_weight = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%sa_dequant1, %sa_scale : tensor<384x256xf32>, tensor<384x1xf32>) outs(%sa_weight_f32_temp : tensor<384x256xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %mulf_result = arith.mulf %in, %s : f32
    linalg.yield %mulf_result : f32
  } -> tensor<384x256xf32>
  
  // transpose操作：将权重从 384x256 转置为 256x384
  %sa_weight_transposed_temp = tensor.empty() : tensor<256x384xf32>
  %sa_weight_transposed = linalg.transpose ins(%sa_dequant_weight : tensor<384x256xf32>) 
                                         outs(%sa_weight_transposed_temp : tensor<256x384xf32>) 
                                         permutation = [1, 0]
  
  // SA大矩阵乘法：512x256 x 256x384 -> 512x384 (应该触发SA tiling)
  %sa_init = arith.constant dense<0.0> : tensor<512x384xf32>
  %sa_result = linalg.matmul ins(%input_large, %sa_weight_transposed : tensor<512x256xf32>, tensor<256x384xf32>) 
                            outs(%sa_init : tensor<512x384xf32>) -> tensor<512x384xf32>
  
  // ========================================  
  // 场景2: PE Tiling - 大矩阵普通计算 (应该触发PE tiling)
  // 256x384 x 384x256 -> 256x256
  // ========================================
  
  // 动态计算的权重矩阵（非常量）384x256 - 这会触发PE pattern
  %pe_weight_base = tensor.empty() : tensor<384x256xf32>
  %pe_weight = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%input_pe : tensor<256x384xf32>) outs(%pe_weight_base : tensor<384x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    // 动态计算权重（保证非常量）
    %processed = arith.mulf %in, %in : f32  // 动态计算权重
    linalg.yield %processed : f32
  } -> tensor<384x256xf32>
  
  %pe_init = arith.constant dense<1.0> : tensor<256x256xf32>  
  %pe_result = linalg.matmul ins(%input_pe, %pe_weight : tensor<256x384xf32>, tensor<384x256xf32>) 
                            outs(%pe_init : tensor<256x256xf32>) -> tensor<256x256xf32>
  
  // ========================================
  // 场景3: 小矩阵测试 - 应该直接转换不做tiling  
  // 64x64 x 64x64 -> 64x64 (小于128，不应该tiling)
  // ========================================
  
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  
  // 创建小矩阵进行测试
  %small_input = arith.constant dense<2.0> : tensor<64x64xf32>
  %small_weight_base = tensor.empty() : tensor<64x64xf32> 
  %small_weight = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], 
    iterator_types = ["parallel", "parallel"]
  } ins(%small_input : tensor<64x64xf32>) outs(%small_weight_base : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %processed = arith.mulf %in, %in : f32  // 动态计算权重
    linalg.yield %processed : f32
  } -> tensor<64x64xf32>
  
  %small_init = arith.constant dense<0.0> : tensor<64x64xf32>
  %small_result = linalg.matmul ins(%small_input, %small_weight : tensor<64x64xf32>, tensor<64x64xf32>) 
                                outs(%small_init : tensor<64x64xf32>) -> tensor<64x64xf32>
  
  return %sa_result, %pe_result, %small_result : tensor<512x384xf32>, tensor<256x256xf32>, tensor<64x64xf32>
}