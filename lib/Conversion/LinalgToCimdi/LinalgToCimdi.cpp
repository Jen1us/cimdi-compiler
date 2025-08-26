#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Cimdi/CimdiDialect.h"

// 引入TableGen自动生成的操作头文件
#define GET_OP_CLASSES
#include "CimdiOps.h.inc"

namespace {
// ============================================================================
//  SA量化Pattern: 匹配PyTorch反量化链条并转换为SA原生量化计算
// ============================================================================
class QuantizedMatmulConversionPattern
    : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
public:
  using OpRewritePattern<mlir::linalg::MatmulOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::MatmulOp matmulOp,
                  mlir::PatternRewriter &rewriter) const override {
    
    
    // 检查matmul的权重输入（第二个参数）是否来自反量化链条
    mlir::Value weightInput = matmulOp.getInputs()[1];
    
    // **Step 1: 检查是否是transpose后的权重**
    mlir::Value preTransposeWeight = weightInput;
    if (auto transposeOp = weightInput.getDefiningOp<mlir::linalg::TransposeOp>()) {
      preTransposeWeight = transposeOp.getInput();
    } else {
      return mlir::failure();
    }
    
    // **Step 2: 检查是否是反量化链条的最后一步 (mulf操作)**
    auto mulfOp = preTransposeWeight.getDefiningOp<mlir::linalg::GenericOp>();
    if (!mulfOp || !isElementwiseMultiply(mulfOp)) {
      return mlir::failure();
    }
    
    // **Step 3: 获取scale参数和subf操作**
    mlir::Value scaleValue = getScaleFromMulf(mulfOp);
    auto subfOp = getSubfFromMulf(mulfOp);
    if (!scaleValue || !subfOp) {
      return mlir::failure();
    }
    
    // **Step 4: 获取zero_point参数和sitofp操作**
    mlir::Value zeroPointValue = getOriginalZeroPointFromSubf(subfOp);  // 获取原始INT8零点
    auto sitofpOp = getSitofpFromSubf(subfOp);
    if (!zeroPointValue || !sitofpOp) {
      return mlir::failure();
    }
    
    // **Step 5: 获取原始量化权重并验证来源是常量**
    mlir::Value quantizedWeight = getQuantizedWeightFromSitofp(sitofpOp);
    if (!quantizedWeight || !isDirectConstant(quantizedWeight)) {
      return mlir::failure();
    }
    
    // **Step 6: 验证权重类型是i4/i8**
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(quantizedWeight.getType());
    if (!weightType) 
      return mlir::failure();
    
    auto elementType = weightType.getElementType();
    if (!elementType.isInteger(8) && !elementType.isInteger(4))
      return mlir::failure();
    
    // 成功匹配！检查是否需要tiling
    mlir::Value activation = matmulOp.getInputs()[0];
    
    // 获取矩阵维度以决定是否需要tiling
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(matmulOp.getResult(0).getType());
    if (!outputType || outputType.getShape().size() != 2) {
      return mlir::failure();
    }

    auto shape = outputType.getShape();
    int64_t M = shape[0], N = shape[1];
    
    // 获取 K 维度
    auto inputAType = mlir::dyn_cast<mlir::RankedTensorType>(activation.getType());
    if (!inputAType) return mlir::failure();
    int64_t K = inputAType.getShape()[1];

    // 如果矩阵足够小，直接转换
    if (M <= 128 && N <= 128 && K <= 128) {
      rewriter.replaceOpWithNewOp<mlir::cimdi::CimdiCIMSAMatmulOp>(
          matmulOp,
          matmulOp.getResult(0).getType(),
          activation,       // FP16激活 
          quantizedWeight,  // 原始量化权重
          scaleValue,       // 缩放因子
          zeroPointValue    // 零点
      );
      return mlir::success();
    }

    // 对于大矩阵，进行SA tiling
    return performSATiling(matmulOp, rewriter, activation, quantizedWeight, 
                          scaleValue, zeroPointValue, M, N, K);
  }

private:
  // SA tiling实现：处理大矩阵的SA量化计算
  mlir::LogicalResult performSATiling(mlir::linalg::MatmulOp op,
                                     mlir::PatternRewriter &rewriter,
                                     mlir::Value activation,
                                     mlir::Value quantizedWeight,
                                     mlir::Value scaleValue,
                                     mlir::Value zeroPointValue,
                                     int64_t M, int64_t N, int64_t K) const {
    constexpr int64_t TILE_SIZE = 128;
    
    mlir::Location loc = op.getLoc();
    mlir::Value C = op.getOutputs()[0];
    
    // 创建常量索引
    auto createConstIndex = [&](int64_t value) -> mlir::Value {
      return rewriter.create<mlir::arith::ConstantIndexOp>(loc, value);
    };
    
    mlir::Value c0 = createConstIndex(0);
    mlir::Value c1 = createConstIndex(1);
    mlir::Value tileSize = createConstIndex(TILE_SIZE);
    mlir::Value stepM = createConstIndex(std::min(TILE_SIZE, M));
    mlir::Value stepN = createConstIndex(std::min(TILE_SIZE, N));
    mlir::Value stepK = createConstIndex(std::min(TILE_SIZE, K));
    mlir::Value upperM = createConstIndex(M);
    mlir::Value upperN = createConstIndex(N);
    mlir::Value upperK = createConstIndex(K);

    // 创建三层嵌套循环：M x N x K
    auto outerLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, upperM, stepM, mlir::ValueRange{C});
    
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    mlir::Value i = outerLoop.getInductionVar();
    mlir::Value currentC = outerLoop.getBody()->getArgument(1);
    
    auto middleLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, upperN, stepN, mlir::ValueRange{currentC});
    
    rewriter.setInsertionPointToStart(middleLoop.getBody());
    mlir::Value j = middleLoop.getInductionVar();
    currentC = middleLoop.getBody()->getArgument(1);
    
    // 对于K维度的累加，需要处理初始化
    mlir::Value initC = currentC;
    
    auto innerLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, upperK, stepK, mlir::ValueRange{initC});
    
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    mlir::Value k = innerLoop.getInductionVar();
    mlir::Value accC = innerLoop.getBody()->getArgument(1);
    
    // 计算当前分块的实际大小
    mlir::Value tileSizeM = rewriter.create<mlir::arith::MinSIOp>(
        loc, tileSize, rewriter.create<mlir::arith::SubIOp>(loc, upperM, i));
    mlir::Value tileSizeN = rewriter.create<mlir::arith::MinSIOp>(
        loc, tileSize, rewriter.create<mlir::arith::SubIOp>(loc, upperN, j));
    mlir::Value tileSizeK = rewriter.create<mlir::arith::MinSIOp>(
        loc, tileSize, rewriter.create<mlir::arith::SubIOp>(loc, upperK, k));
    
    // 提取子矩阵activation (A) 和 quantizedWeight (B)
    llvm::SmallVector<mlir::OpFoldResult> offsetsA = {i, k};
    llvm::SmallVector<mlir::OpFoldResult> sizesA = {tileSizeM, tileSizeK};
    llvm::SmallVector<mlir::OpFoldResult> stridesA = {c1, c1};
    
    llvm::SmallVector<mlir::OpFoldResult> offsetsB = {k, j};
    llvm::SmallVector<mlir::OpFoldResult> sizesB = {tileSizeK, tileSizeN};
    llvm::SmallVector<mlir::OpFoldResult> stridesB = {c1, c1};
    
    mlir::Value subActivation = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, activation, offsetsA, sizesA, stridesA);
    mlir::Value subQuantizedWeight = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, quantizedWeight, offsetsB, sizesB, stridesB);
    
    // 提取当前的C分块
    llvm::SmallVector<mlir::OpFoldResult> offsetsC = {i, j};
    llvm::SmallVector<mlir::OpFoldResult> sizesC = {tileSizeM, tileSizeN};
    llvm::SmallVector<mlir::OpFoldResult> stridesC = {c1, c1};
    
    mlir::Value subC = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, accC, offsetsC, sizesC, stridesC);
    
    // 提取对应的scale和zero_point分块（如果它们不是标量）
    mlir::Value subScale = extractParameterSlice(rewriter, loc, scaleValue, j, tileSizeN);
    mlir::Value subZeroPoint = extractParameterSlice(rewriter, loc, zeroPointValue, j, tileSizeN);
    
    // 矩阵乘法累加逻辑：
    // - K=0时: C = A*B (使用原始C作为输入，但实际上应该重置为零)
    // - K>0时: C += A*B (使用累加的C值)
    auto isFirstKTile = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, k, c0);
    
    // 创建零张量用于第一次K迭代的初始化
    // 使用linalg.fill创建动态大小的零张量
    auto elementType = mlir::dyn_cast<mlir::RankedTensorType>(subC.getType()).getElementType();
    auto zeroValue = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    auto zeroTensor = rewriter.create<mlir::linalg::FillOp>(
        loc, mlir::ValueRange{zeroValue}, mlir::ValueRange{subC});
    
    // 第一次K迭代使用零初始化，后续使用累加值
    auto inputC = rewriter.create<mlir::arith::SelectOp>(
        loc, isFirstKTile, zeroTensor.getResult(0), subC);
    
    // 创建 CIMDi SA matmul 操作
    auto saMatmul = rewriter.create<mlir::cimdi::CimdiCIMSAMatmulOp>(
        loc, subC.getType(), subActivation, subQuantizedWeight, subScale, subZeroPoint);
    
    // 对于SA，我们需要将量化计算结果与之前的累加值相加（当K>0时）
    mlir::Value finalResult;
    if (K > TILE_SIZE) { // 只有当K需要分块时才需要累加
      auto addResult = rewriter.create<mlir::arith::SelectOp>(
          loc, isFirstKTile, saMatmul.getResult(),
          rewriter.create<mlir::arith::AddFOp>(loc, inputC, saMatmul.getResult()));
      finalResult = addResult;
    } else {
      finalResult = saMatmul.getResult();
    }
    
    // 将结果插回累加矩阵
    mlir::Value updatedAccC = rewriter.create<mlir::tensor::InsertSliceOp>(
        loc, finalResult, accC, offsetsC, sizesC, stridesC);
    
    rewriter.create<mlir::scf::YieldOp>(loc, updatedAccC);
    
    // 设置中间循环的yield
    rewriter.setInsertionPointAfter(innerLoop);
    rewriter.create<mlir::scf::YieldOp>(loc, innerLoop.getResult(0));
    
    // 设置外层循环的yield
    rewriter.setInsertionPointAfter(middleLoop);
    rewriter.create<mlir::scf::YieldOp>(loc, middleLoop.getResult(0));
    
    // 替换原操作
    rewriter.replaceOp(op, outerLoop.getResult(0));
    
    return mlir::success();
  }
  
  // 辅助函数：提取scale或zero_point的对应分块
  mlir::Value extractParameterSlice(mlir::PatternRewriter &rewriter, 
                                   mlir::Location loc,
                                   mlir::Value param,
                                   mlir::Value offset,
                                   mlir::Value size) const {
    auto paramType = mlir::dyn_cast<mlir::RankedTensorType>(param.getType());
    if (!paramType) {
      // 如果是标量，直接返回
      return param;
    }
    
    auto shape = paramType.getShape();
    if (shape.size() == 0) {
      // 标量张量，直接返回
      return param;
    } else if (shape.size() == 1) {
      // 1D张量，按列提取
      mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      llvm::SmallVector<mlir::OpFoldResult> offsets = {offset};
      llvm::SmallVector<mlir::OpFoldResult> sizes = {size};
      llvm::SmallVector<mlir::OpFoldResult> strides = {c1};
      
      return rewriter.create<mlir::tensor::ExtractSliceOp>(
          loc, param, offsets, sizes, strides);
    } else if (shape.size() == 2) {
      // 2D张量，按列提取 (假设是 [rows, cols] 格式)
      mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      mlir::Value fullRows = rewriter.create<mlir::arith::ConstantIndexOp>(loc, shape[0]);
      
      llvm::SmallVector<mlir::OpFoldResult> offsets = {c0, offset};
      llvm::SmallVector<mlir::OpFoldResult> sizes = {fullRows, size};
      llvm::SmallVector<mlir::OpFoldResult> strides = {c1, c1};
      
      return rewriter.create<mlir::tensor::ExtractSliceOp>(
          loc, param, offsets, sizes, strides);
    }
    
    // 其他情况直接返回原参数
    return param;
  }

  // 直接检查是否为常量操作（arith.constant）
  bool isDirectConstant(mlir::Value value) const {
    return value.getDefiningOp<mlir::arith::ConstantOp>() != nullptr;
  }
  
  // **关键函数：检查权重是否最终源自常量**
  bool isWeightFromConstant(mlir::Value weight) const {
    // 递归追踪权重的数据流，寻找arith.constant源头
    llvm::DenseSet<mlir::Value> visited; // 避免循环引用
    return traceToConstant(weight, visited);
  }
  
  bool traceToConstant(mlir::Value value, llvm::DenseSet<mlir::Value>& visited) const {
    // 避免无限递归
    if (visited.count(value))
      return false;
    visited.insert(value);
    
    // 检查是否直接是常量
    if (value.getDefiningOp<mlir::arith::ConstantOp>())
      return true;
    
    // 检查是否是函数参数中的常量（模型权重通常作为参数传入）
    if (mlir::isa<mlir::BlockArgument>(value)) {
      // 函数参数可能是预加载的权重，我们认为是"常量"
      return true;
    }
    
    auto defOp = value.getDefiningOp();
    if (!defOp)
      return false;
    
    // 对于linalg.generic操作，检查其输入是否来自常量
    if (auto genericOp = mlir::dyn_cast<mlir::linalg::GenericOp>(defOp)) {
      // 检查所有输入操作数
      for (auto input : genericOp.getInputs()) {
        if (traceToConstant(input, visited)) {
          return true; // 只要有一个输入来自常量即可
        }
      }
    }
    
    // 对于transpose等操作，检查其输入
    if (auto transposeOp = mlir::dyn_cast<mlir::linalg::TransposeOp>(defOp)) {
      return traceToConstant(transposeOp.getInput(), visited);
    }
    
    return false;
  }

  // 辅助函数：检查是否为逐元素乘法操作
  bool isElementwiseMultiply(mlir::linalg::GenericOp op) const {
    auto body = op.getBody();
    if (body->getOperations().size() != 2) {
      return false;
    }
    
    auto& firstOp = body->front();
    if (!mlir::isa<mlir::arith::MulFOp>(firstOp)) return false;
    
    auto& lastOp = body->back();
    return mlir::isa<mlir::linalg::YieldOp>(lastOp);
  }
  
  // 辅助函数：从乘法操作获取scale
  mlir::Value getScaleFromMulf(mlir::linalg::GenericOp mulfOp) const {
    for (auto input : mulfOp.getInputs()) {
      if (auto constOp = input.getDefiningOp<mlir::arith::ConstantOp>()) {
        return input;
      }
    }
    return nullptr;
  }
  
  // 辅助函数：从乘法操作获取减法操作
  mlir::linalg::GenericOp getSubfFromMulf(mlir::linalg::GenericOp mulfOp) const {
    for (auto input : mulfOp.getInputs()) {
      if (auto subfOp = input.getDefiningOp<mlir::linalg::GenericOp>()) {
        if (isElementwiseSubtract(subfOp)) {
          return subfOp;
        }
      }
    }
    return nullptr;
  }
  
  // 辅助函数：检查是否为逐元素减法操作
  bool isElementwiseSubtract(mlir::linalg::GenericOp op) const {
    auto body = op.getBody();
    if (body->getOperations().size() != 2) {
      return false;
    }
    
    auto& firstOp = body->front();
    if (!mlir::isa<mlir::arith::SubFOp>(firstOp)) return false;
    
    auto& lastOp = body->back();
    return mlir::isa<mlir::linalg::YieldOp>(lastOp);
  }
  
  // 辅助函数：从减法操作获取原始INT8零点（跳过sitofp转换）
  mlir::Value getOriginalZeroPointFromSubf(mlir::linalg::GenericOp subfOp) const {
    // 零点是减法的第二个操作数，我们需要获取sitofp之前的原始整数
    if (subfOp.getInputs().size() >= 2) {
      mlir::Value zeroPointF32 = subfOp.getInputs()[1];
      
      // 检查这个f32零点是否来自sitofp转换
      if (auto sitofpGeneric = zeroPointF32.getDefiningOp<mlir::linalg::GenericOp>()) {
        if (isSitofpOp(sitofpGeneric) && sitofpGeneric.getInputs().size() >= 1) {
          // 返回sitofp之前的原始整数零点
          return sitofpGeneric.getInputs()[0];
        }
      }
      
      // 如果不是从sitofp来的，直接返回（可能已经是整数类型）
      return zeroPointF32;
    }
    return nullptr;
  }
  
  // 辅助函数：从减法操作获取sitofp操作
  mlir::linalg::GenericOp getSitofpFromSubf(mlir::linalg::GenericOp subfOp) const {
    if (subfOp.getInputs().size() >= 1) {
      if (auto sitofpOp = subfOp.getInputs()[0].getDefiningOp<mlir::linalg::GenericOp>()) {
        if (isSitofpOp(sitofpOp)) {
          return sitofpOp;
        }
      }
    }
    return nullptr;
  }
  
  // 辅助函数：检查是否为sitofp操作
  bool isSitofpOp(mlir::linalg::GenericOp op) const {
    
    auto body = op.getBody();
    
    if (body->getOperations().size() != 2) { // 应该有一个sitofp和一个yield
      return false;
    }
    
    // 检查第一个操作是否是sitofp
    auto& firstOp = body->front();
    bool isSitofp = mlir::isa<mlir::arith::SIToFPOp>(firstOp);
    
    if (!isSitofp) return false;
    
    // 检查第二个操作是否是yield
    auto& lastOp = body->back();
    bool isYield = mlir::isa<mlir::linalg::YieldOp>(lastOp);
    
    return isYield;
  }
  
  // 辅助函数：从sitofp操作获取原始量化权重
  mlir::Value getQuantizedWeightFromSitofp(mlir::linalg::GenericOp sitofpOp) const {
    if (sitofpOp.getInputs().size() >= 1) {
      return sitofpOp.getInputs()[0];
    }
    return nullptr;
  }
};

// 负责将 linalg.matmul 转换为 tiled cimdi.pe.matmul 操作
class MatmulOpConversion
    : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
public:
  explicit MatmulOpConversion(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::linalg::MatmulOp>(context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::MatmulOp op,
                  mlir::PatternRewriter &rewriter) const override {


    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getOutputs()[0].getType());
    if (!outputType || outputType.getShape().size() != 2) {
      return mlir::failure();
    }

    auto shape = outputType.getShape();
    int64_t M = shape[0], N = shape[1];
    
    // 获取 K 维度
    auto inputAType = mlir::dyn_cast<mlir::RankedTensorType>(op.getInputs()[0].getType());
    if (!inputAType) return mlir::failure();
    int64_t K = inputAType.getShape()[1];

    // 如果矩阵足够小，直接转换
    if (M <= 128 && N <= 128 && K <= 128) {
      rewriter.replaceOpWithNewOp<mlir::cimdi::CimdiPEMatmulOp>(
          op, op.getResult(0).getType(), op.getInputs()[0],
          op.getInputs()[1], op.getOutputs()[0]);
      return mlir::success();
    }

    // 对于大矩阵，进行 tiling
    return performTiling(op, rewriter, M, N, K);
  }

private:
  mlir::LogicalResult performTiling(mlir::linalg::MatmulOp op,
                                   mlir::PatternRewriter &rewriter,
                                   int64_t M, int64_t N, int64_t K) const {
    constexpr int64_t TILE_SIZE = 128;
    
    mlir::Location loc = op.getLoc();
    mlir::Value A = op.getInputs()[0];
    mlir::Value B = op.getInputs()[1];
    mlir::Value C = op.getOutputs()[0];
    
    // 创建常量索引
    auto createConstIndex = [&](int64_t value) -> mlir::Value {
      return rewriter.create<mlir::arith::ConstantIndexOp>(loc, value);
    };
    
    mlir::Value c0 = createConstIndex(0);
    mlir::Value c1 = createConstIndex(1);
    mlir::Value tileSize = createConstIndex(TILE_SIZE);
    mlir::Value stepM = createConstIndex(std::min(TILE_SIZE, M));
    mlir::Value stepN = createConstIndex(std::min(TILE_SIZE, N));
    mlir::Value stepK = createConstIndex(std::min(TILE_SIZE, K));
    mlir::Value upperM = createConstIndex(M);
    mlir::Value upperN = createConstIndex(N);
    mlir::Value upperK = createConstIndex(K);

    // 创建三层嵌套循环：M x N x K
    auto outerLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, upperM, stepM, mlir::ValueRange{C});
    
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    mlir::Value i = outerLoop.getInductionVar();
    mlir::Value currentC = outerLoop.getBody()->getArgument(1);
    
    auto middleLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, upperN, stepN, mlir::ValueRange{currentC});
    
    rewriter.setInsertionPointToStart(middleLoop.getBody());
    mlir::Value j = middleLoop.getInductionVar();
    currentC = middleLoop.getBody()->getArgument(1);
    
    // 对于K维度的累加，需要处理初始化
    mlir::Value initC = currentC;
    
    auto innerLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, upperK, stepK, mlir::ValueRange{initC});
    
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    mlir::Value k = innerLoop.getInductionVar();
    mlir::Value accC = innerLoop.getBody()->getArgument(1);
    
    // 计算当前分块的实际大小
    mlir::Value tileSizeM = rewriter.create<mlir::arith::MinSIOp>(
        loc, tileSize, rewriter.create<mlir::arith::SubIOp>(loc, upperM, i));
    mlir::Value tileSizeN = rewriter.create<mlir::arith::MinSIOp>(
        loc, tileSize, rewriter.create<mlir::arith::SubIOp>(loc, upperN, j));
    mlir::Value tileSizeK = rewriter.create<mlir::arith::MinSIOp>(
        loc, tileSize, rewriter.create<mlir::arith::SubIOp>(loc, upperK, k));
    
    // 提取子矩阵A和B
    llvm::SmallVector<mlir::OpFoldResult> offsetsA = {i, k};
    llvm::SmallVector<mlir::OpFoldResult> sizesA = {tileSizeM, tileSizeK};
    llvm::SmallVector<mlir::OpFoldResult> stridesA = {c1, c1};
    
    llvm::SmallVector<mlir::OpFoldResult> offsetsB = {k, j};
    llvm::SmallVector<mlir::OpFoldResult> sizesB = {tileSizeK, tileSizeN};
    llvm::SmallVector<mlir::OpFoldResult> stridesB = {c1, c1};
    
    mlir::Value subA = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, A, offsetsA, sizesA, stridesA);
    mlir::Value subB = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, B, offsetsB, sizesB, stridesB);
    
    // 提取当前的C分块
    llvm::SmallVector<mlir::OpFoldResult> offsetsC = {i, j};
    llvm::SmallVector<mlir::OpFoldResult> sizesC = {tileSizeM, tileSizeN};
    llvm::SmallVector<mlir::OpFoldResult> stridesC = {c1, c1};
    
    mlir::Value subC = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, accC, offsetsC, sizesC, stridesC);
    
    // 矩阵乘法累加逻辑：
    // - K=0时: C = A*B (使用原始C作为输入，但实际上应该重置为零)
    // - K>0时: C += A*B (使用累加的C值)
    auto isFirstKTile = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, k, c0);
    
    // 创建零张量用于第一次K迭代的初始化
    // 使用linalg.fill创建动态大小的零张量
    auto elementType = mlir::dyn_cast<mlir::RankedTensorType>(subC.getType()).getElementType();
    auto zeroValue = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    auto zeroTensor = rewriter.create<mlir::linalg::FillOp>(
        loc, mlir::ValueRange{zeroValue}, mlir::ValueRange{subC});
    
    // 第一次K迭代使用零初始化，后续使用累加值
    auto inputC = rewriter.create<mlir::arith::SelectOp>(
        loc, isFirstKTile, zeroTensor.getResult(0), subC);
    
    // 创建 CIMDi PE matmul 操作
    auto peMatmul = rewriter.create<mlir::cimdi::CimdiPEMatmulOp>(
        loc, subC.getType(), subA, subB, inputC);
    
    // 将结果插回累加矩阵
    mlir::Value updatedAccC = rewriter.create<mlir::tensor::InsertSliceOp>(
        loc, peMatmul.getResult(), accC, offsetsC, sizesC, stridesC);
    
    rewriter.create<mlir::scf::YieldOp>(loc, updatedAccC);
    
    // 设置中间循环的yield
    rewriter.setInsertionPointAfter(innerLoop);
    rewriter.create<mlir::scf::YieldOp>(loc, innerLoop.getResult(0));
    
    // 设置外层循环的yield
    rewriter.setInsertionPointAfter(middleLoop);
    rewriter.create<mlir::scf::YieldOp>(loc, middleLoop.getResult(0));
    
    // 替换原操作
    rewriter.replaceOp(op, outerLoop.getResult(0));
    
    return mlir::success();
  }
};

// 主Pass，驱动Tiling和Lowering
struct LinalgToCimdiPass
    : public mlir::PassWrapper<LinalgToCimdiPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToCimdiPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::cimdi::CimdiDialect, mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                    mlir::quant::QuantDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    // 添加Lowering Pattern - 注意顺序很重要！
    // SA pattern必须在PE pattern之前，否则所有matmul都会被PE pattern匹配走
    patterns.add<QuantizedMatmulConversionPattern>(context);  // 先匹配SA量化计算
    patterns.add<MatmulOpConversion>(context);                // 再匹配PE普通计算
    // 使用Greedy Driver应用所有Pattern
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }

  mlir::StringRef getArgument() const final {
    return "convert-linalg-to-cimdi";
  }
  mlir::StringRef getDescription() const final {
    return "Tile large matrices and lower Linalg operations to the CIMDI dialect.";
  }
};

} // namespace

// Pass注册函数
void registerLinalgToCimdiPass() {
  mlir::PassRegistration<LinalgToCimdiPass>();
}