#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

// 引入你自己的方言头文件
#include "mlir/Dialect/Cimdi/CimdiDialect.h"

// 引入TableGen自动生成的操作头文件
#define GET_OP_CLASSES
#include "CimdiOps.h.inc"

namespace {

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
    
    auto innerLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, upperK, stepK, mlir::ValueRange{currentC});
    
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    mlir::Value k = innerLoop.getInductionVar();
    currentC = innerLoop.getBody()->getArgument(1);
    
    // 计算当前分块的实际大小
    mlir::Value tileSizeM = rewriter.create<mlir::arith::MinSIOp>(
        loc, rewriter.create<mlir::arith::SubIOp>(loc, upperM, i), stepM);
    mlir::Value tileSizeN = rewriter.create<mlir::arith::MinSIOp>(
        loc, rewriter.create<mlir::arith::SubIOp>(loc, upperN, j), stepN);
    mlir::Value tileSizeK = rewriter.create<mlir::arith::MinSIOp>(
        loc, rewriter.create<mlir::arith::SubIOp>(loc, upperK, k), stepK);
    
    // 提取子矩阵
    llvm::SmallVector<mlir::OpFoldResult> offsetsA = {i, k};
    llvm::SmallVector<mlir::OpFoldResult> sizesA = {tileSizeM, tileSizeK};
    llvm::SmallVector<mlir::OpFoldResult> stridesA = {c1, c1};
    
    llvm::SmallVector<mlir::OpFoldResult> offsetsB = {k, j};
    llvm::SmallVector<mlir::OpFoldResult> sizesB = {tileSizeK, tileSizeN};
    llvm::SmallVector<mlir::OpFoldResult> stridesB = {c1, c1};
    
    llvm::SmallVector<mlir::OpFoldResult> offsetsC = {i, j};
    llvm::SmallVector<mlir::OpFoldResult> sizesC = {tileSizeM, tileSizeN};
    llvm::SmallVector<mlir::OpFoldResult> stridesC = {c1, c1};
    
    mlir::Value subA = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, A, offsetsA, sizesA, stridesA);
    mlir::Value subB = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, B, offsetsB, sizesB, stridesB);
    mlir::Value subC = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, currentC, offsetsC, sizesC, stridesC);
    
    // 创建 CIMDi PE matmul 操作
    auto peMatmul = rewriter.create<mlir::cimdi::CimdiPEMatmulOp>(
        loc, subC.getType(), subA, subB, subC);
    
    // 将结果插回原矩阵
    mlir::Value updatedC = rewriter.create<mlir::tensor::InsertSliceOp>(
        loc, peMatmul.getResult(), currentC, offsetsC, sizesC, stridesC);
    
    rewriter.create<mlir::scf::YieldOp>(loc, updatedC);
    
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
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    // 添加Lowering Pattern
    patterns.add<MatmulOpConversion>(context);

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