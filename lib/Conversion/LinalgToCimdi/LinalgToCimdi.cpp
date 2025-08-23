#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// 引入你自己的方言头文件
#include "mlir/Dialect/Cimdi/CimdiDialect.h"

// 引入TableGen自动生成的操作头文件
#define GET_OP_CLASSES
#include "CimdiOps.h.inc"

namespace {

// 负责将尺寸匹配的 linalg.matmul 转换为 cimdi.digital.pe.matmul
class MatmulOpConversion
    : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
public:
  explicit MatmulOpConversion(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::linalg::MatmulOp>(context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::MatmulOp op,
                  mlir::PatternRewriter &rewriter) const override {

    // 检查：只转换那些尺寸小于或等于128x128的matmul
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getOutputs()[0].getType());
    if (!outputType || outputType.getShape().size() != 2 ||
        outputType.getShape()[0] > 128 || outputType.getShape()[1] > 128) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::cimdi::CimdiPEMatmulOp>(
        op, op.getResult(0).getType(), op.getInputs()[0],
        op.getInputs()[1], op.getOutputs()[0]);

    return mlir::success();
  }
};

// 主Pass，驱动Tiling和Lowering
struct LinalgToCimdiPass
    : public mlir::PassWrapper<LinalgToCimdiPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToCimdiPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::cimdi::CimdiDialect>();
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