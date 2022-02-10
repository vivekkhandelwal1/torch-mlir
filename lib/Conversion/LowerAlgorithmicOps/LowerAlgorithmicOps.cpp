//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/LowerAlgorithmicOps/LowerAlgorithmicOps.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace torch;
using namespace Torch;
using namespace TorchConversion;
using namespace torch_upstream; // For ScalarType and type
using namespace TMTensor;

// -----------------------------------------------------------------------------
// Patterns (as this grows, it should be organized into multiple files)
// -----------------------------------------------------------------------------
// This is going to eventually be O(#aten ops), which is in the 100s.
//
// Most of these patterns consist of:
// 1. Checking that the operand/result types and other static properties are
//    good-enough to create a valid linalg op (such as operands being of
//    ranks/dtypes acceptable to the linalg op).
// 2. Creating dynamic error guards, usually checking a predicate on the
//    compatibility of operand shapes.
// 3. Creating init tensors for the computation op. Usually this involves
//    reifying IR for a shape transfer function based on the operand shapes.
// 4. Creating a named linalg op to replace the original op.
//
// TODO: Use linalg OpDSL to autogenerate at least 1)/2)/3) such
// that these patterns become mostly mechanical associations of
// "aten.foo -> linalg.foo".

static LogicalResult verifyLinalgCompatibleTypes(Operation *op,
                                                 PatternRewriter &rewriter) {
  // Check the value tensor is ranked as expected by Linalg.
  // TODO: Remove this check but use a separate verification pass to verify the
  // invariants expected by later passes.
  auto isValidLinalgType = [](Type type) {
    auto tensor = type.dyn_cast<ValueTensorType>();
    return !tensor ||
           tensor.toBuiltinTensor().dyn_cast_or_null<RankedTensorType>();
  };

  bool valid = llvm::all_of(op->getOperandTypes(), isValidLinalgType) &&
               llvm::all_of(op->getResultTypes(), isValidLinalgType);
  if (!valid)
    return rewriter.notifyMatchFailure(op, "type cannot be lowered to linalg");
  return success();
}

// Creates a tensor with required `sizes` and `elemTy` and fills it with
// initElem.
static Value createInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                              Type elemTy, Value initElem) {
  Value initTensor = b.create<linalg::InitTensorOp>(loc, sizes, elemTy);
  return b.create<linalg::FillOp>(loc, initElem, initTensor).getResult(0);
}

static Value castIntToIndex(OpBuilder &b, Location loc, Value v) {
  assert(v.getType().isa<IntegerType>() && "must be called with integer type");
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

namespace {
class ConvertAtenBincountOp : public OpConversionPattern<AtenBincountOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBincountOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.self();
    Value minlength = adaptor.minlength();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();

    // Finding the maximum value in the input tensor.
    Value initTensorMaxShape = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value constantZero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value initTensorMax =
        createInitTensor(rewriter, loc, {initTensorMaxShape},
                         inputType.getElementType(), constantZero);

    AffineExpr zeroDimExpr = mlir::getAffineDimExpr(0, op.getContext());
    AffineExpr oneDimExpr = mlir::getAffineDimExpr(1, op.getContext());
    auto indexingMaps = AffineMap::inferFromExprList({zeroDimExpr, oneDimExpr});
    SmallVector<StringRef, 2> iteratorTypes{"reduction", "parallel"};

    Value maxTensor =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensorMax.getType(),
                /*input*/ input,
                /*output*/ initTensorMax,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value pred = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::ugt, args[0], args[1]);
                  Value max =
                      b.create<arith::SelectOp>(loc, pred, args[0], args[1]);
                  b.create<linalg::YieldOp>(loc, max);
                })
            ->getResult(0);

    Value maxIndex = castIntToIndex(rewriter, loc, constantZero);
    Value maxInput =
        rewriter.create<tensor::ExtractOp>(loc, maxTensor, maxIndex);

    Value inputSize = rewriter.create<tensor::DimOp>(loc, input, 0);
    Value constantOne = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    Value updates = createInitTensor(rewriter, loc, {inputSize},
                                     constantOne.getType(), constantOne);

    SmallVector<ReassociationIndices> reassociation(inputType.getRank());
    reassociation[0].push_back(0);
    reassociation[0].push_back(1);
    auto resultType =
        RankedTensorType::get({-1, 1}, inputType.getElementType());
    auto expandedInput = rewriter.create<tensor::ExpandShapeOp>(
        loc, resultType, input, reassociation);
    resultType.dump();
    expandedInput.dump();
    resultType = RankedTensorType::get(expandedInput.getType().getShape(),
                                       constantOne.getType());
    Value truncatedInput =
        rewriter.create<arith::TruncIOp>(loc, resultType, expandedInput);
    Value bincountSize =
        rewriter.create<arith::MaxUIOp>(loc, maxInput, minlength);
    bincountSize = castIntToIndex(rewriter, loc, bincountSize);

    Value bincount = createInitTensor(rewriter, loc, {bincountSize},
                                      constantOne.getType(), constantOne);

    auto scatterOp = rewriter.create<TMTensor::ScatterOp>(
        loc, bincount.getType(), ValueRange{updates, truncatedInput}, bincount,
        false);
    scatterOp->dump();
    Region &scatterOpRegion = scatterOp.region();
    if (scatterOpRegion.empty())
      llvm::errs() << "No region found\n";
    // auto regionArgs = scatterOpRegion.getArguments();
    // OpBuilder regionBuilder(scatterOpRegion);
    // regionBuilder.create<arith::AddIOp>(scatterOp->getLoc(), regionArgs[0],
    // regionArgs[1]);

    // resultType = getTypeConverter()
    //                       ->convertType(op->getResult(0).getType())
    //                       .cast<RankedTensorType>();
    // Value extendedResult =
    //     rewriter.create<arith::ExtUIOp>(loc, resultType,
    //     scatterOp->getResult(0));
    // rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
    // extendedResult);
    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertLowerAlgorithmicOps
    : public ConvertLowerAlgorithmicOpsBase<ConvertLowerAlgorithmicOps> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<StandardOpsDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    registry.insert<TMTensorDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           cf::ControlFlowDialect, math::MathDialect,
                           tensor::TensorDialect, arith::ArithmeticDialect,
                           TMTensorDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenBincountOp>();
    patterns.add<ConvertAtenBincountOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::createConvertLowerAlgorithmicOpsPass() {
  return std::make_unique<ConvertLowerAlgorithmicOps>();
}
