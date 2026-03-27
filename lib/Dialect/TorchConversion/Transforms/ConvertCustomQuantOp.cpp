//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace mlir::torch::TorchConversion {

#define GEN_PASS_DEF_CONVERTCUSTOMQUANTOP
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h.inc"

static FailureOr<int64_t> getConstantIntegerFromDefiningOp(Value operand) {
  if (auto arithConst = dyn_cast_or_null<arith::ConstantOp>(operand.getDefiningOp())) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithConst.getValue())) {
      return intAttr.getInt();
    }
  }
  auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(operand.getDefiningOp());
  if (!castOp) {
    return failure();
  }
  auto constOp =
      dyn_cast<Torch::ConstantIntOp>(castOp.getOperand(0).getDefiningOp());
  if (!constOp) {
    return failure();
  }
  return constOp.getValue();
}

namespace {
class ConvertCustomQuantizedMatmulOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getName().str() != "quant.matmul_rhs_group_quant") {
      return failure();
    }
    Location loc = op->getLoc();
    if (failed(verifyLinalgCompatibleTypes(op, rewriter))) {
      return failure();
    }

    // get inputs: lhs, rhsQuant, scales, zps
    Value lhs = adaptor.getOperands()[0];
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    if (!lhsType) {
      return failure();
    }
    auto lhsShape = lhsType.getShape();
    int lhsReductDimSize = lhsShape.back();

    Value rhsQuant = adaptor.getOperands()[1];
    auto rhsType = cast<RankedTensorType>(rhsQuant.getType());
    if (!rhsType) {
      return failure();
    }
    auto rhsShape = rhsType.getShape();
    int rhsReductDimSize = rhsShape.back();
    Type rhsElementType = rhsType.getElementType();

    Value scales = adaptor.getOperands()[2];
    Value zps = adaptor.getOperands()[3];
    Value unpackedTypeWidth = adaptor.getOperands()[4];
    Value groupSize = adaptor.getOperands()[5];

    FailureOr<int64_t> maybeGroupSize = getConstantIntegerFromDefiningOp(groupSize);
    if (failed(maybeGroupSize)) {
      return failure();
    }
    FailureOr<int64_t> maybeUnpackedBitWidth =
        getConstantIntegerFromDefiningOp(unpackedTypeWidth);
    if (failed(maybeUnpackedBitWidth)) {
      return failure();
    }
    int64_t gs = *maybeGroupSize;
    int64_t unpackedBitWidth = *maybeUnpackedBitWidth;
    if (unpackedBitWidth !=
        static_cast<int>(rhsElementType.getIntOrFloatBitWidth())) {
      return failure();
    }

    // get outputs
    Type newResultType = getTypeConverter()->convertType(op.getType(0));
    auto resultType = cast<RankedTensorType>(newResultType);
    if (!resultType) {
      return failure();
    }
    auto resultShape = resultType.getShape();
    Type elementType = resultType.getElementType();

    // expand lhs
    std::vector<int64_t> lhsExpandedShape = {lhsShape[0], lhsShape[1],
                                             lhsReductDimSize / gs, gs};
    RankedTensorType lhsExpandedType =
        RankedTensorType::get(lhsExpandedShape, elementType);
    SmallVector<ReassociationIndices, 4> lhsReassociation = {{0}, {1}, {2, 3}};
    Value lhsExpanded = tensor::ExpandShapeOp::create(
        rewriter, loc, lhsExpandedType, lhs, lhsReassociation);

    // expand rhs
    std::vector<int64_t> rhsExpandedShape = {rhsShape[0], rhsReductDimSize / gs,
                                             gs};
    RankedTensorType rhsExpandedType =
        RankedTensorType::get(rhsExpandedShape, rhsElementType);
    SmallVector<ReassociationIndices, 4> rhsReassociation = {{0}, {1, 2}};
    Value rhsExpanded = tensor::ExpandShapeOp::create(
        rewriter, loc, rhsExpandedType, rhsQuant, rhsReassociation);
    Value cst0 = arith::ConstantOp::create(rewriter, loc,
                                           FloatAttr::get(elementType, 0.0));

    auto scalesType = dyn_cast<RankedTensorType>(scales.getType());
    auto zpsType = dyn_cast<RankedTensorType>(zps.getType());
    if (!scalesType || !zpsType || scalesType.getRank() != 3 ||
        zpsType.getRank() != 3) {
      return failure();
    }
    if (!isa<FloatType>(elementType)) {
      return failure();
    }

    SmallVector<Value> dynDims;
    for (int i = 0; i < lhsType.getRank(); i++) {
      if (lhsType.isDynamicDim(i)) {
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, lhs, i));
      }
    }
    Value empty = tensor::EmptyOp::create(rewriter, loc, resultShape,
                                          elementType, dynDims);
    Value output =
        linalg::FillOp::create(rewriter, loc, cst0, empty).getResult(0);

    AffineExpr d0, d1, d2, d3, d4;
    bindDims(getContext(), d0, d1, d2, d3, d4);
    auto c0 = rewriter.getAffineConstantExpr(0);
    auto lhsMap = AffineMap::get(5, 0, {d0, d1, d3, d4}, rewriter.getContext());
    auto rhsMap = AffineMap::get(5, 0, {d2, d3, d4}, rewriter.getContext());
    auto scaleMap = AffineMap::get(5, 0, {d2, d3, c0}, rewriter.getContext());
    auto zpMap = AffineMap::get(5, 0, {d2, d3, c0}, rewriter.getContext());
    auto outMap = AffineMap::get(5, 0, {d0, d1, d2}, rewriter.getContext());
    SmallVector<AffineMap, 5> fusedIndexingMaps = {lhsMap, rhsMap, scaleMap,
                                                   zpMap, outMap};
    SmallVector<utils::IteratorType> fusedIteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::reduction,
        utils::IteratorType::reduction};

    bool rhsIsInteger = isa<IntegerType>(rhsElementType);
    bool rhsIsFloat = isa<FloatType>(rhsElementType);
    if (!rhsIsInteger && !rhsIsFloat) {
      return failure();
    }

    Value matmulQuant = linalg::GenericOp::create(
                            rewriter, loc, output.getType(),
                            ValueRange{lhsExpanded, rhsExpanded, scales, zps},
                            output,
                            /*indexingMaps=*/fusedIndexingMaps,
                            /*iteratorTypes=*/fusedIteratorTypes,
                            [&](OpBuilder &b, Location bodyLoc,
                                ValueRange args) {
                              auto castToElementType = [&](Value value) -> Value {
                                Type type = value.getType();
                                if (type == elementType) {
                                  return value;
                                }
                                if (auto floatType = dyn_cast<FloatType>(type)) {
                                  auto targetFloatType =
                                      cast<FloatType>(elementType);
                                  if (floatType.getWidth() <
                                      targetFloatType.getWidth()) {
                                    return arith::ExtFOp::create(
                                        b, bodyLoc, elementType, value);
                                  }
                                  if (floatType.getWidth() >
                                      targetFloatType.getWidth()) {
                                    return arith::TruncFOp::create(
                                        b, bodyLoc, elementType, value);
                                  }
                                  return value;
                                }
                                if (auto intType = dyn_cast<IntegerType>(type)) {
                                  if (intType.isUnsignedInteger()) {
                                    return arith::UIToFPOp::create(
                                        b, bodyLoc, elementType, value);
                                  }
                                  return arith::SIToFPOp::create(
                                      b, bodyLoc, elementType, value);
                                }
                                return Value();
                              };

                              Value lhsValue = castToElementType(args[0]);
                              Value rhsValue = castToElementType(args[1]);
                              Value scaleValue = castToElementType(args[2]);
                              Value zpValue = castToElementType(args[3]);
                              Value out = args[4];
                              if (!lhsValue || !rhsValue || !scaleValue ||
                                  !zpValue) {
                                linalg::YieldOp::create(b, bodyLoc, out);
                                return;
                              }

                              Value dequantizedRhs = rhsValue;
                              if (rhsIsInteger) {
                                dequantizedRhs = arith::SubFOp::create(
                                    b, bodyLoc, rhsValue, zpValue);
                              }
                              dequantizedRhs = arith::MulFOp::create(
                                  b, bodyLoc, dequantizedRhs, scaleValue);
                              Value product = arith::MulFOp::create(
                                  b, bodyLoc, lhsValue, dequantizedRhs);
                              Value accum =
                                  arith::AddFOp::create(b, bodyLoc, product, out);
                              linalg::YieldOp::create(b, bodyLoc, accum);
                            })
                            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, matmulQuant);
    return success();
  }
};

class ConvertTorchPrimsDevicePutOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getName().str() != "torch.prims.device_put") {
      return failure();
    }
    if (adaptor.getOperands().empty() || op->getNumResults() != 1) {
      return failure();
    }

    Value forwarded = adaptor.getOperands()[0];
    Type resultType = getTypeConverter()->convertType(op.getType(0));
    if (!resultType) {
      return failure();
    }
    if (forwarded.getType() != resultType) {
      auto rankedResultType = dyn_cast<RankedTensorType>(resultType);
      if (!rankedResultType) {
        return failure();
      }
      forwarded = tensor::CastOp::create(rewriter, op.getLoc(),
                                         rankedResultType, forwarded);
    }

    rewriter.replaceOp(op, forwarded);
    return success();
  }
};

class ConvertTorchAtenBitwiseLeftShiftTensorScalarOp
    : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getName().str() != "torch.aten.bitwise_left_shift.Tensor_Scalar") {
      return failure();
    }
    if (failed(verifyLinalgCompatibleTypes(op, rewriter))) {
      return failure();
    }
    if (adaptor.getOperands().size() != 2 || op->getNumResults() != 1) {
      return failure();
    }

    Value input = adaptor.getOperands()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return failure();
    }
    auto elementType = dyn_cast<IntegerType>(inputType.getElementType());
    if (!elementType) {
      return failure();
    }

    FailureOr<int64_t> maybeShiftAmount =
        getConstantIntegerFromDefiningOp(adaptor.getOperands()[1]);
    if (failed(maybeShiftAmount)) {
      return failure();
    }
    int64_t shiftAmount = *maybeShiftAmount;

    Type newResultType = getTypeConverter()->convertType(op.getType(0));
    auto resultType = dyn_cast<RankedTensorType>(newResultType);
    if (!resultType) {
      return failure();
    }

    SmallVector<Value> dynDims;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      if (inputType.isDynamicDim(i)) {
        dynDims.push_back(tensor::DimOp::create(rewriter, op.getLoc(), input, i));
      }
    }
    Value empty = tensor::EmptyOp::create(rewriter, op.getLoc(),
                                          resultType.getShape(),
                                          resultType.getElementType(), dynDims);

    SmallVector<AffineMap, 2> indexingMaps = {
        rewriter.getMultiDimIdentityMap(inputType.getRank()),
        rewriter.getMultiDimIdentityMap(inputType.getRank())};
    SmallVector<utils::IteratorType> iteratorTypes(
        inputType.getRank(), utils::IteratorType::parallel);

    Value shifted = linalg::GenericOp::create(
                        rewriter, op.getLoc(), empty.getType(), input, empty,
                        indexingMaps, iteratorTypes,
                        [&](OpBuilder &b, Location loc, ValueRange args) {
                          Value shiftConst = arith::ConstantOp::create(
                              b, loc,
                              IntegerAttr::get(args[0].getType(), shiftAmount));
                          Value shl =
                              arith::ShLIOp::create(b, loc, args[0], shiftConst);
                          linalg::YieldOp::create(b, loc, shl);
                        })
                        .getResult(0);
    rewriter.replaceOp(op, shifted);
    return success();
  }
};

class ConvertTorchAoChooseQParamsAffineOp
    : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getName().str() != "torch.torchao.choose_qparams_affine") {
      return failure();
    }
    Location loc = op->getLoc();
    if (failed(verifyLinalgCompatibleTypes(op, rewriter))) {
      return failure();
    }

    auto inputType = dyn_cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    if (!inputType) {
      return failure();
    }

    Type newScaleType = getTypeConverter()->convertType(op.getType(0));
    Type newZeroPointType = getTypeConverter()->convertType(op.getType(1));
    auto scaleType = dyn_cast<RankedTensorType>(newScaleType);
    auto zeroPointType = dyn_cast<RankedTensorType>(newZeroPointType);
    if (!scaleType || !zeroPointType) {
      return failure();
    }

    if (inputType.getRank() < scaleType.getRank() ||
        inputType.getRank() < zeroPointType.getRank()) {
      return failure();
    }

    SmallVector<Value> scaleDynDims;
    for (int64_t i = 0; i < scaleType.getRank(); ++i) {
      if (scaleType.isDynamicDim(i)) {
        scaleDynDims.push_back(
            tensor::DimOp::create(rewriter, loc, adaptor.getOperands()[0], i));
      }
    }
    SmallVector<Value> zeroPointDynDims;
    for (int64_t i = 0; i < zeroPointType.getRank(); ++i) {
      if (zeroPointType.isDynamicDim(i)) {
        zeroPointDynDims.push_back(
            tensor::DimOp::create(rewriter, loc, adaptor.getOperands()[0], i));
      }
    }

    Value scaleEmpty = tensor::EmptyOp::create(
        rewriter, loc, scaleType.getShape(), scaleType.getElementType(),
        scaleDynDims);
    Value zeroPointEmpty = tensor::EmptyOp::create(
        rewriter, loc, zeroPointType.getShape(), zeroPointType.getElementType(),
        zeroPointDynDims);

    auto getFloatConst = [&](double value, Type type) -> Value {
      auto floatType = dyn_cast<FloatType>(type);
      if (!floatType) {
        return Value();
      }
      return arith::ConstantOp::create(rewriter, loc,
                                       FloatAttr::get(floatType, value));
    };

    Value one = getFloatConst(1.0, scaleType.getElementType());
    Value zero = getFloatConst(0.0, zeroPointType.getElementType());
    if (!one || !zero) {
      return failure();
    }

    Value scales =
        linalg::FillOp::create(rewriter, loc, one, scaleEmpty).getResult(0);
    Value zeroPoints =
        linalg::FillOp::create(rewriter, loc, zero, zeroPointEmpty).getResult(0);
    rewriter.replaceOp(op, ValueRange{scales, zeroPoints});
    return success();
  }
};

class ConvertTorchAoQuantizeAffineOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getName().str() != "torch.torchao.quantize_affine") {
      return failure();
    }
    Location loc = op->getLoc();
    if (failed(verifyLinalgCompatibleTypes(op, rewriter))) {
      return failure();
    }

    Value input = adaptor.getOperands()[0];
    Value scale = adaptor.getOperands()[2];
    Value zeroPoint = adaptor.getOperands()[3];
    Value qminValue = adaptor.getOperands()[5];
    Value qmaxValue = adaptor.getOperands()[6];

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto scaleType = dyn_cast<RankedTensorType>(scale.getType());
    auto zeroPointType = dyn_cast<RankedTensorType>(zeroPoint.getType());
    if (!inputType || !scaleType || !zeroPointType) {
      return failure();
    }
    Type newResultType = getTypeConverter()->convertType(op.getType(0));
    auto resultType = dyn_cast<RankedTensorType>(newResultType);
    if (!resultType) {
      return failure();
    }

    int64_t rank = inputType.getRank();
    if (rank <= 0 || resultType.getRank() != rank) {
      return failure();
    }
    if (scaleType.getRank() != rank && scaleType.getRank() != rank - 1 &&
        scaleType.getRank() != 0) {
      return failure();
    }
    if (zeroPointType.getRank() != rank && zeroPointType.getRank() != rank - 1 &&
        zeroPointType.getRank() != 0) {
      return failure();
    }

    auto resultElementType = dyn_cast<IntegerType>(resultType.getElementType());
    if (!resultElementType) {
      return failure();
    }

    int64_t qmin = -127;
    int64_t qmax = 127;
    if (FailureOr<int64_t> maybeQmin =
            getConstantIntegerFromDefiningOp(qminValue);
        succeeded(maybeQmin)) {
      qmin = *maybeQmin;
    }
    if (FailureOr<int64_t> maybeQmax =
            getConstantIntegerFromDefiningOp(qmaxValue);
        succeeded(maybeQmax)) {
      qmax = *maybeQmax;
    }

    SmallVector<Value> dynDims;
    for (int64_t i = 0; i < rank; ++i) {
      if (inputType.isDynamicDim(i)) {
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, input, i));
      }
    }
    Value empty = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                          resultType.getElementType(), dynDims);

    SmallVector<AffineExpr> dimExprs;
    dimExprs.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      dimExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    AffineMap inputMap = AffineMap::get(rank, 0, dimExprs, rewriter.getContext());

    SmallVector<AffineExpr> scaleExprs;
    if (scaleType.getRank() == rank) {
      scaleExprs = dimExprs;
    } else if (scaleType.getRank() == rank - 1) {
      scaleExprs.assign(dimExprs.begin(), dimExprs.end() - 1);
    }
    AffineMap scaleMap =
        AffineMap::get(rank, 0, scaleExprs, rewriter.getContext());

    SmallVector<AffineExpr> zeroPointExprs;
    if (zeroPointType.getRank() == rank) {
      zeroPointExprs = dimExprs;
    } else if (zeroPointType.getRank() == rank - 1) {
      zeroPointExprs.assign(dimExprs.begin(), dimExprs.end() - 1);
    }
    AffineMap zeroPointMap =
        AffineMap::get(rank, 0, zeroPointExprs, rewriter.getContext());

    SmallVector<AffineMap, 4> indexingMaps = {inputMap, scaleMap, zeroPointMap,
                                              inputMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    FloatType computeType = dyn_cast<FloatType>(inputType.getElementType());
    if (!computeType) {
      computeType = rewriter.getF32Type();
    }

    Value quantized = linalg::GenericOp::create(
                          rewriter, loc, empty.getType(),
                          ValueRange{input, scale, zeroPoint}, empty,
                          /*indexingMaps=*/indexingMaps,
                          /*iteratorTypes=*/iteratorTypes,
                          [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
                            auto castToComputeType = [&](Value value) -> Value {
                              Type type = value.getType();
                              if (type == computeType) {
                                return value;
                              }
                              if (auto floatType = dyn_cast<FloatType>(type)) {
                                if (floatType.getWidth() < computeType.getWidth()) {
                                  return arith::ExtFOp::create(b, bodyLoc,
                                                               computeType, value);
                                }
                                if (floatType.getWidth() > computeType.getWidth()) {
                                  return arith::TruncFOp::create(
                                      b, bodyLoc, computeType, value);
                                }
                                return value;
                              }
                              if (auto intType = dyn_cast<IntegerType>(type)) {
                                if (intType.isUnsignedInteger()) {
                                  return arith::UIToFPOp::create(b, bodyLoc,
                                                                 computeType, value);
                                }
                                return arith::SIToFPOp::create(b, bodyLoc,
                                                               computeType, value);
                              }
                              return Value();
                            };

                            Value inputF = castToComputeType(args[0]);
                            Value scaleF = castToComputeType(args[1]);
                            Value zeroPointF = castToComputeType(args[2]);
                            if (!inputF || !scaleF || !zeroPointF) {
                              linalg::YieldOp::create(b, bodyLoc, args[3]);
                              return;
                            }

                            Value scaled =
                                arith::DivFOp::create(b, bodyLoc, inputF, scaleF);
                            Value shifted = arith::AddFOp::create(
                                b, bodyLoc, scaled, zeroPointF);
                            Value qminF = arith::ConstantOp::create(
                                b, bodyLoc, FloatAttr::get(computeType, qmin));
                            Value qmaxF = arith::ConstantOp::create(
                                b, bodyLoc, FloatAttr::get(computeType, qmax));
                            Value clampedLow = arith::MaximumFOp::create(
                                b, bodyLoc, shifted, qminF);
                            Value clamped = arith::MinimumFOp::create(
                                b, bodyLoc, clampedLow, qmaxF);

                            bool resultIsUnsigned =
                                resultElementType.isUnsignedInteger();
                            Value i32Quantized;
                            if (resultIsUnsigned) {
                              i32Quantized = arith::FPToUIOp::create(
                                  b, bodyLoc, b.getIntegerType(32), clamped);
                            } else {
                              i32Quantized = arith::FPToSIOp::create(
                                  b, bodyLoc, b.getIntegerType(32), clamped);
                            }

                            int resultBitWidth =
                                resultElementType.getIntOrFloatBitWidth();
                            Value casted = i32Quantized;
                            if (resultBitWidth < 32) {
                              casted = arith::TruncIOp::create(
                                  b, bodyLoc, resultElementType, casted);
                            } else if (resultBitWidth > 32) {
                              if (resultIsUnsigned) {
                                casted = arith::ExtUIOp::create(
                                    b, bodyLoc, resultElementType, casted);
                              } else {
                                casted = arith::ExtSIOp::create(
                                    b, bodyLoc, resultElementType, casted);
                              }
                            }

                            linalg::YieldOp::create(b, bodyLoc, casted);
                          })
                          .getResult(0);
    rewriter.replaceOp(op, quantized);
    return success();
  }
};
} // namespace

namespace {
class ConvertCustomQuantOpPass
    : public impl::ConvertCustomQuantOpBase<ConvertCustomQuantOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<Torch::TorchDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
                           tensor::TensorDialect, arith::ArithDialect,
                           Torch::TorchDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<OperatorOp>();
    patterns.add<ConvertCustomQuantizedMatmulOp, ConvertTorchPrimsDevicePutOp,
                 ConvertTorchAtenBitwiseLeftShiftTensorScalarOp,
                 ConvertTorchAoChooseQParamsAffineOp,
                 ConvertTorchAoQuantizeAffineOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertCustomQuantOpPass() {
  return std::make_unique<ConvertCustomQuantOpPass>();
}

} // namespace mlir::torch::TorchConversion
