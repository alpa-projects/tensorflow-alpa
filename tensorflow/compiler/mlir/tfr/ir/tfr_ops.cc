/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.h"

#include <algorithm>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/FunctionImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tfr/ir/tfr_types.h"

namespace mlir {

namespace TFR {

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for inlining within the TFR dialect.
struct TFRInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // Returns true if the given region 'src' can be inlined into the region
  // 'dest' that is attached to an operation registered to the current dialect.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }

  // Returns true if the given operation 'op', that is registered to this
  // dialect, can be inlined into the region 'dest' that is attached to an
  // operation registered to the current dialect.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }

  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    auto retValOp = dyn_cast<TFRReturnOp>(op);
    if (!retValOp) return;

    for (auto ret_value : llvm::zip(valuesToRepl, retValOp.operands())) {
      std::get<0>(ret_value).replaceAllUsesWith(std::get<1>(ret_value));
    }
  }

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
    if (!input.getType().isa<IntegerType>() ||
        !result_type.isa<IntegerType>()) {
      return nullptr;
    }
    auto input_itype = input.getType().cast<IntegerType>();
    auto result_itype = result_type.cast<IntegerType>();
    if (input_itype.getWidth() == result_itype.getWidth()) return nullptr;
    if (input_itype.getWidth() > result_itype.getWidth()) {
      return builder.create<TruncateIOp>(conversion_loc, result_type, input);
    } else {
      return builder.create<SignExtendIOp>(conversion_loc, result_type, input);
    }
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// TFR Dialect
//===----------------------------------------------------------------------===//

TFRDialect::TFRDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfr", context, TypeID::get<TFRDialect>()) {
  // TFR depends on TensorFlow for its canonicalization
  context->getOrLoadDialect<TF::TensorFlowDialect>();

  addTypes<TFRTensorType, TFRTensorListType, TFRAttrType>();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc.inc"
      >();

  addInterfaces<TFRInlinerInterface>();
}

Operation *TFRDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (ConstantOp::isBuildableWith(value, type))
    return builder.create<ConstantOp>(loc, type, value);
  return nullptr;
}

bool TFRType::classof(Type type) {
  return llvm::isa<TFRDialect>(type.getDialect());
}

//===----------------------------------------------------------------------===//
// Custom op methods
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ConstantTensorOp op) {
  auto input_type = op.arg().getType();
  auto output_type = op.out().getType();

  if (auto output_tensor_type = output_type.dyn_cast<TFRTensorType>()) {
    return success();
  }

  auto output_tensor_type = output_type.dyn_cast<RankedTensorType>();
  if (!output_tensor_type || !output_tensor_type.hasStaticShape()) {
    op.emitError("output type should be static and ranked.");
    return failure();
  }

  if (output_tensor_type.getRank() == 0) {
    bool same_scalar = output_tensor_type.getElementType() == input_type;
    if (!same_scalar) {
      op.emitError("input and output should have the same scalar types.");
    }
    return success(same_scalar);
  }

  if (auto input_vector_type = input_type.dyn_cast<VectorType>()) {
    bool same_element_type = output_tensor_type.getElementType() ==
                             input_vector_type.getElementType();
    bool same_shape =
        output_tensor_type.getShape() == input_vector_type.getShape();
    if (!same_element_type || !same_shape) {
      op.emitError("input and output should have same shape and element type.");
    }
    return success(same_element_type && same_shape);
  }

  op.emitError("input can not be converted to an output tensor.");
  return failure();
}

static LogicalResult Verify(TFRFuncOp func) {
  // Collect all attribute names used by the tensor and tensor list arguments
  // and returns. Also, collect the names of all the attribute arguments as the
  // defined list. Later on, the used attribute names will be verified to be in
  // the defined list.
  llvm::SmallVector<StringAttr, 4> input_used_attrs, output_used_attrs;

  // While scanning the arguments, record the start/end indices of each argument
  // type, so the order can be verified as well.
  // TODO(fengliuai): the attribute arguments with default values need to be
  // at the end?
  int first_tensor = -1, last_tensor = -1, first_tensor_list = -1,
      last_tensor_list = -1, first_attr = -1;
  for (auto arg : llvm::enumerate(func.getType().getInputs())) {
    Type arg_type = arg.value();

    if (auto tensor = arg_type.dyn_cast<TFRTensorType>()) {
      if (first_tensor == -1) {
        first_tensor = arg.index();
      }
      last_tensor = arg.index();
      auto used = tensor.getAttrKeys();
      input_used_attrs.append(used.begin(), used.end());
      continue;
    }

    if (auto tensor_list = arg_type.dyn_cast<TFRTensorListType>()) {
      if (first_tensor_list == -1) {
        first_tensor_list = arg.index();
      }
      last_tensor_list = arg.index();
      auto used = tensor_list.getAttrKeys();
      input_used_attrs.append(used.begin(), used.end());
      continue;
    }

    if (!arg_type.isa<TensorType>()) {
      if (first_attr == -1) {
        first_attr = arg.index();
      }
      auto name =
          func.getArgAttrOfType<StringAttr>(arg.index(), kAttrArgumentNameAttr);
      if (!name) {
        func.emitError(
            llvm::Twine(arg.index()) +
            " attribute argument doesn't have a tfr.name attribute.");
        return failure();
      }
      continue;
    }

    func.emitError("Builtin TensorType isn't allowed as the argument.");
    return failure();
  }

  // Collect all the undefined attributes used in the inputs.
  llvm::SmallVector<StringAttr, 4> undefined_attrs;
  for (auto attr : input_used_attrs) {
    if (!func->getAttr(attr.getValue())) {
      undefined_attrs.push_back(attr);
    }
  }

  // Verify the argument order: tensors, tensor list, attributes; and also
  // verify there is at most one tensor list argument.
  if (first_attr != -1 &&
      (first_attr < last_tensor_list || first_attr < last_tensor)) {
    func.emitError(
        "tfr.tensor/tfr.tensor_list argument should be before non tensor "
        "arguments.");
    return failure();
  }
  // The order between tensor arguments and tensor list arguments and the number
  // of tensor list arguments are verified only when they couldn't be determined
  // by the attributes.
  if (!undefined_attrs.empty()) {
    if (first_tensor_list != -1 && first_tensor_list < last_tensor) {
      func.emitError(
          "tfr.tensor argument should be before tfr.tensor_list argument.");
      return failure();
    }
    if (first_tensor_list != last_tensor_list) {
      func.emitError("More than one tfr.tensor_list argument isn't allowed.");
      return failure();
    }
  }

  // Verify the result order: tensor, tensor list, and also verify at most one
  // tensor list result.
  int undefined_input_attrs_number = undefined_attrs.size();
  bool seen_tensor_list = false, has_tensor_list_order_error = false,
       has_multiple_tensor_lists_error = false;
  for (auto result_type : func.getType().getResults()) {
    if (auto tensor = result_type.dyn_cast<TFRTensorType>()) {
      if (seen_tensor_list) {
        has_tensor_list_order_error = true;
      } else {
        auto used = tensor.getAttrKeys();
        output_used_attrs.append(used.begin(), used.end());
      }
      continue;
    }

    if (auto tensor_list = result_type.dyn_cast<TFRTensorListType>()) {
      if (seen_tensor_list) {
        has_multiple_tensor_lists_error = true;
      } else {
        seen_tensor_list = true;
        auto used = tensor_list.getAttrKeys();
        output_used_attrs.append(used.begin(), used.end());
      }
      continue;
    }

    func.emitError(
        "None tfr.tensor/tfr.tensor_list results aren't allowed as a "
        "result.");
    return failure();
  }

  // Collect all the undefined attributes used in the outputs.
  for (auto attr : output_used_attrs) {
    if (!func->getAttr(attr.getValue())) {
      undefined_attrs.push_back(attr);
    }
  }

  // Verify there are no tensor/tensor list order error and multiple tensor
  // list arguments error.
  if (undefined_input_attrs_number != undefined_attrs.size()) {
    if (has_tensor_list_order_error) {
      func.emitError(
          "tfr.tensor result should be before tfr.tensor_list result.");
      return failure();
    } else if (has_multiple_tensor_lists_error) {
      func.emitError("More than one tfr.tensor_list result isn't allowed.");
      return failure();
    }
  }

  // TODO(fengliuai): We might want to refine this constraint because the
  // tensor element type can be derived.
  if (!undefined_attrs.empty()) {
    llvm::SmallVector<std::string, 4> attr_names(undefined_attrs.size());
    std::transform(undefined_attrs.begin(), undefined_attrs.end(),
                   attr_names.begin(),
                   [](StringAttr attr) { return attr.getValue().str(); });
    func.emitError(llvm::Twine("Undefined attributes are used: ",
                               llvm::join(attr_names, ",")));
    return failure();
  }

  return success();
}

static ParseResult ParseFuncOp(OpAsmParser &parser, OperationState *result) {
  auto build_func_type = [](Builder &builder, ArrayRef<Type> arg_types,
                            ArrayRef<Type> results,
                            function_like_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(arg_types, results);
  };
  return function_like_impl::parseFunctionLikeOp(
      parser, *result, /*allowVariadic=*/false, build_func_type);
}

static void PrintFuncOp(OpAsmPrinter &p, TFRFuncOp op) {
  FunctionType fn_type = op.getType();
  function_like_impl::printFunctionLikeOp(
      p, op, fn_type.getInputs(), /*isVariadic=*/false, fn_type.getResults());
}

}  // namespace TFR
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc.inc"

namespace mlir {
namespace TFR {
struct ConvertConstToTensorConst : public OpRewritePattern<ConstantTensorOp> {
  using OpRewritePattern<ConstantTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantTensorOp cst_tensor_op,
                                PatternRewriter &rewriter) const override {
    Location loc = cst_tensor_op.getLoc();
    Type out_type = cst_tensor_op.getType();
    Operation *new_cst = nullptr;

    ArrayAttr array;
    if (matchPattern(cst_tensor_op.arg(), m_Constant(&array))) {
      llvm::DenseSet<Type> all_types;
      for (auto it : array) {
        all_types.insert(it.getType());
      }
      if (all_types.size() != 1) return failure();
      ShapedType new_out_type = RankedTensorType::get(
          {static_cast<int64_t>(array.size())}, *all_types.begin());
      DenseElementsAttr attr =
          DenseElementsAttr::get(new_out_type, array.getValue());
      new_cst = rewriter.create<TF::ConstOp>(loc, new_out_type, attr);
      if (out_type.isa<TFRTensorType>()) {
        new_cst = rewriter.create<CastOp>(loc, out_type, new_cst->getResult(0));
      }
      rewriter.replaceOp(cst_tensor_op, new_cst->getResult(0));
      return success();
    }

    Attribute scalar;
    if (matchPattern(cst_tensor_op.arg(), m_Constant(&scalar))) {
      Type new_out_type = RankedTensorType::get({}, scalar.getType());
      new_cst = rewriter.create<TF::ConstOp>(loc, new_out_type, scalar);
      if (out_type.isa<TFRTensorType>()) {
        new_cst = rewriter.create<CastOp>(loc, out_type, new_cst->getResult(0));
      }
      rewriter.replaceOp(cst_tensor_op, new_cst->getResult(0));
      return success();
    }
    return failure();
  }
};

struct RemoveRedundantCast : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp cast_op,
                                PatternRewriter &rewriter) const override {
    auto preceding_cast =
        llvm::dyn_cast_or_null<CastOp>(cast_op.arg().getDefiningOp());
    if (!preceding_cast) {
      return failure();
    }
    Value input = preceding_cast.arg();
    Type input_type = input.getType();
    Type output_type = cast_op.getType();

    // Preserve quantization information for intermediate tensors.
    auto intermediate_type = preceding_cast.getType().dyn_cast<TensorType>();
    if (intermediate_type &&
        intermediate_type.getElementType().isa<quant::QuantizedType>()) {
      return failure();
    }

    // If the two types are the same, the back-to-back tfr.cast ops can be
    // removed.
    if (input_type == output_type || output_type.isa<UnrankedTensorType>()) {
      rewriter.replaceOp(cast_op, {input});
      return success();
    }

    // If the rank of the input tensor isn't ranked, we replace the pair
    // with tf.EnsureShape op so it can be removed after shape inference or
    // confirmed at runtime.
    if (input_type.isa<UnrankedTensorType>() && output_type.isa<ShapedType>()) {
      auto shape = output_type.cast<ShapedType>().getShape();
      auto shape_attr = TF::ShapeAttr::get(rewriter.getContext(), shape);
      rewriter.replaceOpWithNewOp<TF::EnsureShapeOp>(cast_op, output_type,
                                                     input, shape_attr);
    }

    return success();
  }
};

struct GetTensorShape : public OpRewritePattern<GetShapeOp> {
  using OpRewritePattern<GetShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetShapeOp shape_op,
                                PatternRewriter &rewriter) const override {
    Operation *preceding_op = shape_op.arg().getDefiningOp();
    if (auto cast_op = llvm::dyn_cast_or_null<CastOp>(preceding_op)) {
      // replace this pair by shape.shape_of, so the folding works.
      rewriter.replaceOpWithNewOp<shape::ShapeOfOp>(shape_op, cast_op.arg());
      return success();
    }
    return failure();
  }
};

struct RemoveRedundantGetElement : public OpRewritePattern<GetElementOp> {
  using OpRewritePattern<GetElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetElementOp ge_op,
                                PatternRewriter &rewriter) const override {
    IntegerAttr index;
    if (!matchPattern(ge_op.index(), m_Constant(&index))) {
      return failure();
    }
    auto preceding_build_list = llvm::dyn_cast_or_null<BuildListOp>(
        ge_op.tensor_list().getDefiningOp());
    if (!preceding_build_list ||
        preceding_build_list.getNumOperands() <= index.getInt()) {
      return failure();
    }
    Value input = preceding_build_list.getOperand(index.getInt());
    Type output_type = ge_op.getType();
    if (input.getType() != output_type &&
        !output_type.isa<UnrankedTensorType>()) {
      return failure();
    }
    rewriter.replaceOp(ge_op, {input});
    return success();
  }
};

struct RemoveRedundantGetLength : public OpRewritePattern<GetLengthOp> {
  using OpRewritePattern<GetLengthOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetLengthOp gl_op,
                                PatternRewriter &rewriter) const override {
    auto preceding_build_list = llvm::dyn_cast_or_null<BuildListOp>(
        gl_op.tensor_list().getDefiningOp());
    if (!preceding_build_list) {
      return failure();
    }
    int64_t num_tensors = preceding_build_list.getNumOperands();
    rewriter.replaceOpWithNewOp<ConstantOp>(gl_op,
                                            rewriter.getIndexAttr(num_tensors));
    return success();
  }
};

struct BuildConstantListAsAttr : public OpRewritePattern<BuildListOp> {
  using OpRewritePattern<BuildListOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BuildListOp bl_op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Attribute, 4> array_list;
    array_list.reserve(bl_op.getNumOperands());
    for (const auto &operand : bl_op.getOperands()) {
      Attribute array_elt;
      if (!matchPattern(operand, m_Constant(&array_elt))) {
        return failure();
      }
      array_list.push_back(array_elt);
    }
    auto array_attr = rewriter.getArrayAttr(array_list);
    rewriter.replaceOpWithNewOp<TFR::ConstOp>(bl_op, array_attr);
    return success();
  }
};

quant::QuantizedType getQuantizedElementType(CastOp cast_op) {
  if (!cast_op || !cast_op.getInputElementType()) {
    return {};
  }
  return cast_op.getInputElementType()
      .cast<TypeAttr>()
      .getValue()
      .dyn_cast<quant::QuantizedType>();
}

struct RemoveRawDataOp : public OpRewritePattern<TFRQuantRawDataOp> {
  using OpRewritePattern<TFRQuantRawDataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFRQuantRawDataOp raw_data_op,
                                PatternRewriter &rewriter) const override {
    auto preceding_cast = dyn_cast<CastOp>(raw_data_op.input().getDefiningOp());
    if (!getQuantizedElementType(preceding_cast)) {
      return failure();
    }
    // If there are redundant casts, hoist output of raw data op originating op.
    if (auto redundant_cast = preceding_cast.arg().getDefiningOp()) {
      if (!isa<CastOp>(redundant_cast) ||
          cast<CastOp>(redundant_cast).arg().getType() !=
              preceding_cast.out().getType()) {
        return failure();
      }
      raw_data_op.output().replaceAllUsesWith(
          cast<CastOp>(redundant_cast).arg());
    } else {
      // If the argument of cast op is input, then simply remove the RawData op.
      raw_data_op.output().replaceAllUsesWith(preceding_cast.out());
    }
    return success();
  }
};

struct RemoveQParamsOp : public OpRewritePattern<TFRQuantQParamsOp> {
  using OpRewritePattern<TFRQuantQParamsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFRQuantQParamsOp qparams_op,
                                PatternRewriter &rewriter) const override {
    auto cast_op = dyn_cast<TFR::CastOp>(qparams_op.input().getDefiningOp());
    auto cast_qtype = getQuantizedElementType(cast_op);
    if (!cast_qtype) {
      return failure();
    }
    // Reads quantization parameters from the quantized type, and converts
    // them to constants.
    rewriter.setInsertionPoint(qparams_op);
    if (auto qtype = cast_qtype.dyn_cast<quant::UniformQuantizedType>()) {
      auto scale = rewriter.create<ConstantOp>(
          qparams_op->getLoc(), rewriter.getF32FloatAttr(qtype.getScale()));
      auto scale_tensor = rewriter.create<TFR::ConstantTensorOp>(
          qparams_op->getLoc(), qparams_op.scale().getType(),
          scale.getResult());
      auto zp = rewriter.create<ConstantOp>(
          qparams_op->getLoc(),
          rewriter.getI32IntegerAttr(qtype.getZeroPoint()));
      auto zp_tensor = rewriter.create<TFR::ConstantTensorOp>(
          qparams_op->getLoc(), qparams_op.zp().getType(), zp.getResult());
      qparams_op.scale().replaceAllUsesWith(scale_tensor.getResult());
      qparams_op.zp().replaceAllUsesWith(zp_tensor.getResult());
    } else {
      // TODO(b/192720615): Add handling of UniformQuantizedPerAxisType
      return failure();
    }
    return success();
  }
};

// TODO(b/193731721): Migrate tfr_ builtin canonicalizations to LowerTFROpPass
struct RemoveScaleFactorOp : public OpRewritePattern<TFRQuantScaleFactorOp> {
  using OpRewritePattern<TFRQuantScaleFactorOp>::OpRewritePattern;

  // Replace quant_scale_factor with constant tensor equivalent to
  // TFR_ConstantTensorOp (
  //  ConstantOp (ConstAttr<F32Attr (in_scale[0] * in_scale [1] / filter_scale))
  // )
  LogicalResult matchAndRewrite(TFRQuantScaleFactorOp scale_factor_op,
                                PatternRewriter &rewriter) const override {
    auto out_scale_op = scale_factor_op.out_scale().getDefiningOp<ConstantOp>();
    if (!out_scale_op) {
      return failure();
    }
    auto in_scales_op =
        scale_factor_op.in_scales().getDefiningOp<BuildListOp>();
    if (!in_scales_op || in_scales_op.getNumOperands() != 2) {
      // BuildListOp is variadic, but we require two values: input_scale
      // and filter_scale.
      return failure();
    }
    auto in_scale_tensor =
        in_scales_op.getOperand(0).getDefiningOp<ConstantTensorOp>();
    if (!in_scale_tensor) {
      return failure();
    }
    auto in_scale_op = in_scale_tensor.arg().getDefiningOp<ConstantOp>();
    if (!in_scale_op) {
      return failure();
    }
    const double out_scale =
        out_scale_op.value().cast<FloatAttr>().getValueAsDouble();
    const double in_scale =
        in_scale_op.value().cast<FloatAttr>().getValueAsDouble();
    const Location loc = scale_factor_op->getLoc();
    const Type out_type = scale_factor_op.scale_factor().getType();
    ConstantOp const_op;
    if (auto filter_scale_tensor_op =
            in_scales_op.getOperand(1).getDefiningOp<ConstantTensorOp>()) {
      auto filter_scale_op =
          filter_scale_tensor_op.arg().getDefiningOp<ConstantOp>();
      if (!filter_scale_op) {
        return failure();
      }
      const double filter_scale =
          filter_scale_op.value().cast<FloatAttr>().getValueAsDouble();
      const_op = rewriter.create<ConstantOp>(
          loc, rewriter.getF32FloatAttr(in_scale * filter_scale / out_scale));
    } else if (auto perchannel_tensor_op =
                   in_scales_op.getOperand(1).getDefiningOp<CastOp>()) {
      auto perchannel_op =
          perchannel_tensor_op.arg().getDefiningOp<ConstantOp>();
      if (!perchannel_op) {
        return failure();
      }
      auto filter_scales_attr = perchannel_op.value().cast<DenseElementsAttr>();
      if (!filter_scales_attr) {
        return failure();
      }
      SmallVector<float> scale_factors;
      scale_factors.reserve(filter_scales_attr.size());
      for (auto value : filter_scales_attr.getFloatValues()) {
        scale_factors.push_back(in_scale * value.convertToFloat() / out_scale);
      }
      const_op = rewriter.create<ConstantOp>(
          loc, rewriter.getF32ArrayAttr(scale_factors));
    }
    if (!const_op) {
      return failure();
    }
    auto const_tensor_op =
        rewriter.create<ConstantTensorOp>(loc, out_type, const_op.getResult());
    scale_factor_op.scale_factor().replaceAllUsesWith(const_tensor_op.out());
    return success();
  }
};

struct RemoveRescaleOp : public OpRewritePattern<TFRQuantRescaleOp> {
  using OpRewritePattern<TFRQuantRescaleOp>::OpRewritePattern;

  // Replace quant_rescale (input, scale, zp) with
  // tf.Cast(tf.Round(tf.Cast(input, f32) * scale) + tf.Cast(zp, f32), i32)
  LogicalResult matchAndRewrite(TFRQuantRescaleOp rescale_op,
                                PatternRewriter &rewriter) const override {
    Value input = rescale_op.input();
    Value scale = rescale_op.scale();
    Value zp = rescale_op.zp();
    const Location loc = rescale_op->getLoc();
    const auto result_types = rescale_op->getResultTypes();
    auto c_false =
        rewriter.create<ConstantOp>(loc, rewriter.getBoolAttr(false));
    TypeAttr f32_attr = TypeAttr::get(rewriter.getF32Type());
    TFRAttrType output_type = TFRAttrType::get(rewriter.getContext());
    auto constant_f32_op = rewriter.create<ConstOp>(loc, output_type, f32_attr);
    TypeAttr i32_attr = TypeAttr::get(rewriter.getI32Type());
    auto constant_i32_op = rewriter.create<ConstOp>(loc, output_type, i32_attr);
    auto cast_input_to_float_op = rewriter.create<CallOp>(
        loc, result_types, rewriter.getSymbolRefAttr("tf__cast"),
        ArrayRef<Value>{input, constant_f32_op, c_false});
    auto input_x_scale_op = rewriter.create<CallOp>(
        loc, result_types, rewriter.getSymbolRefAttr("tf__mul"),
        ArrayRef<Value>{cast_input_to_float_op.getResult(0), scale});
    auto round_rescaled_op = rewriter.create<CallOp>(
        loc, result_types, rewriter.getSymbolRefAttr("tf__round"),
        ArrayRef<Value>{input_x_scale_op->getResult(0)});
    auto cast_zp_to_float_op = rewriter.create<CallOp>(
        loc, result_types, rewriter.getSymbolRefAttr("tf__cast"),
        ArrayRef<Value>{zp, constant_f32_op, c_false});
    auto recentered_op = rewriter.create<CallOp>(
        loc, result_types, rewriter.getSymbolRefAttr("tf__add"),
        ArrayRef<Value>{round_rescaled_op->getResult(0),
                        cast_zp_to_float_op->getResult(0)});
    auto cast_output_to_i32 = rewriter.create<CallOp>(
        loc, result_types, rewriter.getSymbolRefAttr("tf__cast"),
        ArrayRef<Value>{recentered_op->getResult(0), constant_i32_op, c_false});
    rescale_op.output().replaceAllUsesWith(cast_output_to_i32.getResult(0));
    return success();
  }
};

void ConstantTensorOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ConvertConstToTensorConst>(context);
}

void CastOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<RemoveRedundantCast>(context);
}

void GetShapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<GetTensorShape>(context);
}

void GetElementOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<RemoveRedundantGetElement>(context);
}

void GetLengthOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<RemoveRedundantGetLength>(context);
}

void BuildListOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<BuildConstantListAsAttr>(context);
}

void TFRQuantRawDataOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<RemoveRawDataOp>(context);
}

void TFRQuantQParamsOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<RemoveQParamsOp>(context);
}

void TFRQuantRescaleOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<RemoveRescaleOp>(context);
}

void TFRQuantScaleFactorOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<RemoveScaleFactorOp>(context);
}

OpFoldResult TFR::EqualOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "equal op has two operands");
  auto ctx = getContext();
  if (operands[0] == operands[1]) return BoolAttr::get(ctx, true);
  return BoolAttr::get(ctx, false);
}

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

// CallableOpInterface
Region *TFRFuncOp::getCallableRegion() {
  return isExternal() ? nullptr : &body().front();
}

// CallableOpInterface
ArrayRef<Type> TFRFuncOp::getCallableResults() {
  return getType().getResults();
}

//===----------------------------------------------------------------------===//
// Dialect type definitions
//===----------------------------------------------------------------------===//

// Parses a TFR type.
//   tfr_type ::= tensor_type | tensor_list_type | attr_type
//   string_list ::= `[` string-literal (, string-literal)+ `]`
//   tensor_type ::= `tensor`
//                 | `tensor<` (string-literal | string_list)  '>'
//   tensor_list_type ::= `tensor_list`
//                      | `tensor_list<` (string-literal | string_list)  '>'
//   attr_type ::= `attr`
Type TFRDialect::parseType(DialectAsmParser &parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  MLIRContext *ctx = loc.getContext();

  StringRef typeNameSpelling;
  if (failed(parser.parseKeyword(&typeNameSpelling))) return {};
  llvm::SmallVector<StringAttr, 4> attrs;
  if (succeeded(parser.parseOptionalLess())) {
    bool l_square_parsed = false;
    if (succeeded(parser.parseOptionalLSquare())) {
      l_square_parsed = true;
    }

    do {
      StringRef attr;
      if (failed(parser.parseKeyword(&attr))) return {};
      attrs.push_back(StringAttr::get(ctx, attr));
    } while (succeeded(parser.parseOptionalComma()));

    if (l_square_parsed && failed(parser.parseRSquare())) {
      parser.emitError(parser.getNameLoc(), "expected ']'");
    }

    if (failed(parser.parseGreater())) {
      parser.emitError(parser.getNameLoc(), "expected '>'");
    }
  }

  if (typeNameSpelling == "tensor") {
    return TFRTensorType::getChecked(attrs, loc);
  } else if (typeNameSpelling == "tensor_list") {
    return TFRTensorListType::getChecked(attrs, loc);
  } else if (typeNameSpelling == "attr") {
    return TFRAttrType::getChecked(loc, loc.getContext());
  } else {
    parser.emitError(parser.getNameLoc(), "unknown type " + typeNameSpelling);
    return {};
  }
}

void TFRDialect::printType(Type type, DialectAsmPrinter &os) const {
  llvm::ArrayRef<StringAttr> attrs;

  if (type.isa<TFRAttrType>()) {
    os << "attr";
    return;
  }
  if (auto tensor_ty = type.dyn_cast<TFRTensorType>()) {
    attrs = tensor_ty.getAttrKeys();
    os << "tensor";
  } else if (auto tensor_list_ty = type.dyn_cast<TFRTensorListType>()) {
    attrs = tensor_list_ty.getAttrKeys();
    os << "tensor_list";
  } else {
    llvm_unreachable("Unhandled tfr type");
  }

  if (attrs.empty()) return;
  os << "<";

  if (attrs.size() > 1) {
    os << "[";
  }

  llvm::interleaveComma(attrs, os,
                        [&](StringAttr attr) { os << attr.getValue(); });

  if (attrs.size() > 1) {
    os << "]";
  }
  os << ">";
}

}  // namespace TFR
}  // namespace mlir
