/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file slice_channel.cc
 * \brief
 * \author Bing Xu
*/

#include "./elemwise_op_common.h"
#include "./nn/mkldnn/mkldnn_slice-inl.h"

namespace mxnet {
namespace op {
// template<>
// Operator* CreateOp<cpu>(SliceChannelParam param, int dtype) {
//   Operator* op = nullptr;
//   MSHADOW_TYPE_SWITCH(dtype, DType, {
//     op = new SliceChannelOp<cpu, DType>(param);
//   })
//   return op;
// }

// Operator* SliceChannelProp::CreateOperatorEx(Context ctx,
//                                              std::vector<TShape>* in_shape,
//                                              std::vector<int>* in_type) const {
//   DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
// }

template<typename xpu>
void SliceChannelEx(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<NDArray>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<NDArray>& outputs) {
  const SliceChannelParam& param = nnvm::get<SliceChannelParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), static_cast<size_t>(param.num_outputs));
  auto in_stype = inputs[0].storage_type();

  if (in_stype == kDefaultStorage) {
    MKLDNNSliceChannel(param, ctx, inputs[0], req[0], outputs);
  } else {
    LOG(FATAL) << "MKLDNN SliceChannel(split) is not implemented for this storage type" << in_stype;
  }
}

DMLC_REGISTER_PARAMETER(SliceChannelParam);

NNVM_REGISTER_OP(SliceChannel)
MXNET_ADD_SPARSE_OP_ALIAS(SliceChannel)
.add_alias("split")
.describe(R"code(Splits an array along a particular axis into multiple sub-arrays.

.. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.

**Note** that `num_outputs` should evenly divide the length of the axis
along which to split the array.

Example::

   x  = [[[ 1.]
          [ 2.]]
         [[ 3.]
          [ 4.]]
         [[ 5.]
          [ 6.]]]
   x.shape = (3, 2, 1)

   y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
   y = [[[ 1.]]
        [[ 3.]]
        [[ 5.]]]

       [[[ 2.]]
        [[ 4.]]
        [[ 6.]]]

   y[0].shape = (3, 1, 1)

   z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
   z = [[[ 1.]
         [ 2.]]]

       [[[ 3.]
         [ 4.]]]

       [[[ 5.]
         [ 6.]]]

   z[0].shape = (1, 2, 1)

`squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.
**Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
along the `axis` which it is split.
Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.

Example::

   z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
   z = [[ 1.]
        [ 2.]]

       [[ 3.]
        [ 4.]]

       [[ 5.]
        [ 6.]]
   z[0].shape = (2 ,1 )

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SliceChannelParam>)
.set_num_inputs(1)
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const SliceChannelParam& param = nnvm::get<SliceChannelParam>(attrs.parsed);
    return param.num_outputs;
  })
.set_attr<nnvm::FInferShape>("FInferShape", SliceChannelOpShape)
.set_attr<nnvm::FInferType>("FInferType", SliceChannelInferType)
.set_attr<FInferStorageType>("FInferStorageType", SliceChannelForwardInferStorageType)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_slice_channel"})
.set_attr<FCompute>("FCompute<cpu>", SliceChannelForward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SliceChannelEx<cpu>)
//.set_return_type("NDArray-or-Symbol[]")
.add_argument("data", "NDArray-or-Symbol", "The input")
.add_arguments(SliceChannelParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_slice_channel)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const SliceChannelParam& param = nnvm::get<SliceChannelParam>(attrs.parsed);
    return param.num_outputs;
  })
.set_num_outputs(1)
.set_attr_parser(ParamParser<SliceChannelParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SliceChannelBackward<cpu>);

}  // namespace op
}  // namespace mxnet
