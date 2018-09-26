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
 * \file mkldnn_slice-inl.h
 * \brief
 * \author 
*/
#include "../../tensor/matrix_op-inl.h"
#include "./mkldnn_base-inl.h"
#include "./mkldnn_ops-inl.h"
#include "../../slice_channel-inl.h"


#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {


void MKLDNNSlice(const SliceParam &param, const OpContext& ctx,
                 const NDArray &in, OpReqType req, const NDArray &out);

void MKLDNNSliceAxis(const SliceAxisParam &param, const OpContext &ctx,
                     const NDArray &in, OpReqType req, const NDArray &out);

void MKLDNNSliceLike(const SliceLikeParam &param, const OpContext &ctx,
                     const NDArray &in, OpReqType req, const NDArray &out);

void MKLDNNSliceChannel(const SliceChannelParam &param, const OpContext& ctx,
                        const NDArray &in, OpReqType req, const std::vector<NDArray>& out);

}  // namespace op
}  // namespace mxnet
#endif
