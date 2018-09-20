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
 * \file mkldnn_slice.cc
 * \brief
 * \author Mengjie Li
*/
#include "./mkldnn_slice-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {


void MKLDNNSlice(const SliceParam &param, const OpContext& ctx,
                  const NDArray &in, OpReqType req, const NDArray &out) {
  const TShape ishape = in.shape();
  const TShape oshape = out.shape();
  uint32_t N = ishape.ndim();
  TShape begin(N), end(N); // Here begin also functions as offsets for view pd
  for (uint32_t i = 0; i < N; ++i) {
    int s = 0;
    if (param.begin[i]) {
      s = *param.begin[i];
      if (s < 0) s += ishape[i];
    }
    begin[i] = s;
    end[i] = s + oshape[i];
  }

  // convert TShape to mkldnn::memory::dims
  mkldnn::memory::dims dims(N);
  mkldnn::memory::dims offsets(N);
  for (size_t i = 0; i < N; i++) {
    dims[i] = oshape[i];
    offsets[i] = begin[i];
  }
  auto in_mem = in.GetMKLDNNData();
  auto in_mem_pd = in_mem->get_primitive_desc();

  std::shared_ptr<mkldnn::view::primitive_desc> view_pd;
  view_pd.reset(new mkldnn::view::primitive_desc(in_mem_pd, dims, offsets));
  auto out_mem_pd = out.GetMKLDNNData()->get_primitive_desc();
  auto out_mem = CreateMKLDNNMem(out, out_mem_pd, req);
  auto reorder_pd = reorder::primitive_desc(view_pd.get()->dst_primitive_desc(), out_mem_pd);

  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(mkldnn::reorder(reorder_pd, *in_mem, *out_mem.second));
  CommitOutput(out, out_mem);
  stream->Submit();
}

void MKLDNNSliceAxis(const SliceAxisParam &param, const OpContext &ctx,
                  const NDArray &in, OpReqType req, const NDArray &out) {
  const TShape ishape = in.shape();
  const TShape oshape = out.shape();
  int axis, begin, end;
  GetSliceAxisParams(param, in.shape(), &axis, &begin, &end);
   
  uint32_t N = ishape.ndim();
  mkldnn::memory::dims dims(N);
  mkldnn::memory::dims offsets(N);
  for (size_t i = 0; i < N; i++) {
    dims[i] = oshape[i];
    if (i == uint32_t(axis))  offsets[i] = begin;
    else  offsets[i] = 0;
  }
  auto in_mem = in.GetMKLDNNData();
  auto in_mem_pd = in_mem->get_primitive_desc();

  std::shared_ptr<mkldnn::view::primitive_desc> view_pd;
  view_pd.reset(new mkldnn::view::primitive_desc(in_mem_pd, dims, offsets));
  auto out_mem_pd = out.GetMKLDNNData()->get_primitive_desc();
  auto out_mem = CreateMKLDNNMem(out, out_mem_pd, req);
  auto reorder_pd = reorder::primitive_desc(view_pd.get()->dst_primitive_desc(), out_mem_pd);

  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(mkldnn::reorder(reorder_pd, *in_mem, *out_mem.second));
  CommitOutput(out, out_mem);
  stream->Submit();
}

void MKLDNNSliceLike(const SliceLikeParam &param, const OpContext &ctx,
                  const NDArray &in, OpReqType req, const NDArray &out) {
  const TShape ishape = in.shape();
  const TShape oshape = out.shape();

  mkldnn::memory::dims dims(oshape.ndim());
  mkldnn::memory::dims offsets(oshape.ndim());
  for (size_t i = 0; i < dims.size(); i++) {
    dims[i] = oshape[i];
    offsets[i] = 0;
  }

  auto in_mem = in.GetMKLDNNData();
  auto in_mem_pd = in_mem->get_primitive_desc();

  std::shared_ptr<mkldnn::view::primitive_desc> view_pd;
  view_pd.reset(new mkldnn::view::primitive_desc(in_mem_pd, dims, offsets));

  auto out_mem_pd = out.GetMKLDNNData()->get_primitive_desc();
  auto out_mem = CreateMKLDNNMem(out, out_mem_pd, req);
  mkldnn::reorder::primitive_desc reorder_pd(view_pd.get()->dst_primitive_desc(), out_mem_pd);

  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(mkldnn::reorder(reorder_pd, *in_mem, *out_mem.second));
  CommitOutput(out, out_mem);
  stream->Submit();
}

void MKLDNNSliceChannel(const SliceChannelParam &param, const OpContext& ctx,
                        const NDArray &in, OpReqType req, const std::vector<NDArray>& out) {
  const TShape ishape = in.shape();
  const TShape oshape = out[0].shape();
  uint32_t N = oshape.ndim();
  // axis, num_outputs, squeeze_axis
  const int axis = param.axis;
  const int num_outputs = param.num_outputs;
  const int squeeze_axis = param.squeeze_axis;
  
  mkldnn::memory::dims dims(N), offsets(N);
  MKLDNNStream *stream = MKLDNNStream::Get();
  // In total we have num_outputs
  // For each output, we need to set up dims and offsets in order to use view_pd
  for (uint32_t i = 0; i < num_outputs; i++) {
    // Set up dims and offsets
    for (uint32_t j = 0; j < N; j++) {
      dims[j] = oshape[j];
      if (j == axis)  offsets[j] = oshape[j] * i;
      else  offsets[j] = 0;
    }
    auto in_mem = in.GetMKLDNNData();
    auto in_mem_pd = in_mem->get_primitive_desc();

    std::shared_ptr<mkldnn::view::primitive_desc> view_pd;
    view_pd.reset(new mkldnn::view::primitive_desc(in_mem_pd, dims, offsets));
    auto out_mem_pd = out[i].GetMKLDNNData()->get_primitive_desc();
    auto out_mem = CreateMKLDNNMem(out[i], out_mem_pd, req);
    auto reorder_pd = reorder::primitive_desc(view_pd.get()->dst_primitive_desc(), out_mem_pd);

    stream->RegisterPrim(mkldnn::reorder(reorder_pd, *in_mem, *out_mem.second));
    CommitOutput(out[i], out_mem);
  }

  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif