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
 * \author 
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
  const uint32_t axis = param.axis;
  const uint32_t num_outputs = param.num_outputs;
  // We need to figure out whether it is actually "squeezing"
  int squeeze_axis;
  if (ishape.ndim() == oshape.ndim()) {
      squeeze_axis = 0;
  } else {
      squeeze_axis = 1;
  }
  
  MKLDNNStream *stream = MKLDNNStream::Get();
  
  // In total we have num_outputs
  // For each output, we need to set up dims and offsets in order to use view_pd
  if (squeeze_axis == 1) { // For the case squeeze_axis == 1
    mkldnn::memory::dims dims(N+1), offsets(N+1);
    for (uint32_t i = 0; i < num_outputs; i++) {
        // Set up dims and offsets
        for (uint32_t j = 0; j < N+1; j++) {
            if (j == axis) {
                dims[j] = 1;
                offsets[j] = i;
            } else if (j < axis){
                dims[j] = oshape[j];
                offsets[j] = 0;
            } else { //if j > axis
                dims[j] = oshape[j-1];
                offsets[j] = 0;
            }
        }
        auto in_mem = in.GetMKLDNNData();
        auto in_mem_pd = in_mem->get_primitive_desc();
        auto out_mem_pd = out[i].GetMKLDNNData()->get_primitive_desc();
        auto out_mem = CreateMKLDNNMem(out[i], out_mem_pd, req);
        auto temp_dtype = static_cast<mkldnn::memory::data_type>(in_mem_pd.desc().data.data_type);
        auto temp_format = static_cast<mkldnn::memory::format>(GetDefaultFormat(in_mem_pd.desc()));
        mkldnn::memory::desc temp_md(dims, temp_dtype, temp_format);
        mkldnn::memory::primitive_desc temp_pd(temp_md, in_mem_pd.get_engine());

        std::shared_ptr<mkldnn::view::primitive_desc> view_pd;
        view_pd.reset(new mkldnn::view::primitive_desc(in_mem_pd, dims, offsets));
        // 1. Create reorder_pd based on view_pd and temp_mem_pd
        // 2. MKLDNNCopy from temp_mem_pd to out_mem_pd
        auto reorder_pd = reorder::primitive_desc(view_pd.get()->dst_primitive_desc(), temp_pd);
        mkldnn_mem_ptr temp_mem(new mkldnn::memory(temp_pd, in_mem->get_data_handle()));
        stream->RegisterMem(temp_mem);
        stream->RegisterPrim(mkldnn::reorder(reorder_pd, *in_mem, *temp_mem));
        stream->RegisterPrim(mkldnn::reorder(*temp_mem, *out_mem));
        
        //auto reorder_pd = reorder::primitive_desc(view_pd.get()->dst_primitive_desc(), out_mem_pd);

        CommitOutput(out[i], out_mem);
        // out[i] = out[i].CreateView(oshape, out[i].getDataType());
    }
  } else { // For the case squeeze_axis == 0
    mkldnn::memory::dims dims(N), offsets(N);
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
  }


  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif