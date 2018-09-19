#include "../../tensor/matrix_op-inl.h"
#include "./mkldnn_base-inl.h"
#include "./mkldnn_ops-inl.h"


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

}  // namespace op
}  // namespace mxnet
#endif
