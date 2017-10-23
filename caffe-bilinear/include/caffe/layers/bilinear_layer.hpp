#ifndef CAFFE_BILINEAR_LAYER_HPP_
#define CAFFE_BILINEAR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/im2col_layer.hpp"
namespace caffe {


/**
 * @brief Applies bilinear transformation to a pair of tensors.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BilinearLayer : public Layer<Dtype> {
 public:
  explicit BilinearLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~BilinearLayer();

  virtual inline const char* type() const { return "Bilinear"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<cudaStream_t> streams_;

  vector<shared_ptr<Im2colLayer<Dtype> > > im2col_layers_;
  vector<Blob<Dtype>*> im2col_temp_bottom_vec_;
  vector<Blob<Dtype>*> im2col_temp_top_vec_;
  vector<Blob<Dtype>*> im2col_top_vec_;
  Blob<Dtype> bottom_input_a_cols_;
  Blob<Dtype> bottom_input_b_cols_;
  vector<Blob<Dtype>*> geam_top_vec_;
  Blob<Dtype> geam_top_a_;
  Blob<Dtype> geam_top_b_;
  Blob<Dtype> gemm_top_;

  int num_;
  int channels_a_;
  int channels_b_;

  int kernel_count_;
  int num_patches_per_image_;

  vector<bool> im2col_propagate_down_;
};



}  // namespace caffe

#endif  // CAFFE_BILINEAR_LAYER_HPP_


