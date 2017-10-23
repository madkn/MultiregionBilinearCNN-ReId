// This code is written by Yaroslav Ganin (http://yaroslav.ganin.net/)
#include <vector>

#include "caffe/layers/bilinear_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define MAX_STREAMS 16

namespace caffe {

template <typename Dtype>
void BilinearLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  streams_.resize(MAX_STREAMS);
  for (int s = 0; s < streams_.size(); ++s) {
    CUDA_CHECK(cudaStreamCreate(&streams_[s]));
  }

  // Setup internal im2col layers.
  im2col_top_vec_.clear();
  im2col_top_vec_.push_back(&bottom_input_a_cols_);
  im2col_top_vec_.push_back(&bottom_input_b_cols_);

  im2col_layers_.resize(2);
  for (int input_idx = 0; input_idx < 2; ++input_idx) {
    im2col_temp_bottom_vec_.resize(1);
    im2col_temp_bottom_vec_[0] = bottom[input_idx];
    im2col_temp_top_vec_.resize(1);
    im2col_temp_top_vec_[0] = im2col_top_vec_[input_idx];
    im2col_layers_[input_idx].reset(new Im2colLayer<Dtype>(this->layer_param_));
    im2col_layers_[input_idx]->SetUp(im2col_temp_bottom_vec_, 
                                     im2col_temp_top_vec_);
  }

  // Setup first transpose op.
  geam_top_vec_.clear();
  geam_top_vec_.push_back(&geam_top_a_);
  geam_top_vec_.push_back(&geam_top_b_);

  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    kernel_count_ = conv_param.kernel_h() * conv_param.kernel_w();
  } else {
    kernel_count_ = conv_param.kernel_size(0) * conv_param.kernel_size(0);
  }
  im2col_propagate_down_.assign(1, true);
}

template <typename Dtype>
void BilinearLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->shape(0);
  channels_a_ = bottom[0]->shape(1);
  channels_b_ = bottom[1]->shape(1);

  // std::cout << channels_a_ << std::endl << channels_b_ << std::endl;

  for (int input_idx = 0; input_idx < 2; ++input_idx) {
    im2col_temp_bottom_vec_[0] = bottom[input_idx];
    im2col_temp_top_vec_[0] = im2col_top_vec_[input_idx];
    im2col_layers_[input_idx]->Reshape(im2col_temp_bottom_vec_, 
                                       im2col_temp_top_vec_);
  }

  int top_h = im2col_top_vec_[0]->shape(2);
  int top_w = im2col_top_vec_[0]->shape(3);
  num_patches_per_image_ = top_h * top_w;

  // std::cout << top_h << std::endl << top_w << std::endl;
  // std::cout << geam_top_vec_.size() << std::endl;

  for (int input_idx = 0; input_idx < 2; ++input_idx) {
    const int num_rows = im2col_top_vec_[input_idx]->count(0, 2);
    // std::cout << num_rows << std::endl;
    vector<int> geam_top_shape(2);
    geam_top_shape[0] = num_patches_per_image_;
    geam_top_shape[1] = num_rows;
    geam_top_vec_[input_idx]->Reshape(geam_top_shape);
  }

  vector<int> gemm_top_shape(2);
  gemm_top_shape[0] = num_patches_per_image_;
  gemm_top_shape[1] = num_ * channels_a_ * channels_b_;
  gemm_top_.Reshape(gemm_top_shape);

  vector<int> top_shape(4);
  top_shape[0] = num_;
  top_shape[1] = channels_a_ * channels_b_;
  top_shape[2] = top_h;
  top_shape[3] = top_w;
  top[0]->Reshape(top_shape);
}



template <typename Dtype>
void BilinearLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void BilinearLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
BilinearLayer<Dtype>::~BilinearLayer() {
  for (int s = 0; s < MAX_STREAMS; ++s) {
    cudaStreamDestroy(streams_[s]);
  }
}

#ifdef CPU_ONLY
STUB_GPU(BilinearLayer);
#endif

INSTANTIATE_CLASS(BilinearLayer);
REGISTER_LAYER_CLASS(Bilinear);

}  // namespace caffe
