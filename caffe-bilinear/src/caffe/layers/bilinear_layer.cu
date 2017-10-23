// This code is written by Yaroslav Ganin (http://yaroslav.ganin.net/)

#include <vector>

#include "caffe/layers/bilinear_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

__global__ void sync_patches() { }

template <typename Dtype>
void BilinearLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int input_idx = 0; input_idx < 2; ++input_idx) {
    // First, we transform input tensors using im2col in order to localize
    // patches.
    im2col_temp_bottom_vec_[0] = bottom[input_idx];
    im2col_temp_top_vec_[0] = im2col_top_vec_[input_idx];
    im2col_layers_[input_idx]->Forward(im2col_temp_bottom_vec_, 
                                       im2col_temp_top_vec_);

    // Next, we transpose resulting tensors so that patch index dimension 
    // becomes the slowest changing dimension.
    int num_rows = im2col_top_vec_[input_idx]->count() / num_patches_per_image_;
    int num_cols = num_patches_per_image_;
  
    const Dtype* im2col_top_data = im2col_top_vec_[input_idx]->gpu_data();
    Dtype* geam_top_data = geam_top_vec_[input_idx]->mutable_gpu_data();

    caffe_gpu_geam<Dtype>(CblasTrans, CblasTrans, num_cols, num_rows,
        (Dtype) 1., im2col_top_data, im2col_top_data, (Dtype) 0.,
        geam_top_data);
  }

  // We use GEMM to get the bilinear maps for all the patches.
  // Total number of patches is batch_size x num_patches_per_image.
  int num_patches = num_ * num_patches_per_image_;
  int M = channels_a_;
  int N = channels_b_;
  int K = kernel_count_;

  for (int patch_index = 0; patch_index < num_patches; ++patch_index) {
    const Dtype* patch_a_data = geam_top_vec_[0]->gpu_data();
    const Dtype* patch_b_data = geam_top_vec_[1]->gpu_data();
    Dtype* gemm_top_data = gemm_top_.mutable_gpu_data();

    patch_a_data += patch_index * M * K;
    patch_b_data += patch_index * N * K;
    gemm_top_data += patch_index * M * N;

    Caffe::set_cublas_stream(streams_[patch_index % streams_.size()]);

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, N, K, (Dtype) 1.,
                          patch_a_data, patch_b_data, (Dtype) 0., 
                          gemm_top_data);
  }
  // Synchronize the work across patches, each of which went into its own
  // stream, by launching an empty kernel into the default (null) stream.
  // NOLINT_NEXT_LINE(whitespace/operators)
  sync_patches<<<1, 1>>>();

  Caffe::set_cublas_stream(0);

  // Finally, we transpose patch index dimension back.
  int num_rows = num_patches_per_image_;
  int num_cols = top[0]->count() / num_patches_per_image_;

  const Dtype* gemm_top_data = gemm_top_.gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  caffe_gpu_geam<Dtype>(CblasTrans, CblasTrans, num_cols, num_rows,
      (Dtype) 1., gemm_top_data, gemm_top_data, (Dtype) 0.,
      top_data);
}

template <typename Dtype>
void BilinearLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // During backpropagation we reverse the order of operations.

  // (Un)transpose top diffs.
  int num_rows = top[0]->count() / num_patches_per_image_;
  int num_cols = num_patches_per_image_;

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* gemm_top_diff = gemm_top_.mutable_gpu_diff();

  caffe_gpu_geam<Dtype>(CblasTrans, CblasTrans, num_cols, num_rows,
      (Dtype) 1., top_diff, top_diff, (Dtype) 0.,
      gemm_top_diff);

  // GEMM gradients.
  int num_patches = num_ * num_patches_per_image_;
  int M = channels_a_;
  int N = channels_b_;
  int K = kernel_count_;

  // Gradient with respect to the left GEMM operand.
  if (propagate_down[0]) {
    for (int patch_index = 0; patch_index < num_patches; ++patch_index) {
      const Dtype* patch_b_data = geam_top_vec_[1]->gpu_data();
      const Dtype* gemm_top_diff = gemm_top_.gpu_diff();
      Dtype* patch_a_diff = geam_top_vec_[0]->mutable_gpu_diff();

      patch_b_data += patch_index * N * K;
      gemm_top_diff += patch_index * M * N;
      patch_a_diff += patch_index * M * K;

      Caffe::set_cublas_stream(streams_[patch_index % streams_.size()]);

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, K, N, (Dtype) 1.,
                            gemm_top_diff, patch_b_data, (Dtype) 0.,
                            patch_a_diff);
    }
  }

  // Gradient with respect to the right GEMM operand.
  if (propagate_down[1]) {
    for (int patch_index = 0; patch_index < num_patches; ++patch_index) {
      const Dtype* patch_a_data = geam_top_vec_[0]->gpu_data();
      const Dtype* gemm_top_diff = gemm_top_.gpu_diff();
      Dtype* patch_b_diff = geam_top_vec_[1]->mutable_gpu_diff();

      patch_a_data += patch_index * M * K;
      gemm_top_diff += patch_index * M * N;
      patch_b_diff += patch_index * N * K;

      Caffe::set_cublas_stream(streams_[patch_index % streams_.size()]);

      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N, K, M, (Dtype) 1.,
                            gemm_top_diff, patch_a_data, (Dtype) 0.,
                            patch_b_diff);
    }
  }
  // Synchronize the work across patches, each of which went into its own
  // stream, by launching an empty kernel into the default (null) stream.
  // NOLINT_NEXT_LINE(whitespace/operators)
  sync_patches<<<1, 1>>>();

  // Gradients with respect to bottom data.
  for (int input_idx = 0; input_idx < 2; ++input_idx) {
    if (!propagate_down[input_idx]) {
      continue;
    }
    
    // Again, we (un)transpose diffs so that they are suitable for the
    // col2im op.
    int num_rows = num_patches_per_image_;
    int num_cols = im2col_top_vec_[input_idx]->count() / num_patches_per_image_;
  
    const Dtype* geam_top_diff = geam_top_vec_[input_idx]->gpu_diff();
    Dtype* im2col_top_diff = im2col_top_vec_[input_idx]->mutable_gpu_diff();

    caffe_gpu_geam<Dtype>(CblasTrans, CblasTrans, num_cols, num_rows,
        (Dtype) 1., geam_top_diff, geam_top_diff, (Dtype) 0.,
        im2col_top_diff);

    // Finally, we invoke bprop routine of the internal im2col layer.
    im2col_temp_top_vec_[0] = im2col_top_vec_[input_idx];
    im2col_propagate_down_[0] = propagate_down[input_idx];
    im2col_temp_bottom_vec_[0] = bottom[input_idx];
    im2col_layers_[input_idx]->Backward(im2col_temp_top_vec_,
                                        im2col_propagate_down_,
                                        im2col_temp_bottom_vec_);
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(BilinearLayer);

}  // namespace caffe
