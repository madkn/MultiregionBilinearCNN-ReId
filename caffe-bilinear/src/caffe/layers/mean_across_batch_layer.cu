#include <cfloat>
#include <vector>

#include "caffe/layers/mean_across_batch_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



template <typename Dtype>
void MeanAcrossBatchLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  int chunk_size = this->layer_param().mean_across_batch_param().chunk_size();
  int count = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
  
  for (int i = 0; i < (bottom[0]->num() + chunk_size)/chunk_size; i++){
      int s = i * chunk_size;
      int e = std::min((i + 1) * chunk_size, bottom[0]->num());
      
      if (s >= e)
	  break;

      caffe_gpu_set(count, Dtype(0.), mean_in_chunk_.mutable_gpu_data());
      for (int j = s; j < e; j++){
          caffe_gpu_axpy(count, Dtype(1./(e-s)), bottom[0]->gpu_data() + count * j, mean_in_chunk_.mutable_gpu_data());

      }
      for (int j = s; j < e; j++){
          caffe_copy(count, mean_in_chunk_.gpu_data(), top[0]->mutable_gpu_data() + count * j);
      }

  }


}

template <typename Dtype>
void MeanAcrossBatchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int chunk_size = this->layer_param().mean_across_batch_param().chunk_size();
  int count = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width(); 
  
  for (int i = 0; i < (top[0]->num() + chunk_size)/chunk_size; i++){
      int s = i * chunk_size;
      int e = std::min((i + 1) * chunk_size, bottom[0]->num());
      
      if (s >= e)
	  break;

      caffe_gpu_set(count, Dtype(0.), mean_in_chunk_.mutable_gpu_diff());
      for (int j = s; j < e; j++){
          caffe_gpu_axpy(count, Dtype(1./(e-s)), top[0]->gpu_diff() + count * j, mean_in_chunk_.mutable_gpu_diff());
      }
      for (int j = s; j < e; j++){
          caffe_copy(count, mean_in_chunk_.gpu_diff(), bottom[0]->mutable_gpu_diff() + count * j);
      }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MeanAcrossBatchLayer);


}  // namespace caffe
