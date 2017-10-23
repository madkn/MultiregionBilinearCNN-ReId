#include <cfloat>
#include <vector>

#include "caffe/layers/mean_across_batch_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MeanAcrossBatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  CHECK(bottom.size() == 1);
  CHECK(this->layer_param().mean_across_batch_param().chunk_size() > 0) << "Chunk size for mean calculation should be greater than zero!";

  mean_in_chunk_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void MeanAcrossBatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 // top[0]->ReshapeLike(*bottom[0]);
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),  bottom[0]->height(),  bottom[0]->width());
}

template <typename Dtype>
void MeanAcrossBatchLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  int chunk_size = this->layer_param().mean_across_batch_param().chunk_size();
  int count = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
  
  for (int i = 0; i < (bottom[0]->num() + chunk_size)/chunk_size; i++){
      int s = i * chunk_size;
      int e = std::min((i + 1) * chunk_size, bottom[0]->num());
      
      if (s >= e)
	  break;

      caffe_set(count, Dtype(0.), mean_in_chunk_.mutable_cpu_data());
      for (int j = s; j < e; j++){
          caffe_axpy(count, Dtype(1./(e-s)), bottom[0]->cpu_data() + count * j, mean_in_chunk_.mutable_cpu_data());

      }
      for (int j = s; j < e; j++){
          caffe_copy(count, mean_in_chunk_.cpu_data(), top[0]->mutable_cpu_data() + count * j);
      }

  }


}

template <typename Dtype>
void MeanAcrossBatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int chunk_size = this->layer_param().mean_across_batch_param().chunk_size();
  int count = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width(); 
  
  for (int i = 0; i < (top[0]->num() + chunk_size)/chunk_size; i++){
      int s = i * chunk_size;
      int e = std::min((i + 1) * chunk_size, bottom[0]->num());
      
      if (s >= e)
	  break;

      caffe_set(count, Dtype(0.), mean_in_chunk_.mutable_cpu_diff());
      for (int j = s; j < e; j++){
          caffe_axpy(count, Dtype(1./(e-s)), top[0]->cpu_diff() + count * j, mean_in_chunk_.mutable_cpu_diff());
      }
      for (int j = s; j < e; j++){
          caffe_copy(count, mean_in_chunk_.cpu_diff(), bottom[0]->mutable_cpu_diff() + count * j);
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MeanAcrossBatchLayer);
#endif

INSTANTIATE_CLASS(MeanAcrossBatchLayer);
REGISTER_LAYER_CLASS(MeanAcrossBatch);

}  // namespace caffe
