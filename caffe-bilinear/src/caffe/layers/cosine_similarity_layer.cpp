#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/cosine_similarity_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {
template <typename Dtype>
void CosineSimilarityLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());

  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
void CosineSimilarityLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {

  summer_vec_.Reshape(bottom[0]->num(), 1, 1, 1);
  xy_.Reshape(bottom[0]->num(), 1, 1, 1);
  xx_.Reshape(bottom[0]->num(), 1, 1, 1);
  yy_.Reshape(bottom[0]->num(), 1, 1, 1);
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
  top[1]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void CosineSimilarityLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


  for (int i = 0; i < bottom[0]->num(); ++i){
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
  }

  int channels = bottom[0]->channels();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    xx_.mutable_cpu_data()[i] = caffe_cpu_dot(bottom[0]->channels(), bottom[0]->cpu_data() + i * channels, 
							       bottom[0]->cpu_data() + i * channels);
    yy_.mutable_cpu_data()[i] = caffe_cpu_dot(bottom[1]->channels(), bottom[1]->cpu_data() + i * channels, 
							       bottom[1]->cpu_data() + i * channels);
    xy_.mutable_cpu_data()[i] = caffe_cpu_dot(bottom[0]->channels(), bottom[0]->cpu_data() + i * channels, 
							       bottom[1]->cpu_data() + i * channels);
  }
  caffe_mul(bottom[1]->num(), xx_.cpu_data(),yy_.cpu_data(), summer_vec_.mutable_cpu_data());

  for (int i = 0; i < bottom[0]->num(); ++i) {
    summer_vec_.mutable_cpu_data()[i] = sqrt(summer_vec_.cpu_data()[i]);
  } 
  caffe_div(bottom[1]->num(), xy_.cpu_data(), summer_vec_.cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void CosineSimilarityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
            Dtype denominator = pow(xx_.cpu_data()[j]*yy_.cpu_data()[j], 0.5);  
            Dtype* bout = bottom[i]->mutable_cpu_diff();
            if (i == 0){
	      caffe_cpu_axpby(
	        channels,
                Dtype(-xy_.cpu_data()[j] /(denominator* xx_.cpu_data()[j])),
                bottom[0]->cpu_data() + (j*channels),
                Dtype(0.0),
                bout + (j*channels));

              caffe_cpu_axpby(
              channels,
              Dtype(1.0/denominator),
              bottom[1]->cpu_data() + (j*channels),
              Dtype(1.0),
              bout + (j*channels));
            } else if (i == 1){
	      caffe_cpu_axpby(
	        channels,
                Dtype(-xy_.cpu_data()[j] /(denominator*yy_.cpu_data()[j])),
	        bottom[1]->cpu_data() + (j*channels),
  	        Dtype(0.0),
	        bout + (j*channels));
              caffe_cpu_axpby(
	        channels,
	        Dtype(1.0/denominator),
	        bottom[0]->cpu_data() + (j*channels),
	        Dtype(1.0),
	        bout + (j*channels));	
            }
    
            caffe_cpu_axpby(
              channels,
              Dtype(0.0),
              bout + (j*channels),
              top[0]->cpu_diff()[j],
              bout + (j*channels)
            );
      }
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(CosineSimilarityLayer);
#endif

INSTANTIATE_CLASS(CosineSimilarityLayer);
REGISTER_LAYER_CLASS(CosineSimilarity);
}  // namespace caffe
