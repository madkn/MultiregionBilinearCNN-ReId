#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/cosine_similarity_batch_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {


template <typename Dtype>
void CosineSimilarityBatchLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  
  xy_ = new Dtype*[bottom[0]->num()];
  for (int i = 0; i < bottom[0]->num(); i++){
    xy_[i] = new Dtype[bottom[0]->num()];
  } 
}
         
template <typename Dtype>
void CosineSimilarityBatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  top[0]->Reshape(bottom[0]->num()*(bottom[0]->num() - 1)/2, 1, 1, 1);
  top[1]->Reshape(bottom[1]->num()*(bottom[1]->num() - 1)/2, 1, 1, 1);
  top[2]->Reshape(bottom[1]->num()*(bottom[1]->num() - 1)/2, 1, 1, 1);
  num_ = bottom[0]->num();
}

template <typename Dtype>
void CosineSimilarityBatchLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int channels = bottom[0]->channels();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    for (int j = 0; j < bottom[0]->num(); ++j) {
      xy_[i][j] = caffe_cpu_dot(bottom[0]->channels(), bottom[0]->cpu_data() + i * channels, bottom[0]->cpu_data() + j * channels);
    }	
  }

  int pos_label = this->layer_param_.cosine_similarity_batch_param().pos_label(); 
  int neg_label = this->layer_param_.cosine_similarity_batch_param().neg_label(); 
  bool eliminate_pos_same_camera =  this->layer_param_.cosine_similarity_batch_param().eliminate_pos_same_camera(); 
  bool eliminate_neg_same_camera =  this->layer_param_.cosine_similarity_batch_param().eliminate_neg_same_camera(); 

  bool eliminate_pos =  this->layer_param_.cosine_similarity_batch_param().eliminate_pos();
  bool eliminate_neg =  this->layer_param_.cosine_similarity_batch_param().eliminate_neg();

  
  int k = 0;
  int pos = 0;
  int neg = 0;

  for (int i = 0; i < bottom[0]->num(); ++i) {
   

    for (int j = i+1; j < bottom[0]->num(); ++j) {
      top[0]->mutable_cpu_data()[k] = xy_[i][j]/sqrt(xy_[i][i] * xy_[j][j]);

      if (bottom[1]->cpu_data()[i] == bottom[1]->cpu_data()[j]){
        top[1]->mutable_cpu_data()[k] = pos_label;
        
        if (eliminate_pos or ((eliminate_pos_same_camera) and (bottom[2]->cpu_data()[i] == bottom[2]->cpu_data()[j]))){
	    top[2]->mutable_cpu_data()[k] = 0;
	       
	} else {
	    top[2]->mutable_cpu_data()[k] = 1;
            pos++;		
	}
      }
      else{
        top[1]->mutable_cpu_data()[k] = neg_label;
        if (eliminate_neg or ((eliminate_neg_same_camera) and (bottom[2]->cpu_data()[i] == bottom[2]->cpu_data()[j]))){
	    top[2]->mutable_cpu_data()[k] = 0;
	}
        else {          
           top[2]->mutable_cpu_data()[k] = 1;
           neg++;
        }
      }
      k++;
    }
  } 
 
}

template <typename Dtype>
void CosineSimilarityBatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
  int channels = bottom[0]->channels();
  int num = bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < channels; ++k) {
      Dtype dsdx = 0;
      for (int j = 0; j < num; ++j) {
	//i j --> h    
        int h;
        if (i < j)
          h =  num * i - i * (i+1)/2 + j-i-1;
	else if (i > j)
          h =  num * j - j * (j+1)/2 + i-j-1;               
        else continue;
        Dtype add_dsdx = top[0]->cpu_diff()[h] * top[2]->cpu_data()[h] * (bottom[0]->cpu_data()[j*channels + k]/sqrt(xy_[i][i] * xy_[j][j]) - xy_[i][j]/xy_[i][i] * bottom[0]->cpu_data()[i*channels + k]/sqrt(xy_[i][i] * xy_[j][j]));
        dsdx += add_dsdx;
    }		
    bottom[0]->mutable_cpu_diff()[i*channels + k] = dsdx; 
  }
} 
}

#ifdef CPU_ONLY
STUB_GPU(CosineSimilarityBatchLayer);
#endif

INSTANTIATE_CLASS(CosineSimilarityBatchLayer);
REGISTER_LAYER_CLASS(CosineSimilarityBatch);
}  // namespace caffe
