
#ifndef CAFFE_COSINE_SIMILARITY_BATCH_LAYER_HPP_
#define CAFFE_COSINE_SIMILARITY_BATCH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"



namespace caffe {



template <typename Dtype>
class CosineSimilarityBatchLayer : public Layer<Dtype>{
 public:
  explicit CosineSimilarityBatchLayer(const LayerParameter& param)
      : Layer<Dtype>(param), xy_(NULL) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); 

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline const char* type() const { return "CosineSimilarityBatch"; }
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index == 0;
  }
  virtual ~CosineSimilarityBatchLayer(){
        for (int i = 0; i < num_; i++){
		delete [] xy_[i];
  	}
	delete [] xy_;        
  } 
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  Dtype **xy_; 
  int num_;
};

}  // namespace caffe


#endif  // CAFFE_COSINE_SIMILARITY_BATCH_LAYER_HPP_
