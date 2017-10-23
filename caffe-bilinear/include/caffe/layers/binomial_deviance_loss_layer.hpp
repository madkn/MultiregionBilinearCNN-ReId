#ifndef CAFFE_BINOMIAL_DEVIANCE_LOSS_HPP_
#define CAFFE_BINOMIAL_DEVIANCE_LOSS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BinomialDevianceLossLayer : public LossLayer<Dtype> {
 public:
  explicit BinomialDevianceLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline const char* type() const { return "BinomialDevianceLoss"; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom); 
  
  Blob<Dtype> exp_; 
  Blob<Dtype> M_;  
  Blob<Dtype> W_;  
  int n1, n2; 
  Blob<Dtype> summer_vec_;  
};



}  // namespace caffe

#endif  // CAFFE_BINOMIAL_DEVIANCE_LOSS_HPP_


