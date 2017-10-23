#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/binomial_deviance_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {

template <typename Dtype>
void BinomialDevianceLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num()); 
 
  CHECK_EQ(bottom[0]->channels(), 1); //similarity
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1); //label
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  CHECK_EQ(bottom[2]->channels(), 1); //elimination mask
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);


  exp_.Reshape(bottom[0]->num(), 1, 1, 1);
  M_.Reshape(bottom[0]->num(), 1, 1, 1);
  W_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void BinomialDevianceLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  n1 = 0;
  n2 = 0; 
  for (int i = 0; i < bottom[1]->num(); ++i){
    if (static_cast<int>(bottom[1]->cpu_data()[i]) == 1){
      n1++;
    }	
    else if (static_cast<int>(bottom[1]->cpu_data()[i]) == -1)	{
      n2++;
    }

  }


 // LOG(INFO) << n1 << "  " << n2;
  Dtype c = this->layer_param_.binomial_deviance_loss_param().c(); 
  for (int i = 0; i < bottom[1]->num(); ++i){
    M_.mutable_cpu_data()[i] = static_cast<int>(bottom[1]->cpu_data()[i]);
  
    if (static_cast<int>(bottom[1]->cpu_data()[i]) == 1)
      W_.mutable_cpu_data()[i] = 1.0/n1;
    else if (static_cast<int>(bottom[1]->cpu_data()[i]) == -1)	{
      W_.mutable_cpu_data()[i] = 1.0/n2;
      M_.mutable_cpu_data()[i] = -c;
    }
    else W_.mutable_cpu_data()[i] = 0.0;
  } 
  summer_vec_.Reshape(bottom[0]->num(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->num(); ++i){
    exp_.mutable_cpu_data()[i] = Dtype(1);
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
  }

  Dtype alpha = this->layer_param_.binomial_deviance_loss_param().alpha(); 
  Dtype beta = this->layer_param_.binomial_deviance_loss_param().beta(); 

  caffe_cpu_axpby(
              bottom[1]->num(),
              Dtype(-alpha),
              bottom[0]->cpu_data(),
              Dtype(alpha * beta),
              exp_.mutable_cpu_data());

  caffe_mul(bottom[1]->num(), M_.cpu_data(), exp_.cpu_data(), exp_.mutable_cpu_data());
  caffe_exp(bottom[1]->num(), exp_.cpu_data(), exp_.mutable_cpu_data());
 
  caffe_cpu_axpby(bottom[1]->num(), Dtype(1), exp_.cpu_data(), Dtype(1), summer_vec_.mutable_cpu_data());
  for (int i = 0; i < bottom[0]->num(); ++i){
    summer_vec_.mutable_cpu_data()[i] = log(summer_vec_.cpu_data()[i]);
  }
//// multiply by elimination array
  caffe_mul(bottom[2]->num(), bottom[2]->cpu_data(), summer_vec_.cpu_data(), summer_vec_.mutable_cpu_data());
////
  Dtype loss = caffe_cpu_dot(bottom[1]->num(), W_.cpu_data(), summer_vec_.cpu_data());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BinomialDevianceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  Dtype alpha = this->layer_param_.binomial_deviance_loss_param().alpha(); 
 
  if (propagate_down[0]) { 
    Dtype* bout = bottom[0]->mutable_cpu_diff();

    int num = bottom[0]->num();
    for (int i = 0 ; i < num; i++){
      bout[i] = Dtype(1.0); 
    }
    caffe_cpu_axpby(bottom[0]->num(), Dtype(1), exp_.cpu_data(), Dtype(1), bout);
    caffe_div(bottom[0]->num(), exp_.cpu_data(), bout, bout);
    caffe_mul(bottom[0]->num(), M_.cpu_data(), bout, bout);
    caffe_mul(bottom[0]->num(), W_.cpu_data(), bout, bout);   
    caffe_cpu_axpby(bottom[0]->num(), Dtype(0.0), bout, Dtype(-alpha * top[0]->cpu_diff()[0]), bout);

    //// multiply by elimination array
    caffe_mul(bottom[2]->num(), bottom[2]->cpu_data(), bottom[0]->cpu_diff(), bout);
////
  }
}

#ifdef CPU_ONLY
STUB_GPU(BinomialDevianceLossLayer);
#endif

INSTANTIATE_CLASS(BinomialDevianceLossLayer);
REGISTER_LAYER_CLASS(BinomialDevianceLoss);
}  // namespace caffe
