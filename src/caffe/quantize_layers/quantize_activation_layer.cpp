
#include "../../../include/caffe/quantize_layers/quantize_activation_layer.hpp"

#include "caffe/util/math_functions.hpp"

namespace caffe {

//activation function h(x)=tanh(x)/2+0.5, which ensures x belongs to [0,1]
//TO DO: test other activations in paper
#define h(x) (std::tanh(x)/2+0.5)

template <typename Dtype>
Dtype QuantizeActivationLayer<Dtype>::clip(Dtype n, Dtype lower, Dtype upper) {
  return std::max(lower, std::min(n, upper));
}

template <typename Dtype>
void QuantizeActivationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom.size(), 1);
  bit_A=this->layer_param_.quantize_activation_param().quantize_bits();
}


template <typename Dtype>
void QuantizeActivationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom.size(), 1);
  top[0]->ReshapeLike(*bottom[0]);

  }

//@implementation of quantize function for activations
template <typename Dtype>
Dtype QuantizeActivationLayer<Dtype>::quantize_k(Dtype x,int k){
	  float n = float(std::pow(2,k)-1);
	  //DLOG(INFO)<<" n "<<n;
	  //x=this->clip(x,0,1);
	  //DLOG(INFO)<<" quantize activation "<<x<<" to "<<k<< " bits: "<<std::floor(x * n+0.5) /n;
	  return std::floor(x * n+0.5) /n;
}


template <typename Dtype>
void QuantizeActivationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count=bottom[0]->count();
  Dtype maxium=0;
  for (int index = 0; index <count; ++index) {
	  if(bottom_data[index]>maxium)
		  maxium=bottom_data[index];

  }
  for (int index = 0; index <count; ++index) {
	 // DLOG(INFO)<<"---index--"<<index;
	  //DLOG(INFO)<<"---bottom[0]->count()--"<<bottom[0]->count();
	 // DLOG(INFO)<<"--bottom_data[0]---"<< bottom_data[index];
	  top_data[index] =
			  quantize_k(bottom_data[index]/maxium,this->bit_A);
  }
}

template <typename Dtype>
void QuantizeActivationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if ( propagate_down[0] == false ) return;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  for (int index = 0; index < bottom[0]->count(); index++) {
	  //DLOG(INFO)<<"---bottom_data[index]--"<<bottom_data[index];
	  //DLOG(INFO)<<"---top_diff[ index ]--"<<top_diff[ index ];
	  if ( std::abs(bottom_data[index]) <= Dtype(1) ) {
		  bottom_diff[ index ] = top_diff[ index ];
	  } else {
		  bottom_diff[ index ] = Dtype(0);
	  }
  } 
}

#ifdef CPU_ONLY
STUB_GPU(QuantizeActivationLayer);
#endif

INSTANTIATE_CLASS(QuantizeActivationLayer);
REGISTER_LAYER_CLASS(QuantizeActivation);

}  // namespace caffe

