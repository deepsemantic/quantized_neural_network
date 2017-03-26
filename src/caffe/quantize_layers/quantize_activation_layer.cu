
#include "../../../include/caffe/quantize_layers/quantize_activation_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void QuantizeForward(const int n, const Dtype* in, Dtype* out, const int bits) {
	CUDA_KERNEL_LOOP(index, n) {
		//DLOG(INFO)<<"---index--"<<index;
		//DLOG(INFO)<<"---bottom[0]->count()--"<<in[index];
		//printf("--%d->%f \n ",index,in[index]);
		Dtype x=max(Dtype(0), min(in[index], Dtype(1)));
		out[index] = Dtype((x*bits+0.5)/bits);
	}
}

//@implementation of quantize function
//TO DO: GPU implementation
template <typename Dtype>
void QuantizeActivationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	  //Forward_cpu(bottom, top);
	const Dtype* bottom_data =bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count=bottom[0]->count();
	const int bits=std::pow(2,this->bit_A)-1;
	/*QuantizeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, top_data, bits);
	CUDA_POST_KERNEL_CHECK;*/

	//QuantizeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data,top_data,this->bit_A);
	QuantizeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data,top_data, bits);

	/*for (int index = 0; index < bottom[0]->num(); index++ ) {
	    for (int _h = 0; _h < bottom[0]->height(); _h++ ) {
	      for (int _w = 0; _w < bottom[0]->width(); _w++ ) {
	        for (int _c = 0; _c < bottom[0]->channels(); _c++ ) {
	        	//DLOG(INFO)<<"--->  "<<bottom[0]->data_at(index,_c,_h,_w);
	        	top_data[ top[0]->offset(index,_c,_h,_w)] =
	            quantize_k(bottom[0]->data_at(index,_c,_h,_w),this->bit_A);
	        }
	      }
	    }
	  }*/
}

template <typename Dtype>
__global__ void QuantizeBackward(const int n, const Dtype* bottom_data, const Dtype* top_diff, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, n) {
		if(abs(bottom_data[index])<=Dtype(1)){
			bottom_diff[ index ] = top_diff[ index ];
		}else{
			bottom_diff[ index ] = Dtype(0);
		}
	}
}

template <typename Dtype>
void QuantizeActivationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if ( propagate_down[0] == false ) return;
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count=bottom[0]->count();
	QuantizeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data,top_diff,bottom_diff);

	/*for (int index = 0; index < bottom[0]->count(); index++) {
		if ( std::abs(bottom_data[index]) <= Dtype(1) ) {
			bottom_diff[ index ] = top_diff[ index ];
		} else {
			bottom_diff[ index ] = Dtype(0);
		}
	}*/
  //Backward_cpu(top, propagate_down, bottom);
}



INSTANTIATE_LAYER_GPU_FUNCS(QuantizeActivationLayer);

}  // namespace caffe

