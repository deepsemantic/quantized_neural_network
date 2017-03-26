
#include "../../../include/caffe/quantize_layers/quantize_conv_layer.hpp"

#include "caffe/util/benchmark.hpp"
//#include </usr/local/cuda-7.5/include/thrust/host_vector.h>
//#include </usr/local/cuda-7.5/include/thrust/device_vector.h>

namespace caffe {

#define BLOCK_SIZE 16
#define h(x) (std::tanh(x)/2+0.5)
#define sign(x) ((x)>=0?1:-1)


template <typename Dtype>
__global__ void GPU_weights_quantize(const int n, const Dtype* in, int* out,int bits) {
	// find the maximum of weights in this layer
	if (bits==1){
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = sign(in[index]);
		}
	}
	else{
		Dtype max=0;
		for (int index = 0; index < n; ++index) {
			Dtype x=abs(in[index]);
			if (max<=x)
				max=x;
		}
		CUDA_KERNEL_LOOP(index, n) {
			Dtype x=in[index]/max;
			int x_1=floor(x * bits+0.5);
			out[index] = x_1;
		}
	}
}

template <typename Dtype>
__global__ void GPU_activation_quantize(const int n, const Dtype* in, int* out,int bits) {
	CUDA_KERNEL_LOOP(index, n) {
		Dtype x=h(in[index]);
		int x_1=floor(x * bits+0.5);
		out[index] = x_1;
	}
}

// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
template <typename Dtype>
__global__ void gpu_matrix_multiply_share(bool A_trans, bool B_trans,  int* A, int* B, Dtype* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Dtype* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    Dtype Cvalue = 0.0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {

        // Get sub-matrix Asub of A
        int* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Bsub of B
        int* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int j = 0; j < BLOCK_SIZE; ++j)
        	Cvalue += abs(As[row][j])^ abs(Bs[j][col])*sign(As[row][j])*sign(Bs[j][col]);
        	//Cvalue += As[row][j]*Bs[j][col];//*sign(As[row][j])*sign(Bs[j][col]);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    //if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m)
    Csub[row*k+col] = Cvalue;
}

/*add forward convoultional kernel*/
// A is shape (m,k), B is shape (k,n) and C is shape (m,n)
// A: weight (32x800)  B:input (800x256)  C:output (32x256)
template <typename Dtype>
__global__ void gpu_matrix_multiply(bool A_trans, bool B_trans, const int* A, const int* B, Dtype* C,int M,int K, int N){
	Dtype Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<M && col<N){
    //int A_idx,B_idx;
    //printf("row: %d, col: %d\n",row,col);

    for (int e = 0; e < K; ++e){
        int A_idx=row * K + e;
    	int B_idx=e * N + col;
    	//int B_idx=K * col+e;
    	if (A_trans)
    		A_idx=col * K + e;
    	if(B_trans)
    		B_idx=e * col+N;
        Cvalue += abs(A[A_idx])^abs(B[B_idx])*sign(A[A_idx])*sign(B[B_idx]);
    	//Cvalue += abs(A[A_idx])abs(B[B_idx]);//*sign(A[A_idx])*sign(B[B_idx]);
        //Cvalue += A[A_idx]*B[B_idx]
        //printf("Cvalue %f",Cvalue);
    }

}
    C[row *N + col] = Cvalue;
}

template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	const Dtype* weight = this->blobs_[0]->gpu_data();
	const int count=this->blobs_[0]->count();

	int bits=std::pow(2,this->bit_W)-1;
	thrust::device_vector<int> d_W(count);

	d_W_1 = thrust::raw_pointer_cast( &d_W[0]);
	GPU_weights_quantize<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, weight, d_W_1, bits);
	//GPU_weights_quantize<Dtype><<<BLOCK_SIZE, BLOCK_SIZE>>>(count, weight, d_W_1, bits);

	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		for (int n = 0; n < this->num_; ++n) {
			this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->gpu_data();
				this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
			}
		}
	}
}

template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->gpu_diff();
		// Bias gradient, if necessary.
		if (this->bias_term_ && this->param_propagate_down_[1]) {
			Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
			for (int n = 0; n < this->num_; ++n) {
				this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
			}
		}
		if (this->param_propagate_down_[0] || propagate_down[i]) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			for (int n = 0; n < this->num_; ++n) {
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if (this->param_propagate_down_[0]) {
					this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
				}
				// gradient w.r.t. bottom data, if necessary.
				if (propagate_down[i]) {
					this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
							bottom_diff + n * this->bottom_dim_);
				}
			}
		}
	}

}
template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }

  int bits=std::pow(2,this->bit_A)-1;
  thrust::device_vector<int> d_A(col_offset_);
  d_A_1 = thrust::raw_pointer_cast( &d_A[0]);
  GPU_activation_quantize<Dtype><<<CAFFE_GET_BLOCKS(col_offset_), CAFFE_CUDA_NUM_THREADS>>>(col_offset_, col_buff, d_A_1, bits);


  int M=conv_out_channels_ /group_;
  int K=kernel_dim_;
  int N=conv_out_spatial_dim_;

  dim3 dimBlock(CAFFE_GET_BLOCKS(kernel_dim_), CAFFE_CUDA_NUM_THREADS);
  //dim3 dimBlock(2, CAFFE_CUDA_NUM_THREADS);
  //dim3 dimBlock(BLOCKS, THREADS);

  dim3 dimGrid((N+dimBlock.x-1) / dimBlock.x, (M+dimBlock.y-1) / dimBlock.y);
  //(M,K)x(K,N)=(M,N)
  for (int g = 0; g < group_; ++g) {
	   // maxtrix multiply without using shared memory
	  gpu_matrix_multiply<Dtype><<<dimGrid, dimBlock>>>(false,false,d_W_1  + weight_offset_ * g,
			  d_A_1+ col_offset_*g,output + output_offset_ * g, M, K, N);

	  // maxtrix multiply with shared memory
	  //gpu_matrix_multiply_share<Dtype><<<dimGrid, dimBlock>>>(false,false,d_W_1 + weight_offset_ * g,
	//		  d_A_1+ col_offset_*g,output + output_offset_ * g, M, K, N);
	  CUDA_POST_KERNEL_CHECK;

  }
 }

template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }

 /*int M=conv_out_channels_ /group_;
  int N=kernel_dim_;
  int K=conv_out_spatial_dim_;
  for (int g = 0; g < group_; ++g) {
  	  gpu_matirx_multiply<Dtype><<<BLOCK_SIZE,BLOCK_SIZE>>>(true,false,d_W_1+ weight_offset_ * g,
  			  output + output_offset_ * g,col_buff + col_offset_ * g, N,M,K);
    }*/

  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

INSTANTIATE_LAYER_GPU_FUNCS(QuantizeConvolutionLayer);

}


