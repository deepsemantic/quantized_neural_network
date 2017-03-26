#ifndef CAFFE_QUANTIZE_CONVOLUTION_LAYER_HPP_
#define CAFFE_QUANTIZE_CONVOLUTION_LAYER_HPP_

#include <vector>


#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include </usr/local/cuda-7.5/include/thrust/host_vector.h>
#include </usr/local/cuda-7.5/include/thrust/device_vector.h>

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class QuantizeConvolutionLayer : public Layer<Dtype> {
 public:
  explicit QuantizeConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
#endif
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }
  
  virtual inline const char* type() const { return "QuantizeConvolution"; }

 protected:
  // For weights and activation quantize
/*  vector<int>  weight_quantize;
  vector<int>  input_quantize;
  vector<int>  output_quantize;*/

  vector<int>  weight_quantize;
  vector<int>  input_quantize;
  vector<int>  output_quantize;

  //Blob<Dtype> rand_vec_;
  virtual inline Dtype hard_sigmoid(const Dtype value) {
    return std::max(Dtype(0), std::min(Dtype(1), Dtype((value+1)/2)));
  }
  virtual inline void hard_sigmoid(const shared_ptr<Blob<Dtype> > weights) {
    for (int index = 0; index < weights->count(); index++)
      weights->mutable_cpu_data()[index] = hard_sigmoid(weights->cpu_data()[index]);
  }

  virtual int forward_quantize_k(Dtype x,int k);
  virtual void bit_convolutional_cpu_gemm(bool A_trans,bool B_trans,vector<int> A, vector<int> B, Dtype* C, int M, int N, int K);
  virtual void weights_quantize(const shared_ptr<Blob<Dtype> > weights);
  virtual void activation_quantize(const Dtype* input, vector<int> activation_q);
  virtual inline void copyCPUTo(const shared_ptr<Blob<Dtype> > ori, const shared_ptr<Blob<Dtype> > buf) {
    CHECK_EQ(ori->count(), buf->count());
    caffe_copy(ori->count(), ori->cpu_data(), buf->mutable_cpu_data());
  }
  // Pre
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);

  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  void weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);

  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
  //void GPU_weights_quantize(const shared_ptr<Blob<Dtype> > weights,
  //  		const shared_ptr<Blob<Dtype> > weights_q);
  inline void copyGPUTo(const shared_ptr<Blob<Dtype> > ori, const shared_ptr<Blob<Dtype> > buf) {
      CHECK_EQ(ori->count(), buf->count());
      cudaMemcpy(buf->mutable_gpu_data(), ori->gpu_data(), sizeof(Dtype)*ori->count(), cudaMemcpyDefault);
    }

  void forward_gpu_gemm(const Dtype* input,
		  const Dtype* weights, Dtype* output, bool skip_im2col=false);
  void backward_gpu_gemm(const Dtype* output,
		  const Dtype* weights, Dtype* input);

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }



  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  // virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  // virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;
  int bit_W;
  int bit_A;
  int bit_G;
  //int* d_W;
  //int* d_A;

  int* d_A_1;
  int* d_W_1;


 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
        if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
          im2col_gpu(data, conv_in_channels_,
              conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
              kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
              pad_.cpu_data()[0], pad_.cpu_data()[1],
              stride_.cpu_data()[0], stride_.cpu_data()[1],
              dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
        } else {
          im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
              conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
              kernel_shape_.gpu_data(), pad_.gpu_data(),
              stride_.gpu_data(), dilation_.gpu_data(), col_buff);
        }
      }
      inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
        if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
          col2im_gpu(col_buff, conv_in_channels_,
              conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
              kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
              pad_.cpu_data()[0], pad_.cpu_data()[1],
              stride_.cpu_data()[0], stride_.cpu_data()[1],
              dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
        } else {
          col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
              conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
              kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
              dilation_.gpu_data(), data);
        }
      }
#endif
  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;

};

}  // namespace caffe

//#endif  // CAFFE_QUANTIZE_CONVOLUTION_LAYER_HPP_
