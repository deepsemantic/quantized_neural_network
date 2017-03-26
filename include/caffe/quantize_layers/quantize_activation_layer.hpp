#ifndef CAFFE_QUANTIZEACTIVATION_LAYER_HPP_
#define CAFFE_QUANTIZEACTIVATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

/**
 * @brief Takes a Blob and slices it along either the num or channel dimension,
 *        outputting multiple sliced Blob results.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class QuantizeActivationLayer : public Layer<Dtype> {
 public:
  explicit QuantizeActivationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuantizeActivation"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual Dtype quantize_k(Dtype x,int k);
  //virtual void QuantizeForward(const int count, const Dtype* in, Dtype* out, const int bits);
 // virtual void QuantizeForward(const int count, const Dtype* in, Dtype* out, int bits);
  virtual Dtype clip(Dtype n, Dtype lower, Dtype upper);
  int bit_A;

};

}  // namespace caffe

#endif  // CAFFE_QUANTIZEACTIVATION_LAYER_HPP_
