=======
# The quantization of CNN/LSTM
Ongoing work that tries to quantize the weights, activations as well as gradients to k-bit representation for model compression. Use bit-wise operation to speed up inference procedure in test phase. 

Implemented tasks:
- (1) Add new layer for weights quantization
- (2) Add new layer for activations quantization
- (3) CPU and GPU versions of above tasks

The implementation is based on Caffe (https://github.com/BVLC/caffe).
