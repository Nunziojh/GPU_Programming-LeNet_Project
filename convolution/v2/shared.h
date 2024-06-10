#ifndef C_SHARED_H
#define C_SHARED_H

__global__ void convolution_shared(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding);

#endif