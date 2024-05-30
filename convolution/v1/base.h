#ifndef BASE_H
#define BASE_H

#include "..\functions.h"

__global__ void convolution_base(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f/*, int stride_f*/);

#endif