#ifndef SHARED_H
#define SHARED_H

#include "..\functions.h"

__global__ void convolution_shared(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim/*, int padding_f, int stride_f*/);

#endif