#ifndef MONOLITHIC_SHARED_H
#define MONOLITHIC_SHARED_H

#include "..\functions.h"

__global__ void convolution_3D_shared(float *input, float *kernel, float *output, int in_width, int in_height, int in_depth, int ker_width, int ker_height, int ker_depth, int ker_number, int out_width, int out_height, int out_depth);
__global__ void convolution_forNOutChannels_shared(float *in, float *kernel, float *out, int in_w, int in_h, int in_d, int kernel_w, int kernel_h, int kernel_d, int kernel_n, int out_w, int out_h, int out_d, int out_n);

#endif