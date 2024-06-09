#ifndef MONOLITHIC_H
#define MONOLITHIC_H

__global__ void convolution_3D(float *input, float *kernel, float *output, int in_width, int in_height, int in_depth, int ker_width, int ker_height, int ker_depth, int out_width, int out_height, int out_depth);
__global__ void convolution_forNOutChannels(float *in, float *kernel, float *out, int in_w, int in_h, int in_d, int kernel_w, int kernel_h, int kernel_d, int out_w, int out_h, int out_d, int out_n);
__global__ void full_Convolution(float *input, float *kernel, float *output, int in_width, int in_height, int in_depth, int ker_width, int ker_height, int ker_depth, int ker_number, int out_width, int out_height, int out_depth, int padding);

#endif