#ifndef P_MONOLITHIC_H
#define P_MONOLITHIC_H

__global__ void avg_pooling_monolithic(float *input, float *output, int in_width, int in_height, int out_width, int out_height, int depth, int stride, int window_size);
__global__ void inverse_avg_pooling_monolithic(float *input, float *output, float *window, int in_width, int in_height, int out_width, int out_height, int depth, int stride, int window_size);

#endif