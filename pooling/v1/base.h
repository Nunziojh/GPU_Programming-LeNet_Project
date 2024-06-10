#ifndef P_BASE_H
#define P_BASE_H

__global__ void avg_pooling(float *input, float *output, int in_width, int in_height, int out_width, int out_height, int stride, int window_size);
__global__ void inverse_avg_pooling_base(float *input, float *output, float *window, int in_width, int in_height, int out_width, int out_height, int stride, int window_size);

#endif