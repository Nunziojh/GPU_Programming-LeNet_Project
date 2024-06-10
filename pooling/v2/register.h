#ifndef P_REGISTER_H
#define P_REGISTER_H

__global__ void inverse_avg_pooling_reg(float *input, float *output, float *window, int in_width, int in_height, int out_width, int out_height, int stride, int window_size);

#endif