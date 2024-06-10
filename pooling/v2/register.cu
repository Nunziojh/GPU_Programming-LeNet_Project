#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void inverse_avg_pooling_reg(float *input, float *output, float *window, int in_width, int in_height, int out_width, int out_height, int stride, int window_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < in_width && idy < in_height){

        int i, j;

        float tot = 0.0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;
        window += new_idy * out_width + new_idx;

        float tmp[4];

        for(i = 0; i < window_size; i++){
            for(j = 0; j < window_size; j++){
                tmp[i * window_size + j] = window[i * out_width + j];
                tot += tmp[i * window_size + j];
            }
        }

        output += new_idy * out_width + new_idx;
        if(tot == 0.0){
            for(i = 0; i < window_size; i++){
                for(j = 0; j < window_size; j++){
                    output[i * out_width + j] = 0.0;
                }
            }
        }
        else {
            for(i = 0; i < window_size; i++){
                for(j = 0; j < window_size; j++){
                    output[i * out_width + j] = tmp[i * window_size + j] / tot * input[idy * in_width + idx];
                }
            }
        }

    }
}