#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void avg_pooling(float *input, float *output, int in_width, int in_height, int out_width, int out_height, int stride, int window_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_width && idy < out_height){

        int i, j;

        float tmp = 0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;

        for(i = new_idy; i < (new_idy + window_size); i++){
            for(j = new_idx; j < (new_idx + window_size); j++){
                tmp += (i >= in_height || j >= in_width) ? 0 : input[i * in_width + j];
            }
        }

        output[idy * out_width + idx] = tmp / (float)(window_size * window_size);
    }
}

__global__ void inverse_avg_pooling_base(float *input, float *output, float *window, int in_width, int in_height, int out_width, int out_height, int stride, int window_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < in_width && idy < in_height){

        int i, j;

        float tot = 0.0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;

        for(i = 0; i < window_size; i++){
            for(j = 0; j < window_size; j++){
                tot += ((new_idy + i) >= out_height || (new_idx + j) >= out_width) ? 0.0 : window[(new_idy + i) * out_width + new_idx + j];
            }
        }

        if(tot == 0.0){
            for(i = 0; i < window_size; i++){
                for(j = 0; j < window_size; j++){
                    output[(new_idy + i) * out_width + new_idx + j] = 0.0;
                }
            }
        }
        else {
            for(i = 0; i < window_size; i++){
                for(j = 0; j < window_size; j++){
                    output[(new_idy + i) * out_width + new_idx + j] = window[(new_idy + i) * out_width + new_idx + j] / tot * input[idy * in_width + idx];
                }
            }
        }

    }
}