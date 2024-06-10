#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void avg_pooling_monolithic(float *input, float *output, int in_width, int in_height, int out_width, int out_height, int depth, int stride, int window_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if(idx < out_width && idy < out_height && idz < depth){

        int i, j;

        float tmp = 0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;

        for(i = new_idy; i < (new_idy + window_size); i++){
            for(j = new_idx; j < (new_idx + window_size); j++){
                tmp += (i >= in_height || j >= in_width) ? 0 : input[idz * in_height * in_width + i * in_width + j];
            }
        }

        output[idz * out_height * out_width + idy * out_width + idx] = tmp / (float)(window_size * window_size);
    }
}

__global__ void inverse_avg_pooling_monolithic(float *input, float *output, float *window, int in_width, int in_height, int out_width, int out_height, int depth, int stride, int window_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if(idx < in_width && idy < in_height && idz < depth){

        int i, j;

        float tot = 0.0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;
        window += (idz * out_width * out_height + new_idy * out_width + new_idx);

        float tmp[4];

        for(i = 0; i < window_size; i++){
            for(j = 0; j < window_size; j++){
                tmp[i * window_size + j] = window[i * out_width + j];
                tot += tmp[i * window_size + j];
            }
        }

        output += (idz * out_width * out_height + new_idy * out_width + new_idx);
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
                    output[i * out_width + j] = tmp[i * window_size + j] / tot * input[idz * in_height * in_width + idy * in_width + idx];
                }
            }
        }

    }
}