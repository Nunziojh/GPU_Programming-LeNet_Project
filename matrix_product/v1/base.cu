#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void matrix_product(float *m1, float *m2, float *output, int out_width, int out_height, int m1_width){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_width && idy < out_height){

        float tmp = 0.0;
        m1 = m1 + idy * m1_width;
        for(int i = 0, j = idx; i < m1_width; i++, j += out_width){
            tmp += m1[i] * m2[j];
        }

        output[idy * out_width + idx] = tmp;
    }
}

__global__ void matrix_transpose_product(float *m1, float *m2, float *output, int out_width, int out_height, int m1_height){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_width && idy < out_height){

        float tmp = 0;
        for(int i = 0, j = 0; i < m1_height * out_height; i += out_height, j += out_width){
            tmp += m1[idy + i] * m2[idx + j];
        }
        output[idy * out_width + idx] = tmp;
    }
}

__global__ void matrix_product_transpose(float *m1, float *m2, float *output, int out_width, int out_height, int m1_width){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_width && idy < out_height){

        float tmp = 0;
        for(int i = 0; i < m1_width; i++){
            tmp += m1[idy * m1_width + i] * m2[idx * m1_width + i];
        }
        output[idy * out_width + idx] = tmp;
    }
}

__global__ void matrix_dot_product(float *m1, float *m2, float *output, int width, int height){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < width && idy < height){
        int index = idy * width + idx;
        output[index] = m1[index] * m2[index];
    }
}

__global__ void matrix_scalar_product(float *in_out, float scalar, int width, int height){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < width && idy < height){
        int index = idy * width + idx;
        in_out[index] = in_out[index] * scalar;
    }
}

__global__ void matrix3D_scalar_product(float *in_out, float scalar, int width, int height){
    int c = blockIdx.x;
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    if(idx < width && idy < height){
        int index = idy * width + idx + c * width * height;
        in_out[index] = in_out[index] * scalar;
    }
}