/*#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define H 32
#define W 32
#define KERNEL_DIM 5
#define KERNEL_NUM 22
#define POOLING_WINDOW_SIZE 2
#define LEARNING_RATE 0.5*/
#include <stdio.h>

#include "gpu_functions.h"

__global__ void convolution(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f, int stride_f){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_dim && idy < out_dim){

        int i, j;

        float tmp = 0;
        float val;

        int new_idx = idx * stride_f - padding_f;
        int new_idy = idy * stride_f - padding_f;

        for(i = 0; i < kernel_dim; i++){
            for(j = 0; j < kernel_dim; j++){
                val = ((new_idy + i) < 0 || (new_idy + i) >= in_dim || (new_idx + j) < 0 || (new_idx + j) >= in_dim) ? 0 : in[(new_idy + i) * in_dim + new_idx + j];
                //val = in[(new_idy + i) * in_dim + new_idx + j];
                tmp += kernel[(kernel_dim * kernel_dim - 1) - (i * kernel_dim + j)] * val;
            }
        }
        out[idy * out_dim + idx] = tmp;
    }
}

__global__ void convolution3D(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f, int stride_f){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_dim && idy < out_dim){

        int i, j;

        float tmp = 0;
        float val;

        int new_idx = idx * stride_f - padding_f;
        int new_idy = idy * stride_f - padding_f;

        for(i = 0; i < kernel_dim; i++){
            for(j = 0; j < kernel_dim; j++){
                val = ((new_idy + i) < 0 || (new_idy + i) >= in_dim || (new_idx + j) < 0 || (new_idx + j) >= in_dim) ? 0 : in[(new_idy + i) * in_dim + new_idx + j];
                //val = in[(new_idy + i) * in_dim + new_idx + j];
                tmp += kernel[(kernel_dim * kernel_dim - 1) - (i * kernel_dim + j)] * val;
            }
        }
        out[idy * out_dim + idx] += tmp;
    }
}

__global__ void avg_pooling(float *in, float *out, int h, int w, int new_h, int new_w, int stride){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;

        float tmp = 0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;

        for(i = 0; i < POOLING_WINDOW_SIZE; i++){
            for(j = 0; j < POOLING_WINDOW_SIZE; j++){
                tmp += ((new_idy + i) >= h || (new_idx + j) >= w) ? 0 : in[(new_idy + i) * w + new_idx + j];
            }
        }

        __syncthreads();

        out[idy * new_w + idx] = tmp / (float)(POOLING_WINDOW_SIZE * POOLING_WINDOW_SIZE);
    }
}

__global__ void inverse_avg_pooling(float *in, float *out, float *m, int w_in, int h_in, int new_w, int new_h, int stride){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_in && idy < h_in){

        int i, j;

        float tot = 0.0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;

        for(i = 0; i < POOLING_WINDOW_SIZE; i++){
            for(j = 0; j < POOLING_WINDOW_SIZE; j++){
                tot += ((new_idy + i) >= new_h || (new_idx + j) >= new_w) ? 0.0 : m[(new_idy + i) * new_w + new_idx + j];
            }
        }

        if(tot == 0.0){
            for(i = 0; i < POOLING_WINDOW_SIZE; i++){
                for(j = 0; j < POOLING_WINDOW_SIZE; j++){
                    out[(new_idy + i) * new_w + new_idx + j] = 0.0;
                }
            }
        }
        else {
            for(i = 0; i < POOLING_WINDOW_SIZE; i++){
                for(j = 0; j < POOLING_WINDOW_SIZE; j++){
                    out[(new_idy + i) * new_w + new_idx + j] = m[(new_idy + i) * new_w + new_idx + j] / tot * in[idy * w_in + idx];
                }
            }
        }

    }
}

__global__ void matrix_product(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_out && idy < h_out){

        float tmp = 0.0;
        in1 = in1 + idy * w_in1;
        for(int i = 0, j = idx; i < w_in1; i++, j += w_out){
            tmp += in1[i] * in2[j];
        }

        out[idy * w_out + idx] = tmp;
        //printf("\t(%d, %d)%e\n", idy, idx, out[idy * w_out + idx]);
    }
}

__global__ void matrix_transpose_product(float *in1, float *in2, float *out, int w_out, int h_out, int h_in1){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_out && idy < h_out){

        float tmp = 0;
        for(int i = 0, j = 0; i < h_in1 * h_out; i += h_out, j += w_out){
            tmp += in1[idy + i] * in2[idx + j];
        }
        out[idy * w_out + idx] = tmp;
    }
}

__global__ void matrix_product_transpose(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_out && idy < h_out){

        float tmp = 0;
        for(int i = 0; i < w_in1; i++){
            tmp += in1[idy * w_in1 + i] * in2[idx * w_in1 + i];
        }
        out[idy * w_out + idx] = tmp;
    }
}

__global__ void matrix_dot_product(float *in1, float *in2, float *out, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx;
        out[index] = in1[index] * in2[index];
    }
}

__global__ void matrix_scalar_product(float *io, float scalar, int w, int h, int stampa){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        io[idy * w + idx] = io[idy * w + idx] * scalar;
        if(stampa == 1) printf("\t(%d, %d) %e\n", idy, idx, io[idy * w + idx]);
    }
}

__global__ void matrix3D_scalar_product(float *io, float scalar, int w, int h, int stampa){
    int c = blockIdx.x;
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    if(idx < w && idy < h){
        io = io + (c * w * h);
        io[idy * w + idx] = io[idy * w + idx] * scalar;
        if(stampa == 1) printf("\t(%d, %d) %e\n", idy, idx, io[idy * w + idx]);
    }
}

__global__ void tanh(float *in, int w, int h, int stampa){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        float val = in[idy * w + idx];
        float p = expf(val);
        float m = expf(-val);

        in[idy * w + idx] = (p - m) / (p + m);

        //in[idy * w + idx] = 1.0 / (1.0 + exp(in[idy * w + idx]));
        if(stampa == 1) printf("\t(%d, %d) %e\n", idy, idx, in[idy * w + idx]);
    }
}

__global__ void exponential(float *in, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < len){
        in[idx] = expf(in[idx]);
    }
}

__global__ void subtraction(float *out, float *in1, float*in2, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < dim){
        out[idx] = in1[idx] - in2[idx]; 
    }
}

__global__ void scalar_subtraction(float *out, float *in, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        
        int index = idy * w + idx;
        out[index] = 1 - in[index];
    }
}

__global__ void subtraction_scalar_parametric(float *io, float scalar, int w, int h){
    int c = blockIdx.x;

    int idx = threadIdx.x;
    int idy = threadIdx.y;



    if(idx < w && idy < h){
        io = io + (c * w * h);
        int index = idy * w + idx;
        io[index] = io[index] - scalar;
    }
}

__global__ void transpose(float *out, float *in, int w_out, int h_out){         // BlockDim = alle dimensioni della matrice di uscita
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < w_out && idy < h_out){
        out[idy * w_out + idx] = in[idx * h_out + idy];
    }
}

__global__ void clean_vector(float *m, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < dim){
        m[idx] = 0;
    }
}