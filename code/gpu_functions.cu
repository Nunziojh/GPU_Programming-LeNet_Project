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

        for(i = new_idy; i < (new_idy + POOLING_WINDOW_SIZE); i++){
            for(j = new_idx; j < (new_idx + POOLING_WINDOW_SIZE); j++){
                tmp += (i >= h || j >= w) ? 0 : in[i * w + j];
            }
        }

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

        float tmp[4];
        for(i = 0; i < POOLING_WINDOW_SIZE * POOLING_WINDOW_SIZE; i++) tmp[i] = m[i / POOLING_WINDOW_SIZE * new_w + i % POOLING_WINDOW_SIZE];

        for(i = new_idy; i < (new_idy + POOLING_WINDOW_SIZE); i++){
            for(j = new_idx; j < (new_idx + POOLING_WINDOW_SIZE); j++){
                tot += (i >= new_h || j >= new_w) ? 0 : tmp[i * POOLING_WINDOW_SIZE + j]; //[i * new_w + j];
            }
        }

        if(tot == 0.0){
            for(i = new_idy; i < (new_idy + POOLING_WINDOW_SIZE); i++){
                for(j = new_idx; j < (new_idx + POOLING_WINDOW_SIZE); j++){
                    out[i * new_w + j] = 0.0;
                }
            }
        }
        else {
            for(i = new_idy; i < (new_idy + POOLING_WINDOW_SIZE); i++){
                for(j = new_idx; j < (new_idx + POOLING_WINDOW_SIZE); j++){
                    out[i * new_w + j] = /*m[i * new_w + j]*/tmp[i * POOLING_WINDOW_SIZE + j] / tot * in[idy * w_in + idx];
                }
            }
        }

    }
}

#define TILE_DIM 32
__global__ void matrix_product(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1, int piastrella) { //w_in1 rappresenta la dimensione comune delle due matrici di ingresso
    __shared__ float sA[TILE_DIM][TILE_DIM];   // Tile size of 32x32
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int ph = 0; ph < (((w_in1 - 1) / piastrella) + 1); ph++) {
        if ((Row < h_out) && (threadIdx.x + (ph * piastrella)) < w_in1) {
            sA[threadIdx.y][threadIdx.x] = in1[(Row * w_in1) + threadIdx.x + (ph * piastrella)];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (Col < w_out && (threadIdx.y + ph * piastrella) < w_in1) {
            sB[threadIdx.y][threadIdx.x] = in2[(threadIdx.y + ph * piastrella) * w_out + Col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < piastrella; ++j) {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < h_out && Col < w_out) {
        out[Row * w_out + Col] = Cvalue;
    }
}

/*__global__ void matrix_product(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_out && idy < h_out){

        float tmp = 0.0;
        in1 = in1 + idy * w_in1;
        for(int i = 0, j = idx; i < w_in1; i++, j += w_out){
            tmp += in1[i] * in2[j];
        }

        out[idy * w_out + idx] = tmp;
    }
}*/

__global__ void matrix_transpose_product(float *in1, float *in2, float *out, int w_out, int h_out, int h_in1, int piastrella) {
    __shared__ float sA[TILE_DIM][TILE_DIM];   // Tile size of 32x32
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int ph = 0; ph < (((h_in1 - 1) / piastrella) + 1); ph++) {
        if ((Row < h_out) && (threadIdx.x + (ph * piastrella)) < h_in1) {
            sA[threadIdx.y][threadIdx.x] = in1[(threadIdx.x + ph * piastrella) * h_out + Row]; // (Row * h_in1) + threadIdx.x + (ph * piastrella)
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (Col < w_out && (threadIdx.y + ph * piastrella) < h_in1) {
            sB[threadIdx.y][threadIdx.x] = in2[(threadIdx.y + ph * piastrella) * w_out + Col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < piastrella; ++j) {
            Cvalue += sA[j][threadIdx.y] * sB[j][threadIdx.x];
        }
    }
    if (Row < h_out && Col < w_out) {
        out[Row * w_out + Col] = Cvalue;
    }
}

/*__global__ void matrix_transpose_product(float *in1, float *in2, float *out, int w_out, int h_out, int h_in1){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_out && idy < h_out){

        float tmp = 0;
        for(int i = 0, j = 0; i < h_in1 * h_out; i += h_out, j += w_out){
            tmp += in1[idy + i] * in2[idx + j];
        }
        out[idy * w_out + idx] = tmp;
    }
}*/

__global__ void matrix_product_transpose(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1, int piastrella) { //w_in1 rappresenta la dimensione comune delle due matrici di ingresso
    __shared__ float sA[TILE_DIM][TILE_DIM];   // Tile size of 32x32
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int ph = 0; ph < (((w_in1 - 1) / piastrella) + 1); ph++) {
        if ((Row < h_out) && (threadIdx.x + (ph * piastrella)) < w_in1) {
            sA[threadIdx.y][threadIdx.x] = in1[(Row * w_in1) + threadIdx.x + (ph * piastrella)];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (Col < w_out && (threadIdx.y + ph * piastrella) < w_in1) {
            sB[threadIdx.y][threadIdx.x] = in2[(Col * w_in1) + threadIdx.y + (ph * piastrella)]; //[(threadIdx.y + ph * piastrella) * w_out + Col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < piastrella; ++j) {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];//sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < h_out && Col < w_out) {
        out[Row * w_out + Col] = Cvalue;
    }
}

/*__global__ void matrix_product_transpose(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_out && idy < h_out){

        float tmp = 0;
        for(int i = 0; i < w_in1; i++){
            tmp += in1[idy * w_in1 + i] * in2[idx * w_in1 + i];
        }
        out[idy * w_out + idx] = tmp;
    }
}*/

__global__ void matrix_dot_product(float *in1, float *in2, float *out, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx;
        out[index] = in1[index] * in2[index];
    }
}

__global__ void matrix_scalar_product(float *io, float scalar, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx;
        io[index] = io[index] * scalar;
    }
}

__global__ void matrix3D_scalar_product(float *io, float scalar, int w, int h){
    int c = blockIdx.x;
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx + c * w * h;
        io[index] = io[index] * scalar;
    }
}

__global__ void tanh(float *in, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx;
        float val = in[index];
        float e = expf(2 * val);
        in[index] = (e - 1) / (e + 1);
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
        int index = idy * w + idx + c * w * h;
        io[index] = io[index] - scalar;
    }
}

__global__ void transpose(float *out, float *in, int w_out, int h_out){
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