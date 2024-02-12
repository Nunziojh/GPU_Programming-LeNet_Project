#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define H 32
#define W 32
#define KERNEL_DIM 5
#define KERNEL_NUM 22
#define POOLING_WINDOW_SIZE 2

__global__ void convolution(float *in, float *out, float *kernel, int new_h, int new_w, int padding, int stride){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;
        int r = KERNEL_DIM / 2;
        int c = KERNEL_DIM / 2;

        float tmp = 0;
        float val;

        int new_idx = idx * stride - c + padding;
        int new_idy = idy * stride - r + padding;

        for(i = -r; i <= r; i++){
            for(j = -c; j <= c; j++){
                val = ((new_idy + i) < 0 || (new_idy + i) >= H || (new_idx + j) < 0 || (new_idx + j) >= W) ? 0 : in[(new_idy + i) * W + new_idx + j];
                tmp += kernel[(i+1) * KERNEL_DIM + (j+1)] * val;
            }
        }
        out[idy * new_w + idx] = tmp;
    }
}

__global__ void convolution3D(float *in, float *out, float *kernel, int new_h, int new_w, int padding, int stride){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;
        int r = KERNEL_DIM / 2;
        int c = KERNEL_DIM / 2;

        float tmp = 0;
        float val;

        int new_idx = idx * stride - c + padding;
        int new_idy = idy * stride - r + padding;

        for(i = -r; i <= r; i++){
            for(j = -c; j <= c; j++){
                val = ((new_idy + i) < 0 || (new_idy + i) >= H || (new_idx + j) < 0 || (new_idx + j) >= W) ? 0 : in[(new_idy + i) * W + new_idx + j];
                tmp += kernel[(i+1) * KERNEL_DIM + (j+1)] * val;
            }
        }
        out[idy * new_w + idx] = tmp;
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

__global__ void matrix_product(float *in, float *weights, float *res, int w_weights, int h_in, int w_in){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_weights && idy < h_in){

        float tmp = 0;
        in = in + idy * w_in;       //Inutile
        for(int i = 0, j = 0; i < w_in; i++, j += w_weights){
            tmp += in[i] * weights[j + idx];
        }

        res[idy * w_weights + idx] = tmp;
    }
}

__global__ void tanh(float *in, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){

        in[idy * w + idx] = 1 / (1 + exp(in[idy * w + idx]));
    }
}

int main(){
    srand(time(NULL));

    int padding = 0;
    int stride_c = 1;
    int stride_p = 2;
    int kernel_num_first_layer = 6;
    int kernel_num_second_layer = 16;
    int kernel_num_third_layer = 120;
    int fc_first_dim = 120;
    int fc_second_dim = 84;
    int fc_third_dim = 10;

    int in_h = 32;
    int in_w = 32;
    int out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
    int out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;

    float *kernels_first_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer);
    float *kernels_second_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer);
    float *kernels_third_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer);
    float *fc_first_layer = (float *) malloc(sizeof(float) * fc_first_dim * fc_second_dim);
    float *fc_second_layer = (float *) malloc(sizeof(float) * fc_second_dim * fc_third_dim);
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer; i++) kernels_first_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer; i++) kernels_second_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer; i++) kernels_third_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < fc_first_dim * fc_second_dim; i++) fc_first_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < fc_second_dim * fc_third_dim; i++) fc_second_layer[i] = (float)rand() / (float)RAND_MAX;

    float *kernels_first_layer_dev, *kernels_second_layer_dev, *kernels_third_layer_dev, *fc_first_layer_dev, *fc_second_layer_dev;
    cudaMalloc((void **)&kernels_first_layer_dev, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer);
    cudaMalloc((void **)&kernels_second_layer_dev, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer);
    cudaMalloc((void **)&kernels_third_layer_dev, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer);
    cudaMalloc((void **)&fc_first_layer_dev, sizeof(float) * fc_first_dim * fc_second_dim);
    cudaMalloc((void **)&fc_second_layer_dev, sizeof(float) * fc_second_dim * fc_third_dim);

    float *img = (float *) malloc(sizeof(float) * in_h * in_w);
    for(int i = 0; i < in_w * in_w; i++) img[i] = i;

    float *img_dev, *first_conv;
    cudaMalloc((void **)&img_dev, sizeof(float) * in_w * in_w);
    cudaMalloc((void **)&first_conv, sizeof(float) * out_w * out_h * kernel_num_first_layer);

    cudaMemcpy(img_dev, img, sizeof(float) * in_w * in_h, cudaMemcpyHostToDevice);

    dim3 block = {out_w, out_h};
    dim3 grid = {out_w / 32 + 1, out_h / 32 + 1};
    for(int i = 0; i < kernel_num_first_layer; i++){
        convolution<<<grid, block>>>(img_dev, first_conv + (i * out_h * out_w), kernels_first_layer_dev + (i * KERNEL_DIM * KERNEL_DIM), out_h, out_w, padding, stride_c);
        tanh<<<grid, block>>>(first_conv + (i * out_h * out_w), out_w, out_h);
    }

    in_h = out_h;
    in_w = out_w; 
    out_h = (in_h - POOLING_WINDOW_SIZE) / stride_p + 1;
    out_w = (in_w - POOLING_WINDOW_SIZE) / stride_p + 1;
    float *first_pool;
    cudaMalloc((void **)&first_pool, sizeof(float) * out_w * out_h * kernel_num_first_layer);
    block = {out_w, out_h};
    grid = {out_w / 32 + 1, out_h / 32 + 1};
    for(int i = 0; i < kernel_num_first_layer; i++){
        avg_pooling<<<grid, block>>>(first_conv + (i * in_h * in_w), first_pool + (i * out_h * out_w), in_h, in_w, out_h, out_w, stride_p);
        tanh<<<grid, block>>>(first_pool + (i * out_h * out_w), out_w, out_h);
    }

    in_h = out_h;
    in_w = out_w;
    out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
    out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;
    float *second_conv;
    cudaMalloc((void **)&second_conv, sizeof(float) * out_w * out_h * kernel_num_second_layer);
    block = {out_w, out_h};
    grid = {out_w / 32 + 1, out_h / 32 + 1};
    for(int i = 0; i < kernel_num_second_layer; i++){
        convolution<<<grid, block>>>(first_pool, second_conv + (i * out_h * out_w), kernels_second_layer_dev + (i * KERNEL_DIM * KERNEL_DIM), out_h, out_w, padding, stride_c);
        tanh<<<grid, block>>>(second_conv + (i * out_h * out_w), out_w, out_h);
    }

    in_h = out_h;       //10
    in_w = out_w;
    out_h = (in_h - POOLING_WINDOW_SIZE) / stride_p + 1;            //5
    out_w = (in_w - POOLING_WINDOW_SIZE) / stride_p + 1;
    float *second_pool;
    cudaMalloc((void **)&second_pool, sizeof(float) * out_w * out_h * kernel_num_second_layer);
    block = {out_w, out_h};
    grid = {out_w / 32 + 1, out_h / 32 + 1};
    for(int i = 0; i < kernel_num_second_layer; i++){
        avg_pooling<<<grid, block>>>(second_conv + (i * in_h * in_w), second_pool + (i * out_h * out_w), in_h, in_w, out_h, out_w, stride_p);
        tanh<<<grid, block>>>(second_pool + (i * out_h * out_w), out_w, out_h);
    }

    in_h = out_h;       //5
    in_w = out_w;
    out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;       //1
    out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;
    float *third_conv;
    cudaMalloc((void **)&third_conv, sizeof(float) * out_w * out_h * kernel_num_third_layer);
    block = {out_w, out_h};
    grid = {out_w / 32 + 1, out_h / 32 + 1};
    for(int i = 0; i < kernel_num_third_layer; i++){
        convolution<<<grid, block>>>(second_pool, third_conv + (i * out_h * out_w), kernels_third_layer_dev + (i * KERNEL_DIM * KERNEL_DIM), out_h, out_w, padding, stride_c);
        tanh<<<grid, block>>>(third_conv + (i * out_h * out_w), out_w, out_h);
    }

    float *second_fc;
    cudaMalloc((void **)&second_fc, sizeof(float) * fc_second_dim);
    block = {fc_second_dim};
    grid = {block.x / 32 + 1};
    matrix_product<<<grid, block>>>(third_conv, fc_first_layer_dev, second_fc, fc_second_dim, 1, fc_first_dim);
    tanh<<<grid, block>>>(second_fc, fc_second_dim, 1);

    float *third_fc;
    cudaMalloc((void **)&third_fc, sizeof(float) * fc_third_dim);
    block = {fc_third_dim};
    grid = {block.x / 32 + 1};
    matrix_product<<<grid, block>>>(second_fc, fc_second_layer_dev, third_fc, fc_third_dim, 1, fc_second_dim);
    

}