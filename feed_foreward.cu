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

    int in_h = H;
    int in_w = W;
    int out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
    int out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;

    float *kernels_first_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer);
    float *kernels_second_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer);
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer; i++) kernels_first_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer; i++) kernels_first_layer[i] = (float)rand() / (float)RAND_MAX;

    float *img = (float *) malloc(sizeof(float) * in_h * in_w);
    for(int i = 0; i < in_w * in_w; i++) img[i] = i;

    float *img_dev, *dev_res;
    cudaMalloc((void **)&img_dev, sizeof(float) * in_w * in_w);
    cudaMalloc((void **)&dev_res, sizeof(float) * out_w * out_h * kernel_num_first_layer);

    cudaMemcpy(img_dev, img, sizeof(float) * in_w * in_h, cudaMemcpyHostToDevice);

    dim3 block = {out_w, out_h};
    dim3 grid = {out_w / 32 + 1, out_h / 32 + 1};
    
    for(int i = 0; i < kernel_num_first_layer; i++){
        convolution<<<grid, block>>>(img_dev, dev_res + (i * out_h * out_w), kernels_first_layer + (i * KERNEL_DIM * KERNEL_DIM), out_h, out_w, padding, stride_c);
        in_h = out_h;
        in_w = out_w;
        tanh<<<grid, block>>>(dev_res + (i * in_h * in_w), in_w, in_h);
        out_h = (in_h - POOLING_WINDOW_SIZE) / stride_p + 1;
        out_w = (in_w - POOLING_WINDOW_SIZE) / stride_p + 1;
        block = {out_w, out_h};
        grid = {out_w / 32 + 1, out_h / 32 + 1};
        avg_pooling<<<grid, block>>>(dev_res + (i * in_h * in_w), dev_res + (i * out_h * out_w), in_h, in_w, out_h, out_w, stride_p);
        in_h = out_h;
        in_w = out_w;
        tanh<<<grid, block>>>(dev_res + (i * in_h * in_w), in_w, in_h);
    }

    

    in_h = out_h;
    in_w = out_w;
    out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
    out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;

    dim3 block = {out_w, out_h};
    dim3 grid = {out_w / 32 + 1, out_h / 32 + 1};

    for(int i = 0; i < kernel_num_second_layer; i++){
        convolution<<<grid, block>>>(img_dev, dev_res + (i * out_h * out_w), kernels_second_layer + (i * KERNEL_DIM * KERNEL_DIM), out_h, out_w, padding, stride_c);
        in_h = out_h;
        in_w = out_w;
        tanh<<<grid, block>>>(dev_res + (i * in_h * in_w), in_w, in_h);
        out_h = (in_h - POOLING_WINDOW_SIZE) / stride_p + 1;
        out_w = (in_w - POOLING_WINDOW_SIZE) / stride_p + 1;
        block = {out_w, out_h};
        grid = {out_w / 32 + 1, out_h / 32 + 1};
        avg_pooling<<<grid, block>>>(dev_res + (i * in_h * in_w), dev_res + (i * out_h * out_w), in_h, in_w, out_h, out_w, stride_p);
        in_h = out_h;
        in_w = out_w;
        tanh<<<grid, block>>>(dev_res + (i * in_h * in_w), in_w, in_h);
    }
}