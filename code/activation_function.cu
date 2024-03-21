#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define h_m 10
#define w_m 32
#define leakyReluSlope 0.1

__global__ void sigmoid(float *in){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_m && idy < h_m){

        in[idy * w_m + idx] = 1 / (1 + exp(in[idy * w_m + idx]));
    }
}

__global__ void tanh(float *in){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_m && idy < h_m){

        in[idy * w_m + idx] = 1 / (1 + exp(in[idy * w_m + idx]));
    }
}

__global__ void relu(float *in){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_m && idy < h_m){

        in[idy * w_m + idx] = max(0.0, in[idy * w_m + idx]);
    }
}

__global__ void leakyRelu(float *in){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_m && idy < h_m){

        max(0.0, in[idy * w_m + idx]) + (-leakyReluSlope) * min(0.0, in[idy * w_m + idx]);

        in[idy * w_m + idx] = max(0.0, in[idy * w_m + idx]);
    }
}

__global__ void softmax(float *in, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < len){
        in[idx] = exp(in[idx]);
    }
}

int main(int argc, char **argv){

    float *host = (float *)malloc(sizeof(float) * h_m);
    for(int i = 0; i < h_m; i++) host[i] = i;

    float *dev;
    cudaMalloc((void **)&dev, h_m * sizeof(float));
    cudaMemcpy(dev, host, sizeof(float) * h_m, cudaMemcpyHostToDevice);
    
    dim3 block = {h_m};
    dim3 grid = {w_m / block.x + 1};

    softmax<<<grid, block>>>(dev, h_m);

    cudaMemcpy(host, dev, sizeof(float) * h_m, cudaMemcpyDeviceToHost);

    for(int j = 0; j < h_m; j++){
        printf("%2.2f ", host[j]);
    }
    printf("\n");

    free(host);
    cudaFree(dev);

    return 0;

}