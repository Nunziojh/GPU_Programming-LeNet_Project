#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define dim_f 2
#define h_m 8
#define w_m 8
#define stride 2

__global__ void max_pooling(int *in, int *out, int new_h, int new_w){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;

        int tmp = 0;
        int val;

        int new_idx = idx * stride;
        int new_idy = idy * stride;

        for(i = 0; i < dim_f; i++){
            for(j = 0; j < dim_f; j++){
                val = ((new_idy + i) >= h_m || (new_idx + j) >= w_m) ? 0 : in[(new_idy + i) * w_m + new_idx + j];
                tmp = max(tmp, val);
            }
        }
        out[idy * new_w + idx] = tmp;
    }
}

__global__ void avg_pooling(int *in, float *out, int new_h, int new_w){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;

        float tmp = 0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;

        for(i = 0; i < dim_f; i++){
            for(j = 0; j < dim_f; j++){
                tmp += ((new_idy + i) >= h_m || (new_idx + j) >= w_m) ? 0 : in[(new_idy + i) * w_m + new_idx + j];
            }
        }
        out[idy * new_w + idx] = tmp / (float)(dim_f * dim_f);
    }
}

int main(int argc, char **argv){

    int *host_input = (int *)malloc(sizeof(int) * h_m * w_m);
    for(int i = 0; i < h_m * w_m; i++) host_input[i] = i;

    for(int i = 0; i < h_m; i++){
        for(int j = 0; j < w_m; j++){
            printf("%d ", host_input[i * w_m + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    int new_h = (h_m - dim_f) / stride + 1;
    int new_w = (w_m - dim_f) / stride + 1;
    int *host_res_max = (int *)malloc(sizeof(int) * new_h * new_w);
    float *host_res_avg = (float *)malloc(sizeof(float) * new_h * new_w);

    int *dev_input, *dev_max;
    float *dev_avg;
    cudaMalloc((void **)&dev_input, h_m * w_m * sizeof(int));
    cudaMalloc((void **)&dev_max, new_h * new_w * sizeof(int));
    cudaMalloc((void **)&dev_avg, new_h * new_w * sizeof(float));
    cudaMemcpy(dev_input, host_input, sizeof(int) * h_m * w_m, cudaMemcpyHostToDevice);
    
    dim3 block = {32, 32};
    dim3 grid = {new_w / block.x + 1, new_h / block.y + 1};

    max_pooling<<<grid, block>>>(dev_input, dev_max, new_h, new_w);

    cudaMemcpy(host_res_max, dev_max, sizeof(int) * new_h * new_w, cudaMemcpyDeviceToHost);
    for(int i = 0; i < new_h; i++){
        for(int j = 0; j < new_w; j++){
            printf("%d ", host_res_max[i * new_w + j]);
        }
        printf("\n");
    }
    printf("\n");

    avg_pooling<<<grid, block>>>(dev_input, dev_avg, new_h, new_w);
    
    cudaMemcpy(host_res_avg, dev_avg, sizeof(float) * new_h * new_w, cudaMemcpyDeviceToHost);
    for(int i = 0; i < new_h; i++){
        for(int j = 0; j < new_w; j++){
            printf("%02.3f ", host_res_avg[i * new_w + j]);
        }
        printf("\n");
    }

    free(host_input);
    free(host_res_max);
    free(host_res_avg);
    cudaFree(dev_input);
    cudaFree(dev_max);
    cudaFree(dev_avg);

    return 0;

}