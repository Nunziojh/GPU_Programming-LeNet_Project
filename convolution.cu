#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define h_k 3
#define w_k 3
#define h_m 5
#define w_m 5
#define stride 1
#define padding 0

__constant__ int kernel[h_k * w_k];

__global__ void kernel_function(int *in, int *out, int new_h, int new_w){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;
        int r = h_k / 2;
        int c = w_k / 2;

        int tmp = 0;
        int val;

        int new_idx = idx * stride + c - padding;
        int new_idy = idy * stride + r - padding;

        for(i = -r; i <= r; i++){
            for(j = -c; j <= c; j++){
                val = ((new_idy + i) < 0 || (new_idy + i) >= h_m || (new_idx + j) < 0 || (new_idx + j) >= w_m) ? 0 : in[(new_idy + i) * w_m + new_idx + j];
                tmp += kernel[(r-i) * w_k + (c-j)] * val;
            }
        }
        out[idy * new_w + idx] = tmp;
    }
}



int main(int argc, char **argv){

    srand(time(NULL));

    int host_matrix[] = {1, 5, 2, 3, 6,
                        7, 10, 2, 8, 4,
                        10, 6, 5, 4, 3,
                        2, 2, 2, 1, 1,
                        1, 6, 7, 8, 9};
    int host_kernel[] = {1, 1, 1,
                        0, 0, 0,
                        -1, -1, -1};
    //for(int i = 0; i< w_k * h_k; i++) host_kernel[i] = 1;
    //int *host_input = (int *)malloc(sizeof(int) * h_m * w_m);
    
    int new_h = (h_m + 2 * padding - h_k) / stride + 1;
    int new_w = (w_m + 2 * padding - w_k) / stride + 1;
    int *host_res = (int *)malloc(sizeof(int) * new_h * new_w);

    int *dev_input, *dev_res;
    cudaMalloc((void **)&dev_input, h_m * w_m * sizeof(int));
    cudaMalloc((void **)&dev_res, new_h * new_w * sizeof(int));
    cudaMemcpy(dev_input, host_matrix, sizeof(int) * h_m * w_m, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(kernel, &host_kernel, sizeof(int) * w_k * h_k);
    
    dim3 block = {32, 32};
    dim3 grid = {new_w / block.x + 1, new_h / block.y + 1};

    kernel_function<<<grid, block>>>(dev_input, dev_res, new_h, new_w);

    cudaMemcpy(host_res, dev_res, sizeof(int) * new_h * new_w, cudaMemcpyDeviceToHost);

    for(int i = 0; i < new_h; i++){
        for(int j = 0; j < new_w; j++){
            printf("%d ", host_res[i * new_w + j]);
        }
        printf("\n");
    }

    //free(host_matrix);
    free(host_res);
    cudaFree(dev_input);
    cudaFree(dev_res);

    return 0;

}