#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define h_a 5
#define w_a 2
#define h_b w_a
#define w_b 3

__global__ void kernel_function(int *a, int *b, int *c){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_b && idy < h_a){

        int tmp = 0;
        a = a + idy * w_a;
        for(int i = 0, j = 0; i < w_a; i++, j += w_b){
            tmp += a[i] * b[j + idx];
        }

        c[idy * w_b + idx] = tmp;
    }
}

int main(int argc, char **argv){

    srand(time(NULL));

    int *host_a = (int *)malloc(sizeof(int) * h_a * w_a);
    for(int i = 0; i < h_a * w_a; i++) host_a[i] = i;
    int *host_b = (int *)malloc(sizeof(int) * h_b * w_b);
    for(int i = 0; i < h_b * w_b; i++) host_b[i] = i;
    int *host_c = (int *)malloc(sizeof(int) * h_a * w_b);

    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, h_a * w_a * sizeof(int));
    cudaMalloc((void **)&dev_b, h_b * w_b * sizeof(int));
    cudaMalloc((void **)&dev_c, h_a * w_b * sizeof(int));
    cudaMemcpy(dev_a, host_a, sizeof(int) * h_a * w_a, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, sizeof(int) * h_b * w_b, cudaMemcpyHostToDevice);
    
    dim3 block = {32, 32};
    dim3 grid = {w_b / block.x + 1, h_a / block.y + 1};

    kernel_function<<<grid, block>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(host_c, dev_c, sizeof(int) * h_a * w_b, cudaMemcpyDeviceToHost);

    for(int i = 0; i < h_a; i++){
        for(int j = 0; j < w_b; j++){
            printf("%d ", host_c[i * w_b + j]);
        }
        printf("\n");
    }

    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;

}