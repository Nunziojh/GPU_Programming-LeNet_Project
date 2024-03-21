#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define h_a 3
#define w_a 4
#define h_b 2
#define w_b 4

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

__global__ void matrix_transpose_product(int *in1, int *in2, int *out, int w_out, int h_out, int h_in1){
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

__global__ void matrix_product_transpose(int *in1, int *in2, int *out, int w_out, int h_out, int w_in1){
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

int main(int argc, char **argv){

    srand(time(NULL));

    //int *host_a = (int *)malloc(sizeof(int) * h_a * w_a);
    //for(int i = 0; i < h_a * w_a; i++) host_a[i] = i;
    int host_a[] = {3, 4, 4, 3, 2, 2, 2, 2, 3, 1, 3, 1};
    //int *host_b = (int *)malloc(sizeof(int) * h_b * w_b);
    //for(int i = 0; i < h_b * w_b; i++) host_b[i] = i;
    int host_b[] = {4, 5, 6, 7, 7, 6, 5, 4};
    int *host_c = (int *)malloc(sizeof(int) * w_a * w_b);

    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, h_a * w_a * sizeof(int));
    cudaMalloc((void **)&dev_b, h_b * w_b * sizeof(int));
    cudaMalloc((void **)&dev_c, h_a * h_b * sizeof(int));
    cudaMemcpy(dev_a, host_a, sizeof(int) * h_a * w_a, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, sizeof(int) * h_b * w_b, cudaMemcpyHostToDevice);
    
    dim3 block = {32, 32};
    dim3 grid = {w_b / block.x + 1, h_a / block.y + 1};

    matrix_product_transpose<<<grid, block>>>(dev_a, dev_b, dev_c, h_b, h_a, w_a);

    cudaMemcpy(host_c, dev_c, sizeof(int) * h_a * h_b, cudaMemcpyDeviceToHost);

    for(int i = 0; i < h_a; i++){
        for(int j = 0; j < h_b; j++){
            printf("%d ", host_c[i * h_b + j]);
        }
        printf("\n");
    }

    //free(host_a);
    //free(host_b);
    free(host_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;

}