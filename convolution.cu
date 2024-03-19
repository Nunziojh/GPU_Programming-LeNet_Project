#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define h_k 10
#define w_k 10
#define h_m 5
#define w_m 5
#define stride 1
#define padding 9


__global__ void convolution3D(int *in, int *out, int *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f, int stride_f){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_dim && idy < out_dim){

        int i, j;

        int tmp = 0;
        int val;

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



int main(int argc, char **argv){

    srand(time(NULL));
    int *host_matrix = (int *)malloc(sizeof(int) * w_m * h_m);
    for(int i = 0; i < w_m * h_m; i++) host_matrix[i] = 1;

    int *host_kernel = (int *)malloc(sizeof(int) * w_k * h_k);
    for(int i = 0; i < w_k * h_k; i++) host_kernel[i] = 1;

    //for(int i = 0; i< w_k * h_k; i++) host_kernel[i] = 1;
    //int *host_input = (int *)malloc(sizeof(int) * h_m * w_m);
    
    int new_h = (h_m + 2 * padding - h_k) / stride + 1;
    int new_w = (w_m + 2 * padding - w_k) / stride + 1;
    int *host_res = (int *)malloc(sizeof(int) * new_h * new_w);

    int *dev_input, *dev_kernel, *dev_res;
    cudaMalloc((void **)&dev_input, h_m * w_m * sizeof(int));
    cudaMalloc((void **)&dev_kernel, h_k * w_k * sizeof(int));
    cudaMalloc((void **)&dev_res, new_h * new_w * sizeof(int));
    cudaMemcpy(dev_input, host_matrix, sizeof(int) * h_m * w_m, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel, host_kernel, sizeof(int) * h_k * w_k, cudaMemcpyHostToDevice);

    //cudaMemcpyToSymbol(kernel, &host_kernel, sizeof(int) * w_k * h_k);
    
    dim3 block = {32, 32};
    dim3 grid = {new_w / block.x + 1, new_h / block.y + 1};

    convolution3D<<<grid, block>>>(dev_input, dev_res, dev_kernel, h_m, new_h, h_k, padding, stride);

    cudaMemcpy(host_res, dev_res, sizeof(int) * new_h * new_w, cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for(int i = 0; i < h_m; i++){
        for(int j = 0; j < w_m; j++){
            printf("%d\t", host_matrix[i * w_m + j]);
        }
        printf("\n");
    }

    printf("Kernel:\n");
    for(int i = 0; i < h_k; i++){
        for(int j = 0; j < w_k; j++){
            printf("%d\t", host_kernel[i * w_k + j]);
        }
        printf("\n");
    }

    printf("Output:\n");
    for(int i = 0; i < new_h; i++){
        for(int j = 0; j < new_w; j++){
            printf("%d\t", host_res[i * new_w + j]);
        }
        printf("\n");
    }

    free(host_res);
    free(host_kernel);
    free(host_matrix);
    cudaFree(dev_input);
    cudaFree(dev_kernel);
    cudaFree(dev_res);

    return 0;

}