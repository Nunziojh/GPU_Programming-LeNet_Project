#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

#define INPUT_DIM 20        // DEVE ESSERE MINORE DI 32
#define KERNEL_DIM 5
#define PADDING 0
#define STRIDE 1
#define OUTPUT_DIM (INPUT_DIM + 2 * PADDING - KERNEL_DIM) / STRIDE + 1

__global__ void convolution(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f, int stride_f){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_dim && idy < out_dim){

        int i, j;

        float tmp = 0;
        float val;

        int new_idx = idx * stride_f - padding_f;
        int new_idy = idy * stride_f - padding_f;

        int offset = kernel_dim * kernel_dim - 1;

        for(i = 0; i < kernel_dim * kernel_dim; i++){
            val = in[(new_idy + (i / kernel_dim)) * in_dim + new_idx + (i % kernel_dim)];
            tmp += kernel[offset - i] * val;
        }
        out[idy * out_dim + idx] = tmp;
    }
}

__global__ void convolution_2for(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f, int stride_f){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_dim && idy < out_dim){

        int i, j;

        float tmp = 0;
        float val;

        int new_idx = idx * stride_f - padding_f;
        int new_idy = idy * stride_f - padding_f;

        int offset = kernel_dim * kernel_dim - 1;

        for(i = 0; i < kernel_dim; i++){
            for(j = 0; j < kernel_dim; j++){
                val = in[(new_idy + i) * in_dim + new_idx + j];
                tmp += kernel[offset - (i * kernel_dim + j)] * val;
            }
        }
        out[idy * out_dim + idx] = tmp;
    }
}

__global__ void convolution_controllo(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f, int stride_f){
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
                tmp += kernel[(kernel_dim * kernel_dim - 1) - (i * kernel_dim + j)] * val;
            }
        }
        out[idy * out_dim + idx] = tmp;
    }
}

__global__ void convolution_shared(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f, int stride_f){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float data[];
    __shared__ float filter[];

    int index_data = idy * in_dim + idx;
    if(idx < in_dim && idy < out_dim){
        data[index_data] = in[index_data];
    }

    int index_filter = idy * kernel_dim + idx;
    if(idx < kernel_dim && idy < kernel_dim){
        filter[index_filter] = kernel[index_filter];
    }

    __syncthreads();

    if(idx < out_dim && idy < out_dim){

        int i, j;

        float tmp = 0;
        float val;

        int new_idx = idx * stride_f - padding_f;
        int new_idy = idy * stride_f - padding_f;

        int offset = kernel_dim * kernel_dim - 1;
        for(i = 0; i < kernel_dim * kernel_dim; i++){
            val = data[(new_idy + (i / kernel_dim)) * in_dim + new_idx + (i % kernel_dim)];
            tmp += filter[offset - i] * val;
        }
        out[idy * out_dim + idx] = tmp;
    }
}



int main(int argc, char **argv){

    srand(time(NULL));

    FILE *ft1, *ft2, *fo, *ft1_c, *ft2_c, *fo_c, *ft1_s, *ft2_s, *fo_s, *ft1_2f, *ft2_2f, *fo_2f;

    ft1 = fopen("time_before_copy.txt", "w");
    ft2 = fopen("time_after_copy.txt", "w");
    fo = fopen("res.txt", "w");
    ft1_c = fopen("time_before_copy_c.txt", "w");
    ft2_c = fopen("time_after_copy_c.txt", "w");
    fo_c = fopen("res_c.txt", "w");
    ft1_s = fopen("time_before_copy_s.txt", "w");
    ft2_s = fopen("time_after_copy_s.txt", "w");
    fo_s = fopen("res_s.txt", "w");
    ft1_2f = fopen("time_before_copy_s.txt", "w");
    ft2_2f = fopen("time_after_copy_s.txt", "w");
    fo_2f = fopen("res_s.txt", "w");

    struct timeval start, partial;
    long int u_sec;

    float *host_a = (float *)malloc(sizeof(float) * INPUT_DIM * INPUT_DIM);
    for(int i = 0; i < INPUT_DIM * INPUT_DIM; i++) host_a[i] = i;
    float *host_k = (float *)malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM);
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM; i++) host_k[i] = i;
    float *host_c = (float *)malloc(sizeof(float) * OUTPUT_DIM * OUTPUT_DIM);

    float *dev_a, *dev_k, *dev_c;
    cudaMalloc((void **)&dev_a, INPUT_DIM * INPUT_DIM * sizeof(float));
    cudaMalloc((void **)&dev_b, KERNEL_DIM * KERNEL_DIM * sizeof(float));
    cudaMalloc((void **)&dev_c, OUTPUT_DIM * OUTPUT_DIM * sizeof(float));
    cudaMemcpy(dev_a, host_a, sizeof(float) * INPUT_DIM * INPUT_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, sizeof(float) * KERNEL_DIM * KERNEL_DIM, cudaMemcpyHostToDevice);
    
    dim3 block = {max(OUTPUT_DIM, INPUT_DIM), max(OUTPUT_DIM, INPUT_DIM)};
    dim3 grid = {w_c / block.x + 1, h_c / block.y + 1};

    //Prima versione con singolo ciclo for
    for(int a = 0; a < 20000; a++){

        gettimeofday(&start, NULL);

        convolution<<<grid, block>>>(dev_a, dev_c, dev_k, INPUT_DIM, OUTPUT_DIM, KERNEL_DIM, PADDING, STRIDE);

        cudaDeviceSynchronize();

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft1, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));

        cudaMemcpy(host_c, dev_c, sizeof(float) * OUTPUT_DIM * OUTPUT_DIM, cudaMemcpyDeviceToHost);

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft2, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
    }

    for(int i = 0; i < h_c; i++){
        for(int j = 0; j < w_c; j++){
            fprintf(fo, "%.2f ", host_c[i * OUTPUT_DIM + j]);
        }
        fprintf(fo, "\n");
    }
    fprintf(fo, "\n");

    //Seconda versione con doppio ciclo for
    for(int a = 0; a < 20000; a++){

        gettimeofday(&start, NULL);

        convolution_2for<<<grid, block>>>(dev_a, dev_c, dev_k, INPUT_DIM, OUTPUT_DIM, KERNEL_DIM, PADDING, STRIDE);

        cudaDeviceSynchronize();

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft1_2f, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));

        cudaMemcpy(host_c, dev_c, sizeof(float) * OUTPUT_DIM * OUTPUT_DIM, cudaMemcpyDeviceToHost);

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft2_2f, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
    }

    for(int i = 0; i < h_c; i++){
        for(int j = 0; j < w_c; j++){
            fprintf(fo_2f, "%.2f ", host_c[i * OUTPUT_DIM + j]);
        }
        fprintf(fo_2f, "\n");
    }
    fprintf(fo_2f, "\n");

    //terza verisone con controllo sulle dimensioni
    for(int a = 0; a < 20000; a++){

        gettimeofday(&start, NULL);

        convolution_controllo<<<grid, block>>>(dev_a, dev_c, dev_k, INPUT_DIM, OUTPUT_DIM, KERNEL_DIM, PADDING, STRIDE);

        cudaDeviceSynchronize();

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft1_c, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));

        cudaMemcpy(host_c, dev_c, sizeof(float) * OUTPUT_DIM * OUTPUT_DIM, cudaMemcpyDeviceToHost);

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft2_c, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
    }

    for(int i = 0; i < h_c; i++){
        for(int j = 0; j < w_c; j++){
            fprintf(fo_c, "%.2f ", host_c[i * OUTPUT_DIM + j]);
        }
        fprintf(fo_c, "\n");
    }
    fprintf(fo_c, "\n");

    //Quarta versione con shared memory
    int shared_mem_dim = (INPUT_DIM * INPUT_DIM + KERNEL_DIM * KERNEL_DIM) * sizeof(float);
    for(int a = 0; a < 20000; a++){

        gettimeofday(&start, NULL);

        convolution_shared<<<grid, block, shared_mem_dim>>>(dev_a, dev_c, dev_k, INPUT_DIM, OUTPUT_DIM, KERNEL_DIM, PADDING, STRIDE);

        cudaDeviceSynchronize();

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft1_s, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));

        cudaMemcpy(host_c, dev_c, sizeof(float) * OUTPUT_DIM * OUTPUT_DIM, cudaMemcpyDeviceToHost);

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft2_s, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
    }

    for(int i = 0; i < h_c; i++){
        for(int j = 0; j < w_c; j++){
            fprintf(fo_s, "%.2f ", host_c[i * OUTPUT_DIM + j]);
        }
        fprintf(fo_s, "\n");
    }
    fprintf(fo_s, "\n");



    free(host_a);
    free(host_k);
    free(host_c);
    cudaFree(dev_a);
    cudaFree(dev_k);
    cudaFree(dev_c);

    fclose(ft1);
    fclose(ft2);
    fclose(fo);
    fclose(ft1_c);
    fclose(ft2_c);
    fclose(fo_c);
    fclose(ft1_s);
    fclose(ft2_s);
    fclose(fo_s);
    fclose(ft1_2f);
    fclose(ft2_2f);
    fclose(fo_2f);

    return 0;

}