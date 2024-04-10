#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

#define h_a 8
#define w_a 4
#define h_b 8
#define w_b 3
#define h_c w_a
#define w_c w_b

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
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < h_out && Col < w_out) {
        out[Row * w_out + Col] = Cvalue;
    }
}

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
            Cvalue += sA[threadIdx.y][j] * sB[threadIdx.x][j];//sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < h_out && Col < w_out) {
        out[Row * w_out + Col] = Cvalue;
    }
}


__global__ void kernel_function(float *a, float *b, float *c, int w, int h, int k){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){

        extern __shared__ float mem[];

        float tmp = 0;
        a = a + idy * k;
        for(int i = 0, j = idx; i < k; i++, j += w){
            tmp += a[i] * b[j];
        }

        c[idy * w + idx] = tmp;
    }
}
/*
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
}*/

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

    FILE *ft1, *ft2, *fo;

    ft1 = fopen("time_before_copy.txt", "w");
    ft2 = fopen("time_after_copy.txt", "w");
    fo = fopen("res.txt", "w");

    struct timeval start, partial;
    long int u_sec;

    float *host_a = (float *)malloc(sizeof(float) * h_a * w_a);
    for(int i = 0; i < h_a * w_a; i++) host_a[i] = i;
    //int host_a[] = {3, 4, 4, 3, 2, 2, 2, 2, 3, 1, 3, 1};
    float *host_b = (float *)malloc(sizeof(float) * h_b * w_b);
    for(int i = 0; i < h_b * w_b; i++) host_b[i] = i;
    //int host_b[] = {4, 5, 6, 7, 7, 6, 5, 4};
    float *host_c = (float *)malloc(sizeof(float) * h_c * w_c);


    for(int i = 0; i < h_a; i++){
        for(int j = 0; j < w_a; j++){
            fprintf(stdout, "%.2f ", host_a[i * w_a + j]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");

    for(int i = 0; i < h_b; i++){
        for(int j = 0; j < w_b; j++){
            fprintf(stdout, "%.2f ", host_b[i * w_b + j]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");

    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, h_a * w_a * sizeof(float));
    cudaMalloc((void **)&dev_b, h_b * w_b * sizeof(float));
    cudaMalloc((void **)&dev_c, w_c * h_c * sizeof(float));
    cudaMemcpy(dev_a, host_a, sizeof(float) * h_a * w_a, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, sizeof(float) * h_b * w_b, cudaMemcpyHostToDevice);
    
    dim3 block = {32, 32};
    dim3 grid = {w_c / block.x + 1, h_c / block.y + 1};

    for(int a = 0; a < 1; a++){

        gettimeofday(&start, NULL);

        //kernel_function<<<grid, block>>>(dev_a, dev_b, dev_c, w_b, h_a, w_a);
        matrix_transpose_product<<<grid, block>>>(dev_a, dev_b, dev_c, w_c, h_c, h_a, 32);

        cudaDeviceSynchronize();

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft1, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));

        cudaMemcpy(host_c, dev_c, sizeof(float) * w_c * h_c, cudaMemcpyDeviceToHost);

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft2, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
        
        for(int i = 0; i < h_c; i++){
            for(int j = 0; j < w_c; j++){
                fprintf(fo, "%.2f ", host_c[i * w_c + j]);
            }
            fprintf(fo, "\n");
        }
        fprintf(fo, "\n");
    }

    //free(host_a);
    //free(host_b);
    free(host_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    fclose(ft1);
    fclose(ft2);
    fclose(fo);

    return 0;

}