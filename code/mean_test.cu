#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <float.h>
#include <sys/time.h>

#include "gpu_functions.h"

#define h_a 200
#define w_a 10
#define c_a 1

typedef struct{
    float sum;
    float max;
    float min;
} res;

__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
}

__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void mean_max_min_dev(float *in, res *result, int w, int h, int c){
    
    __shared__ res res_tmp;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadIdx.x == 0){
        res_tmp.max = -FLT_MAX;
        res_tmp.min = FLT_MAX;
        res_tmp.sum = 0;
    }

    __syncthreads();

    if(idx < (w * h * c)){

        atomicMaxFloat(&(res_tmp.max), in[idx]);
        atomicMinFloat(&(res_tmp.min), in[idx]);
        atomicAdd(&(res_tmp.sum), in[idx]);   

        __syncthreads();
    }

    if(threadIdx.x == 0){

        //result += blockIdx.x * sizeof(res);

        result[blockIdx.x].max = res_tmp.max;
        result[blockIdx.x].min = res_tmp.min;
        result[blockIdx.x].sum = res_tmp.sum;
    }
}


__host__ void mean_max_min(float *matrix, float *mean, float *max, float *min, int w, int h, int c){
    float tmp;
    float sum = 0;
    for(int i = 0; i < c; i++){
        for(int j = 0; j < h; j++){
            for(int k = 0; k < w; k++){
                tmp = matrix[k + j * w + i * h * w];
                if(tmp < *min) *min = tmp;
                if(tmp > *max) *max = tmp;
                sum += tmp;
            }
        }
    }
    *mean = sum / (float)(w * h * c);
}

__host__ void mean_normalization(float *matrix_dev, int w, int h, int c){
    float mean = 0;
    float max = -FLT_MAX, min = FLT_MAX;
    float *matrix_host = (float *)malloc(sizeof(float) * w * h * c);
    cudaMemcpy(matrix_host, matrix_dev, sizeof(float) * w * h * c, cudaMemcpyDeviceToHost);
    mean_max_min(matrix_host, &mean, &max, &min, w, h, c);
    dim3 block{(unsigned int)w, (unsigned int)h};
    subtraction_scalar_parametric<<<c, block>>>(matrix_dev, mean, w, h);
    matrix3D_scalar_product<<<c, block>>>(matrix_dev, (2.0 / (max - min)), w, h);
    free(matrix_host);
}

int main(int argc, char **argv){

    srand(time(NULL));

    FILE *ft1, *ft2, *fo, *ft_h;

    ft1 = fopen("time_before_copy.txt", "w");
    ft2 = fopen("time_after_copy.txt", "w");
    ft_h = fopen("time_host.txt", "w");
    fo = fopen("res.txt", "w");

    struct timeval start, partial;
    long int u_sec;

    int number_of_results = ((w_a * h_a * c_a) / 1024 + 1);

    float *host_a = (float *)malloc(sizeof(float) * h_a * w_a * c_a);
    for(int i = 0; i < h_a * w_a * c_a; i++) host_a[i] = i - 1000;
    res *results = (res *)malloc(sizeof(res) * number_of_results);
    int max = -FLT_MAX, min = FLT_MAX, mean, sum;

    float *dev_a;
    res *results_dev;
    cudaMalloc((void **)&dev_a, c_a *h_a * w_a * sizeof(float));
    cudaMalloc((void **)&results_dev, number_of_results * sizeof(res));
    cudaMemcpy(dev_a, host_a, sizeof(float) * h_a * w_a * c_a, cudaMemcpyHostToDevice);
    
    dim3 block = {1024};
    dim3 grid = {(w_a * h_a * c_a) / block.x + 1};

    for(int a = 0; a < 1; a++){

        gettimeofday(&start, NULL);

        mean_max_min_dev<<<grid, block, sizeof(res)>>>(dev_a, results_dev, w_a, h_a, c_a);

        cudaDeviceSynchronize();

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft1, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));

        cudaMemcpy(results, results_dev, sizeof(res) * number_of_results, cudaMemcpyDeviceToHost);

        sum = 0;
        for(int i = 0; i < number_of_results; i++){
            max = (results[i].max > max) ? results[i].max : max;
            max = (results[i].min > min) ? results[i].min : min;
            sum += results[i].sum;
        }
        mean = sum / (w * h * c);

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft2, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
    }

    for(int i = 0; i < number_of_results; i++){
        fprintf(fo, "Max: %f\tMin: %f\tSum: %f\n", results[i].max, results[i].min, results[i].sum); 
    }
    
    printf("Device-> max = %f, min = %f, mean = %f\n\n", max, min, mean);


    for(int a = 0; a < 1; a++){

        gettimeofday(&start, NULL);

        cudaMemcpy(host_a, dev_a, sizeof(float) * w_a * h_a * c_a, cudaMemcpyDeviceToHost);
        mean_max_min(host_a, &mean, &max, &min, w_a, h_a, c_a);

        gettimeofday(&partial, NULL);
        u_sec = (partial.tv_sec - start.tv_sec) * 1000000 + partial.tv_usec - start.tv_usec;
        fprintf(ft_h, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
    }

    printf("Host-> max = %f, min = %f, mean = %f\n\n", max, min, mean);

    free(host_a);
    cudaFree(dev_a);

    fclose(ft1);
    fclose(ft2);
    fclose(ft_h);
    fclose(fo);

    return 0;

}