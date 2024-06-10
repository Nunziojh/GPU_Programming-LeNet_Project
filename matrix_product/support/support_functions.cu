#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "support_functions.h"

__host__ void init_values(float *host_m1, float *host_m2, float *dev_m1, float *dev_m2){
    int i;

    for(i = 0; i < M1_W * M1_H * M1_Z; i++) host_m1[i] = (float)rand() / (float)RAND_MAX;
    for(i = 0; i < M2_W * M2_H * M2_Z; i++) host_m2[i] = /*(i < KERNEL_X * KERNEL_Y * KERNEL_Z) ? 1 : 0;i / (KERNEL_X * KERNEL_Y * KERNEL_Z * KERNEL_N);*/(float)rand() / (float)RAND_MAX;
    cudaMemcpy(dev_m1, host_m1, sizeof(float) * M1_W * M1_H * M1_Z, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_m2, host_m2, sizeof(float) * M2_W * M2_H * M2_Z, cudaMemcpyHostToDevice);
}

#ifdef __linux__
__host__ void start_timer(struct timeval *start){
    gettimeofday(start, NULL);
}
#else
__host__ void start_timer(struct timespec *start){
    timespec_get(start, TIME_UTC);
}
#endif

#ifdef __linux__
__host__ long int stop_timer(struct timeval *start, struct timeval *stop){
    gettimeofday(stop, NULL);
    return (stop->tv_sec - start->tv_sec) * 1000000 + stop->tv_usec - start->tv_usec;
}
#else
__host__ long int stop_timer(struct timespec *start, struct timespec *stop){
    timespec_get(stop, TIME_UTC);
    return (long int)(((stop->tv_sec - start->tv_sec) * 1e9 + (stop->tv_nsec - start->tv_nsec)) / 1000);
}
#endif

__host__ void debug_print(float *host, float *dev, char filename[], FILE *f_res, int dim_x, int dim_y, int dim_z){
    int j, k, l;

    if(dev) cudaMemcpy(host, dev, sizeof(float) * dim_x * dim_y * dim_z, cudaMemcpyDeviceToHost);
    if(filename) f_res = fopen(filename, "w");
    else f_res = stdout;
    for(j = 0; j < dim_z; j++)
    {
        for(k = 0; k < dim_y; k++)
        {
            for(l = 0; l < dim_x; l++)
            {
                fprintf(f_res, "%.3f ", host[((j * dim_y + k) * dim_x + l)]);
            }
            fprintf(f_res, "\n");
        }
        fprintf(f_res, "\t-----------\n");
    }
    if(filename) fclose(f_res);
}