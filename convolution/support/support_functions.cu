#include "support_functions.h"

__global__ void clean_vector_dev(float *m, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < dim){
        m[idx] = 0;
    }
}

__host__ void clean_vector_host(float *vec, int dim){
    for(int i = 0; i < dim; i++) vec[i] = 0;
}

__host__ void init_values(float *host_a, float *host_k, float *host_c, float *dev_a, float *dev_k, float *dev_c){
    int i;

    for(i = 0; i < INPUT_X * INPUT_Y * INPUT_Z * INPUT_N; i++) host_a[i] = (float)rand() / (float)RAND_MAX;
    for(i = 0; i < KERNEL_X * KERNEL_Y * KERNEL_Z * KERNEL_N; i++) host_k[i] = /*(i < KERNEL_X * KERNEL_Y * KERNEL_Z) ? 1 : 0;i / (KERNEL_X * KERNEL_Y * KERNEL_Z * KERNEL_N);*/(float)rand() / (float)RAND_MAX;
    cudaMemcpy(dev_a, host_a, sizeof(float) * INPUT_X * INPUT_Y * INPUT_Z * INPUT_N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k, host_k, sizeof(float) * KERNEL_X * KERNEL_Y * KERNEL_Z * KERNEL_N, cudaMemcpyHostToDevice);
    clean_vector_dev<<<(unsigned int)((OUT_X * OUT_Y * OUT_Z * OUT_N) / min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024) + 1), min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024)>>>(dev_c, OUT_X * OUT_Y * OUT_Z * OUT_N);
    clean_vector_host(host_c, OUT_X * OUT_Y * OUT_Z * OUT_N);
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

__host__ void debug_print(float *host, float *dev, char filename[], FILE *f_res, int dim_x, int dim_y, int dim_z, int dim_n){
    int i, j, k, l;

    if(dev) cudaMemcpy(host, dev, sizeof(float) * dim_x * dim_y * dim_z * dim_n, cudaMemcpyDeviceToHost);
    if(filename) f_res = fopen(filename, "w");
    else f_res = stdout;
    for(i = 0; i < dim_n; i++)
    {
        fprintf(f_res, "Element number: %d\n", i);
        for(j = 0; j < dim_z; j++)
        {
            for(k = 0; k < dim_y; k++)
            {
                for(l = 0; l < dim_x; l++)
                {
                    fprintf(f_res, "%.3f ", host[(((i * dim_z + j) * dim_y + k) * dim_x + l)]);
                }
                fprintf(f_res, "\n");
            }
            fprintf(f_res, "\t-----------\n");
        }
    }    
    if(filename) fclose(f_res);
}