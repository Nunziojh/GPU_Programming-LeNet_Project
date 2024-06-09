#ifndef SUPPORT_FUNCTIONS_H
#define SUPPORT_FUNCTIONS_H

#include "../functions.h"

__global__ void clean_vector_dev(float *m, int dim);
__host__ void clean_vector_host(float *m, int dim);
__host__ void init_values(float *host_a, float *host_k, float *host_c, float *dev_a, float *dev_k, float *dev_c);

#ifdef __linux__
__host__ void start_timer(struct timeval *start);
#else
__host__ void start_timer(struct timespec *start);
#endif

#ifdef __linux__
__host__ long int stop_timer(struct timeval *start, struct timeval *stop);
#else
__host__ long int stop_timer(struct timespec *start, struct timespec *stop);
#endif

__host__ void debug_print(float *host, float *dev, char filename[], FILE *f_res, int dim_x, int dim_y, int dim_z, int dim_n);

#endif