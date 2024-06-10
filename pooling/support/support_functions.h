#ifndef SUPPORT_MP_FUNCTIONS_H
#define SUPPORT_MP_FUNCTIONS_H

#include "../pooling_functions.h"

__host__ void init_values(float *host_m1, float *host_m2, float *dev_m1, float *dev_m2);

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

__host__ void debug_print(float *host, float *dev, char filename[], FILE *f_res, int dim_x, int dim_y, int dim_z);
#endif