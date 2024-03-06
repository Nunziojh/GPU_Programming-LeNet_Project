#include <stdlib.h>
#include <cuda_runtime.h>

#define H 32
#define W 32
#define KERNEL_DIM 5
#define KERNEL_NUM 22
#define POOLING_WINDOW_SIZE 2
#define LEARNING_RATE 0.5


__global__ void convolution(float *in, float *out, float *kernel, int new_h, int new_w, int padding, int stride);
__global__ void convolution3D(float *in, float *out, float *kernel, int new_h, int new_w, int padding, int stride, int kernel_dim);
__global__ void avg_pooling(float *in, float *out, int h, int w, int new_h, int new_w, int stride);
__global__ void inverse_avg_pooling(float *in, float *out, float *m, int w_in, int h_in, int new_w, int new_h, int stride);
__global__ void matrix_product(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1);
__global__ void matrix_transpose_product(float *in1, float *in2, float *out, int w_out, int h_out, int h_in1);
__global__ void matrix_product_transpose(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1);
__global__ void matrix_dot_product(float *in1, float *in2, float *out, int w, int h);
__global__ void matrix_scalar_product(float *io, float scalar, int w, int h, int stampa);
__global__ void tanh(float *in, int w, int h, int stampa);
__global__ void exponential(float *in, int len);
__global__ void subtraction(float *out, float *in1, float*in2, int dim);
__global__ void scalar_subtraction(float *out, float *in, int w, int h);
__global__ void transpose(float *out, float *in, int w_out, int h_out);
__global__ void clean_vector(float *m, int dim);

