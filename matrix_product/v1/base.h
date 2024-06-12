#ifndef MP_BASE_H
#define MP_BASE_H

__global__ void matrix_product(float *m1, float *m2, float *output, int out_width, int out_height, int common_dim);
__global__ void matrix_transpose_product(float *m1, float *m2, float *output, int out_width, int out_height, int m1_height);
__global__ void matrix_product_transpose(float *m1, float *m2, float *output, int out_width, int out_height, int m1_width);
__global__ void matrix_dot_product(float *m1, float *m2, float *output, int width, int height, int depth);
__global__ void matrix_scalar_product(float *in_out, float scalar, int dim);
__global__ void matrix3D_scalar_product(float *in_out, float scalar, int width, int height, int depth);

#endif