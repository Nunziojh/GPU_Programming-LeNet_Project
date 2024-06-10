#ifndef MP_SHARED_H
#define MP_SHARED_H

__global__ void matrix_product_shared(float *m1, float *m2, float *output, int out_width, int out_height, int m1_width, int tile); //w_in1 rappresenta la dimensione comune delle due matrici di ingresso
__global__ void matrix_transpose_product_shared(float *m1, float *m2, float *output, int out_width, int out_height, int m1_height, int tile);
__global__ void matrix_product_transpose_shared(float *m1, float *m2, float *output, int out_width, int out_height, int m1_width, int tile); //w_in1 rappresenta la dimensione comune delle due matrici di ingresso

#endif