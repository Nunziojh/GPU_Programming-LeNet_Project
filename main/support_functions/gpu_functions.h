#ifndef GPUFUNCTIONS_H
#define GPUFUNCTIONS_H

__global__ void tanh(float *in, int w, int h, int d);
__global__ void exponential(float *in, int len);
__global__ void subtraction(float *out, float *in1, float*in2, int dim);
__global__ void scalar_subtraction(float *out, float *in, int w, int h, int d);
__global__ void subtraction_scalar_parametric(float *io, float scalar, int dim);
__global__ void transpose(float *out, float *in, int w_out, int h_out);
__global__ void clean_vector(float *m, int dim);

#endif