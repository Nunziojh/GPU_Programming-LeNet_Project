#include "shared.h"

__global__ void convolution_shared(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim/*, int padding_f, int stride_f*/){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ float s_m[];
    float *data = &s_m[0];
    float *filter = &s_m[in_dim * in_dim];

    int index_data = idy * in_dim + idx;
    if(idx < in_dim && idy < in_dim){
        data[index_data] = in[index_data];
    }

    int index_filter = idy * kernel_dim + idx;
    int offset = kernel_dim * kernel_dim - 1;
    if(idx < kernel_dim && idy < kernel_dim){
        filter[index_filter] = kernel[offset - index_filter];
    }

    __syncthreads();

    if(idx < out_dim && idy < out_dim){

        int i, j;

        float tmp = 0;
        float val;

        data += idy * in_dim + idx;

        for(i = 0; i < kernel_dim; i++){
            for(j = 0; j < kernel_dim; j++){
                val = data[j];
                tmp += filter[j] * val;
            }
            filter += kernel_dim;
            data += in_dim;
        }
        out[idy * out_dim + idx] += tmp;
    }
}