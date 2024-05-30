#include "optimized.h"

__global__ void convolution_optimized(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f/*, int stride_f*/){ //in_dim = dimensioneffettiva dell'ingresso
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ float s_m[];
    float *filter = &s_m[0];
    float *data = &s_m[kernel_dim * kernel_dim]; //bisogna allocare abbastanza spazio da contenere sia l'ingresso che il padding associato all'ingresso

    int new_in_dim = in_dim + 2 * padding_f;
    if(idx < new_in_dim && idy < new_in_dim){//Assicurarsi che sia necessario
        data[idy * new_in_dim + idx] = 0;
    }
    if(idx < in_dim && idy < in_dim){
        data[(idy + padding_f) * new_in_dim + idx + padding_f] = in[idy * in_dim + idx];
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

        int new_idx = idx - padding_f;
        int new_idy = idy - padding_f;
        data += new_idy * new_in_dim + new_idx;

        for(i = 0; i < kernel_dim; i++){
            for(j = 0; j < kernel_dim; j++){
                val = data[j];
                tmp += filter[j] * val;
            }
            filter += kernel_dim;
            data += new_in_dim;
        }
        out[idy * out_dim + idx] += tmp;
    }
}