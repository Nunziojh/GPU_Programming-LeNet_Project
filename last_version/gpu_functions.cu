#include <stdio.h>

#include "gpu_functions.h"

__global__ void convolution(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f, int stride_f){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ float s_m[];
    float *data = &s_m[kernel_dim * kernel_dim];
    float *filter = &s_m[0];

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
        out[idy * out_dim + idx] = tmp;
    }
}

__global__ void convolution3D(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f, int stride_f){ //in_dim = dimensioneffettiva dell'ingresso
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ float s_m[];
    float *filter = &s_m[0];
    float *data = &s_m[kernel_dim * kernel_dim];

    int new_in_dim = in_dim + 2 * padding_f;
    if(idx < new_in_dim && idy < new_in_dim){
        data[idy * new_in_dim + idx] = 0;
    }

    __syncthreads();
    if(idx < in_dim && idy < in_dim){
        data[(idy + padding_f) * (new_in_dim) + idx + padding_f] = in[idy * in_dim + idx];
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
        data += idy * new_in_dim + idx;

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

__global__ void avg_pooling(float *in, float *out, int h, int w, int new_h, int new_w, int stride){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;

        float tmp = 0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;

        for(i = new_idy; i < (new_idy + POOLING_WINDOW_SIZE); i++){
            for(j = new_idx; j < (new_idx + POOLING_WINDOW_SIZE); j++){
                tmp += (i >= h || j >= w) ? 0 : in[i * w + j];
            }
        }

        out[idy * new_w + idx] = tmp / (float)(POOLING_WINDOW_SIZE * POOLING_WINDOW_SIZE);
    }
}

__global__ void inverse_avg_pooling(float *in, float *out, float *m, int w_in, int h_in, int new_w, int new_h, int stride){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_in && idy < h_in){

        int i, j;

        float tot = 0.0;

        int new_idx = idx * stride;
        int new_idy = idy * stride;

        float tmp[4];
        for(i = 0; i < POOLING_WINDOW_SIZE; i++){
            for(j = 0; j < POOLING_WINDOW_SIZE; j++){
                tmp[i * POOLING_WINDOW_SIZE + j] = m[(new_idy + i) * new_w + new_idx + j];
            }
        }
       
        for(i = 0; i < POOLING_WINDOW_SIZE; i++){
            for(j = 0; j < POOLING_WINDOW_SIZE; j++){
                tot += (i >= new_h || j >= new_w) ? 0 : tmp[i * POOLING_WINDOW_SIZE + j];
            }
        }

        if(tot == 0.0){
            for(i = new_idy; i < (new_idy + POOLING_WINDOW_SIZE); i++){
                for(j = new_idx; j < (new_idx + POOLING_WINDOW_SIZE); j++){
                    out[i * new_w + j] = 0.0;
                }
            }
        }
        else {
            for(i = 0; i < POOLING_WINDOW_SIZE; i++){
                for(j = 0; j < POOLING_WINDOW_SIZE; j++){
                    out[(i + new_idy) * new_w + j + new_idx] = tmp[i * POOLING_WINDOW_SIZE + j] / tot * in[idy * w_in + idx];
                }
            }
        }

    }
}

__global__ void matrix_product(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1, int piastrella) { //w_in1 rappresenta la dimensione comune delle due matrici di ingresso
    extern __shared__ float s_m[]; //Shared memory equals to piastrella * piastrella * 2
    float *sA = &s_m[0];
    float *sB = &s_m[piastrella * piastrella];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = threadIdx.y * piastrella + threadIdx.x;
    float Cvalue = 0.0;

    for (int ph = 0; ph < (((w_in1 - 1) / piastrella) + 1); ph++) {
        if ((Row < h_out) && (threadIdx.x + (ph * piastrella)) < w_in1) {
            sA[index] = in1[(Row * w_in1) + threadIdx.x + (ph * piastrella)];
        } else {
            sA[index] = 0.0;
        }
        if (Col < w_out && (threadIdx.y + ph * piastrella) < w_in1) {
            sB[index] = in2[(threadIdx.y + ph * piastrella) * w_out + Col];
        } else {
            sB[index] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < piastrella; ++j) {
            Cvalue += sA[threadIdx.y * piastrella + j] * sB[j * piastrella + threadIdx.x];
        }
        /***
         * Aggiunta di recente. Potrebbe succedere che il ciclo for interno, che esegue il prodotto, sia troppo veloce 
         * e che il thread inizi l'iterazione successiva del ciclo e che sovrascriva i dati che gli altri thread andranno a leggere??
        */
        __syncthreads();
    }
    if (Row < h_out && Col < w_out) {
        out[Row * w_out + Col] = Cvalue;
    }
}

__global__ void matrix_transpose_product(float *in1, float *in2, float *out, int w_out, int h_out, int h_in1, int piastrella) {
    extern __shared__ float s_m[]; //Shared memory equals to piastrella * piastrella * 2
    float *sA = &s_m[0];
    float *sB = &s_m[piastrella * piastrella];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = threadIdx.y * piastrella + threadIdx.x;
    float Cvalue = 0.0;
    sA[index] = 0.0;
    sB[index] = 0.0;

    for (int ph = 0; ph < (((h_in1 - 1) / piastrella) + 1); ph++) {
        if ((Row < h_out) && (threadIdx.x + (ph * piastrella)) < h_in1) {
            sA[index] = in1[(threadIdx.x + ph * piastrella) * h_out + Row];
        } else {
            sA[index] = 0.0;
        }
        if (Col < w_out && (threadIdx.y + ph * piastrella) < h_in1) {
            sB[index] = in2[(threadIdx.y + ph * piastrella) * w_out + Col];
        } else {
            sB[index] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < piastrella; ++j) {
            Cvalue += sA[threadIdx.y * piastrella + j] * sB[j * piastrella + threadIdx.x];
        }

        /***
         * Stesso ragionamento fatto per la matric_product.
        */
        __syncthreads();
    }
    if (Row < h_out && Col < w_out) {
        out[Row * w_out + Col] = Cvalue;
    }
}

__global__ void matrix_product_transpose(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1, int piastrella) { //w_in1 rappresenta la dimensione comune delle due matrici di ingresso
    extern __shared__ float s_m[]; //Shared memory equals to piastrella * piastrella * 2
    float *sA = &s_m[0];
    float *sB = &s_m[piastrella * piastrella];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = threadIdx.y * piastrella + threadIdx.x;
    float Cvalue = 0.0;

    for (int ph = 0; ph < (((w_in1 - 1) / piastrella) + 1); ph++) {
        if ((Row < h_out) && (threadIdx.x + (ph * piastrella)) < w_in1) {
            sA[index] = in1[(Row * w_in1) + threadIdx.x + (ph * piastrella)];
        } else {
            sA[index] = 0.0;
        }
        if (Col < w_out && (threadIdx.y + ph * piastrella) < w_in1) {
            sB[index] = in2[(Col * w_in1) + threadIdx.y + (ph * piastrella)];
        } else {
            sB[index] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < piastrella; ++j) {
            Cvalue += sA[threadIdx.y * piastrella + j] * sB[j * piastrella + threadIdx.x];
        }

        /***
         * Stesso ragionamento fatto per la matrix_product.
        */
        __syncthreads();
    }
    if (Row < h_out && Col < w_out) {
        out[Row * w_out + Col] = Cvalue;
    }
}

__global__ void matrix_dot_product(float *in1, float *in2, float *out, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx;
        out[index] = in1[index] * in2[index];
    }
}

__global__ void matrix_scalar_product(float *io, float scalar, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx;
        io[index] = io[index] * scalar;
    }
}

__global__ void matrix3D_scalar_product(float *io, float scalar, int w, int h){
    int c = blockIdx.x;
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx + c * w * h;
        io[index] = io[index] * scalar;
    }
}

__global__ void tanh(float *in, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        float val = in[idy * w + idx];
        float v = expf(2 * val);

        in[idy * w + idx] = (v - 1) / (v + 1);
    }
}

__global__ void exponential(float *in, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < len){
        in[idx] = expf(in[idx]);
    }
}

__global__ void subtraction(float *out, float *in1, float*in2, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < dim){
        out[idx] = in1[idx] - in2[idx]; 
    }
}

__global__ void scalar_subtraction(float *out, float *in, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx;
        out[index] = 1 - in[index];
    }
}

__global__ void subtraction_scalar_parametric(float *io, float scalar, int w, int h){
    int c = blockIdx.x;
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    if(idx < w && idy < h){
        int index = idy * w + idx + c * w * h;
        io[index] = io[index] - scalar;
    }
}

__global__ void transpose(float *out, float *in, int w_out, int h_out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < w_out && idy < h_out){
        out[idy * w_out + idx] = in[idx * h_out + idy];
    }
}

__global__ void clean_vector(float *m, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < dim){
        m[idx] = 0;
    }
}