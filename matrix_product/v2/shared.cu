#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void matrix_product_shared(float *m1, float *m2, float *output, int out_width, int out_height, int m1_width, int tile) { //w_in1 rappresenta la dimensione comune delle due matrici di ingresso
    extern __shared__ float s_m[]; //Shared memory equals to piastrella * piastrella * 2
    float *sA = &s_m[0];
    float *sB = &(s_m[tile * tile]);

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = threadIdx.y * tile + threadIdx.x;
    float Cvalue = 0.0;

    for (int ph = 0; ph < (((m1_width - 1) / tile) + 1); ph++) {
        if ((Row < out_height) && (threadIdx.x + (ph * tile)) < m1_width) {
            sA[index] = m1[(Row * m1_width) + threadIdx.x + (ph * tile)];
        } else {
            sA[index] = 0.0;
        }
        if (Col < out_width && (threadIdx.y + ph * tile) < m1_width) {
            sB[index] = m2[(threadIdx.y + ph * tile) * out_width + Col];
        } else {
            sB[index] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < tile; ++j) {
            Cvalue += sA[threadIdx.y * tile + j] * sB[j * tile + threadIdx.x];
        }
        /***
         * Aggiunta di recente. Potrebbe succedere che il ciclo for interno, che esegue il prodotto, sia troppo veloce 
         * e che il thread inizi l'iterazione successiva del ciclo e che sovrascriva i dati che gli altri thread andranno a leggere??
        */
        // __syncthreads();
    }
    if (Row < out_height && Col < out_width) {
        output[Row * out_width + Col] = Cvalue;
    }
}

__global__ void matrix_transpose_product_shared(float *m1, float *m2, float *output, int out_width, int out_height, int m1_height, int tile) {
    extern __shared__ float s_m[]; //Shared memory equals to piastrella * piastrella * 2
    float *sA = &s_m[0];
    float *sB = &s_m[tile * tile];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = threadIdx.y * tile + threadIdx.x;
    float Cvalue = 0.0;

    for (int ph = 0; ph < (((m1_height - 1) / tile) + 1); ph++) {
        if ((Row < out_height) && (threadIdx.x + (ph * tile)) < m1_height) {
            sA[index] = m1[(threadIdx.x + ph * tile) * out_height + Row];
        } else {
            sA[index] = 0.0;
        }
        if (Col < out_width && (threadIdx.y + ph * tile) < m1_height) {
            sB[index] = m2[(threadIdx.y + ph * tile) * out_width + Col];
        } else {
            sB[index] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < tile; ++j) {
            Cvalue += sA[threadIdx.y * tile + j] * sB[j * tile + threadIdx.x];
        }

        /***
         * Stesso ragionamento fatto per la matric_product.
        */
       // __syncthreads();
    }
    if (Row < out_height && Col < out_width) {
        output[Row * out_width + Col] = Cvalue;
    }
}

__global__ void matrix_product_transpose_shared(float *m1, float *m2, float *output, int out_width, int out_height, int m1_width, int tile) { //w_in1 rappresenta la dimensione comune delle due matrici di ingresso
    extern __shared__ float s_m[]; //Shared memory equals to piastrella * piastrella * 2
    float *sA = &s_m[0];
    float *sB = &s_m[tile * tile];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = threadIdx.y * tile + threadIdx.x;
    float Cvalue = 0.0;

    for (int ph = 0; ph < (((m1_width - 1) / tile) + 1); ph++) {
        if ((Row < out_height) && (threadIdx.x + (ph * tile)) < m1_width) {
            sA[index] = m1[(Row * m1_width) + threadIdx.x + (ph * tile)];
        } else {
            sA[index] = 0.0;
        }
        if (Col < out_width && (threadIdx.y + ph * tile) < m1_width) {
            sB[index] = m2[(Col * m1_width) + threadIdx.y + (ph * tile)];
        } else {
            sB[index] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < tile; ++j) {
            Cvalue += sA[threadIdx.y * tile + j] * sB[j * tile + threadIdx.x];
        }

        /***
         * Stesso ragionamento fatto per la matrix_product.
        */
        __syncthreads();
    }
    if (Row < out_height && Col < out_width) {
        output[Row * out_width + Col] = Cvalue;
    }
}