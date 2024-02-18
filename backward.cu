#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define H 32
#define W 32
#define KERNEL_DIM 5
#define KERNEL_NUM 22
#define POOLING_WINDOW_SIZE 2

__global__ void convolution(float *in, float *out, float *kernel, int new_h, int new_w, int padding, int stride){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;
        int r = KERNEL_DIM / 2;
        int c = KERNEL_DIM / 2;

        float tmp = 0;
        float val;

        int new_idx = idx * stride - c + padding;
        int new_idy = idy * stride - r + padding;

        for(i = -r; i <= r; i++){
            for(j = -c; j <= c; j++){
                val = ((new_idy + i) < 0 || (new_idy + i) >= H || (new_idx + j) < 0 || (new_idx + j) >= W) ? 0 : in[(new_idy + i) * W + new_idx + j];
                tmp += kernel[(i+1) * KERNEL_DIM + (j+1)] * val;
            }
        }
        out[idy * new_w + idx] = tmp;
    }
}

__global__ void convolution3D(float *in, float *out, float *kernel, int new_h, int new_w, int padding, int stride){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;
        int r = KERNEL_DIM / 2;
        int c = KERNEL_DIM / 2;

        float tmp = 0;
        float val;

        int new_idx = idx * stride - c + padding;
        int new_idy = idy * stride - r + padding;

        for(i = -r; i <= r; i++){
            for(j = -c; j <= c; j++){
                val = ((new_idy + i) < 0 || (new_idy + i) >= H || (new_idx + j) < 0 || (new_idx + j) >= W) ? 0 : in[(new_idy + i) * W + new_idx + j];
                tmp += kernel[(i+1) * KERNEL_DIM + (j+1)] * val;
            }
        }
        out[idy * new_w + idx] += tmp;
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

        for(i = 0; i < POOLING_WINDOW_SIZE; i++){
            for(j = 0; j < POOLING_WINDOW_SIZE; j++){
                tmp += ((new_idy + i) >= h || (new_idx + j) >= w) ? 0 : in[(new_idy + i) * w + new_idx + j];
            }
        }

        __syncthreads();

        out[idy * new_w + idx] = tmp / (float)(POOLING_WINDOW_SIZE * POOLING_WINDOW_SIZE);
    }
}

__global__ void matrix_product(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_out && idy < h_out){

        float tmp = 0;
        in1 = in1 + idy * w_in1;       //Inutile
        for(int i = 0, j = 0; i < w_in1; i++, j += w_out){
            tmp += in1[i] * in2[j + idx];
        }

        out[idy * w_out + idx] = tmp;
    }
}

__global__ void matrix_transpose_product(float *in1, float *in2, float *out, int w_out, int h_out, int h_in1){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_out && idy < h_out){

        float tmp = 0;
        for(int i = 0, j = 0; i < h_in1 * h_out; i += h_out, j += w_out){
            tmp += in1[idy + i] * in2[idx + j];
        }
        out[idy * w_out + idx] = tmp;
    }
}

__global__ void matrix_product_transpose(float *in1, float *in2, float *out, int w_out, int h_out, int w_in1){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w_out && idy < h_out){

        float tmp = 0;
        for(int i = 0; i < w_in1; i++){
            tmp += in1[idy * w_in1 + i] * in2[idx * w_in1 + i];
        }
        out[idy * w_out + idx] = tmp;
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

__global__ void tanh(float *in, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){

        in[idy * w + idx] = 1 / (1 + exp(in[idy * w + idx]));
    }
}

__global__ void exponential(float *in, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < len){
        in[idx] = exp(in[idx]);
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

__global__ void transpose(float *out, float *in, int w_out, int h_out){         // BlockDim = alle dimensioni della matrice di uscita
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < w_out && idy < h_out){
        out[idy * w_out + idx] = in[idx * h_out + idy];
    }
}

int main(){
    srand(time(NULL));

    /***
     * Definizione dei parametri della rete
    */

    int padding = 0;
    int stride_c = 1;
    int stride_p = 2;
    int kernel_num_first_layer = 6;
    int kernel_num_second_layer = 16;
    int kernel_num_third_layer = 120;
    int fc_first_dim = 120;
    int fc_second_dim = 84;
    int fc_third_dim = 10;

    /***
     * Definizione, allocazione e inizializzazione randomica dei kernel e delle matrici dei pesi
    */

    float *kernels_first_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer);
    float *kernels_second_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer);
    float *kernels_third_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer);
    float *fc_first_layer = (float *) malloc(sizeof(float) * fc_second_dim * fc_first_dim);         // W1 (84 x 120)
    float *fc_second_layer = (float *) malloc(sizeof(float) * fc_third_dim * fc_second_dim);        // W2 (10 x 84)
    float *prediction = (float *) malloc(sizeof(float) * fc_third_dim);
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer; i++) kernels_first_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer; i++) kernels_second_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer; i++) kernels_third_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < fc_first_dim * fc_second_dim; i++) fc_first_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < fc_second_dim * fc_third_dim; i++) fc_second_layer[i] = (float)rand() / (float)RAND_MAX;

    /***
     * Definizione e allocazione dei kernel e delle matrici su device
    */

    float *kernels_first_layer_dev, *kernels_second_layer_dev, *kernels_third_layer_dev, *fc_first_layer_dev, *fc_second_layer_dev;
    cudaMalloc((void **)&kernels_first_layer_dev, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer);
    cudaMalloc((void **)&kernels_second_layer_dev, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer);     // kernel[5][5][6][16]
    cudaMalloc((void **)&kernels_third_layer_dev, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer);
    cudaMalloc((void **)&fc_first_layer_dev, sizeof(float) * fc_first_dim * fc_second_dim);
    cudaMalloc((void **)&fc_second_layer_dev, sizeof(float) * fc_second_dim * fc_third_dim);

    cudaMemcpy(kernels_first_layer_dev, kernels_first_layer, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(kernels_second_layer_dev, kernels_second_layer, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(kernels_first_layer_dev, kernels_first_layer, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(fc_first_layer_dev, fc_first_layer, sizeof(float) * fc_first_dim * fc_second_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(fc_second_layer_dev, fc_second_layer, sizeof(float) * fc_second_dim * fc_third_dim, cudaMemcpyHostToDevice);

    int in_h = 32;
    int in_w = 32;
    int out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
    int out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;

    float target[10] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    float *img = (float *) malloc(sizeof(float) * in_h * in_w);
    for(int i = 0; i < in_w * in_w; i++) img[i] = i;

    float *img_dev, *first_conv;
    cudaMalloc((void **)&img_dev, sizeof(float) * in_w * in_w);
    cudaMalloc((void **)&first_conv, sizeof(float) * out_w * out_h * kernel_num_first_layer);

    cudaMemcpy(img_dev, img, sizeof(float) * in_w * in_h, cudaMemcpyHostToDevice);

    dim3 block = {(unsigned int)out_w, (unsigned int)out_h};
    dim3 grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
    for(int i = 0; i < kernel_num_first_layer; i++){
        convolution<<<grid, block>>>(img_dev, first_conv + (i * out_h * out_w), kernels_first_layer_dev + (i * KERNEL_DIM * KERNEL_DIM), out_h, out_w, padding, stride_c);
        tanh<<<grid, block>>>(first_conv + (i * out_h * out_w), out_w, out_h);
    }

    in_h = out_h;
    in_w = out_w; 
    out_h = (in_h - POOLING_WINDOW_SIZE) / stride_p + 1;
    out_w = (in_w - POOLING_WINDOW_SIZE) / stride_p + 1;
    float *first_pool;
    cudaMalloc((void **)&first_pool, sizeof(float) * out_w * out_h * kernel_num_first_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
    for(int i = 0; i < kernel_num_first_layer; i++){
        avg_pooling<<<grid, block>>>(first_conv + (i * in_h * in_w), first_pool + (i * out_h * out_w), in_h, in_w, out_h, out_w, stride_p);
        tanh<<<grid, block>>>(first_pool + (i * out_h * out_w), out_w, out_h);
    }

    in_h = out_h;
    in_w = out_w;
    out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
    out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;
    float *second_conv;
    cudaMalloc((void **)&second_conv, sizeof(float) * out_w * out_h * kernel_num_second_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
    for(int i = 0; i < kernel_num_second_layer; i++){
        for(int j = 0; j < kernel_num_first_layer; j++){
            convolution3D<<<grid, block>>>(first_pool + (j * in_h * in_w), second_conv + (i * out_h * out_w), kernels_second_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer)), out_h, out_w, padding, stride_c);
        }
        tanh<<<grid, block>>>(second_conv + (i * out_h * out_w), out_w, out_h);
    }

    /*for(int i = 0; i < kernel_num_second_layer; i++){
        convolution<<<grid, block>>>(first_pool, second_conv + (i * out_h * out_w), kernels_second_layer_dev + (i * KERNEL_DIM * KERNEL_DIM), out_h, out_w, padding, stride_c);
        tanh<<<grid, block>>>(second_conv + (i * out_h * out_w), out_w, out_h);
    }*/

    in_h = out_h;       //10
    in_w = out_w;
    out_h = (in_h - POOLING_WINDOW_SIZE) / stride_p + 1;            //5
    out_w = (in_w - POOLING_WINDOW_SIZE) / stride_p + 1;
    float *second_pool;
    cudaMalloc((void **)&second_pool, sizeof(float) * out_w * out_h * kernel_num_second_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
    for(int i = 0; i < kernel_num_second_layer; i++){
        avg_pooling<<<grid, block>>>(second_conv + (i * in_h * in_w), second_pool + (i * out_h * out_w), in_h, in_w, out_h, out_w, stride_p);
        tanh<<<grid, block>>>(second_pool + (i * out_h * out_w), out_w, out_h);
    }

    in_h = out_h;       //5
    in_w = out_w;
    out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;       //1
    out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;
    float *third_conv;
    cudaMalloc((void **)&third_conv, sizeof(float) * out_w * out_h * kernel_num_third_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
    for(int i = 0; i < kernel_num_third_layer; i++){
        for(int j = 0; j < kernel_num_second_layer; j++){
            convolution3D<<<grid, block>>>(second_pool + (j * in_h * in_w), third_conv + (i * out_h * out_w), kernels_third_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer)), out_h, out_w, padding, stride_c);
        }
        tanh<<<grid, block>>>(third_conv + (i * out_h * out_w), out_w, out_h);
    }

    float *second_fc;
    cudaMalloc((void **)&second_fc, sizeof(float) * fc_second_dim);
    block = {1, (unsigned int)fc_second_dim};
    grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
    matrix_product<<<grid, block>>>(fc_first_layer_dev, third_conv, second_fc, 1, fc_second_dim, fc_first_dim);
    tanh<<<grid, block>>>(second_fc, fc_second_dim, 1);

    float *third_fc;
    cudaMalloc((void **)&third_fc, sizeof(float) * fc_third_dim);
    block = {1, (unsigned int)fc_third_dim};
    grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
    matrix_product<<<grid, block>>>(fc_second_layer_dev, second_fc, third_fc, 1, fc_third_dim, fc_second_dim);

    block = {1, (unsigned int)fc_third_dim};
    grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};

    exponential<<<grid, block>>>(second_fc, fc_third_dim);

    cudaMemcpy(prediction, fc_second_layer_dev, sizeof(float) * fc_third_dim, cudaMemcpyDeviceToHost);

    float summation = 0.0;
    for(int i = 0; i < fc_third_dim; i++) summation += prediction[i];
    for(int i = 0; i < fc_third_dim; i++) prediction[i] = prediction[i] / summation;
    for(int i = 0; i < fc_third_dim; i++) printf("%2.2f\n", prediction[i]);


    float loss = 0.0;
    for(int i = 0; i < fc_third_dim; i++) loss += target[i] * log(prediction[i]);
    loss = -loss;


    /*
        BackWard Propagatoin
    */

    // DA CONTROLLARE LA DERIVATA RISPETTO DELLA SOFTMAX
    float *prediction_dev, *target_dev;
    cudaMalloc((void **)&prediction_dev, sizeof(float) * fc_third_dim);
    cudaMalloc((void **)&target_dev, sizeof(float) * fc_third_dim);
    cudaMemcpy(prediction_dev, prediction, sizeof(float) * fc_third_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(target_dev, target, sizeof(float) * fc_third_dim, cudaMemcpyHostToDevice);

    float *dZ2;
    cudaMalloc((void **)&dZ2, sizeof(float) * fc_third_dim);            // Da moltiplicare per il numero di immagini usate nel batch
    block = {(unsigned int)fc_third_dim};
    grid = {(unsigned int)(block.x / 1024 +1)};
    subtraction<<<grid, block>>>(dZ2, prediction_dev, target_dev, fc_third_dim);
    // FINE CONTROLLO


    float *dW2;
    cudaMalloc((void **)&dW2, sizeof(float) * fc_third_dim * fc_second_dim);
    block = {(unsigned int)fc_second_dim, (unsigned int)fc_third_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    /*
        prodotto di matrici tra la matrice dZ2 (fc_third_dim x immagini_in_ingresso)
        e le attivazioni del layer precedente, second_fc (fc_second_dim x immagini_in_ingresso) -> second_rc^T (immagini_in_ingresso x fc_second_dim)
        per ottenere la matrice di derivata dei pesi dW2 (fc_third_dim x fc_second_dim)
    */
    matrix_product_transpose<<<grid, block>>>(dZ2, second_fc, dW2, fc_second_dim, fc_third_dim, 1);           // Quando lavoreremo con batch di dimensioni maggiori di 1 dovremo dividere ogni elemento della matrice per m e come ultimo parametro della chiamata a funzione mettere m invece di 1
    
    float *dZ1;
    cudaMalloc((void **)&dZ1, sizeof(float) * fc_second_dim * 1);            //Da moltiplicare per il numero di immagini usate nel batch
    block = {1, (unsigned int)fc_second_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_transpose_product<<<grid, block>>>(fc_second_layer_dev, dZ2, dZ1, 1, fc_second_dim, fc_third_dim);
    float *gdZ1;
    cudaMalloc((void **)&gdZ1, sizeof(float) * fc_second_dim * 1);
    block = {1, (unsigned int)fc_second_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_dot_product<<<grid, block>>>(second_fc, second_fc, gdZ1, 1, fc_second_dim);
    scalar_subtraction<<<grid, block>>>(gdZ1, gdZ1, 1, fc_second_dim);
    matrix_dot_product<<<grid, block>>>(dZ1, gdZ1, dZ1, 1, fc_second_dim);

    float *dW1;
    cudaMalloc((void **)&dW1, sizeof(float) * fc_second_dim * fc_first_dim);
    block = {(unsigned int)fc_first_dim, (unsigned int)fc_second_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_product_transpose<<<grid, block>>>(dZ1, third_conv, dW1, fc_first_dim, fc_second_dim, 1);

    float *dZ0;
    cudaMalloc((void **)&dZ0, sizeof(float) * fc_first_dim * 1);            //Da moltiplicare per il numero di immagini usate nel batch
    block = {1, (unsigned int)fc_first_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_transpose_product<<<grid, block>>>(fc_first_layer_dev, dZ1, dZ0, 1, fc_first_dim, fc_second_dim);
    float *gdZ0;
    cudaMalloc((void **)&gdZ0, sizeof(float) * fc_first_dim * 1);
    block = {1, (unsigned int)fc_first_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_dot_product<<<grid, block>>>(third_conv, third_conv, gdZ0, 1, fc_first_dim);
    scalar_subtraction<<<grid, block>>>(gdZ0, gdZ0, 1, fc_first_dim);
    matrix_dot_product<<<grid, block>>>(dZ0, gdZ0, dZ0, 1, fc_first_dim);

    float *dF2;
    out_h = 5;
    out_w = 5;
    cudaMalloc((void **)&dF2, sizeof(float) * out_h * out_w * kernel_num_second_layer * kernel_num_third_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_third_layer; i++){
        for(int j = 0; j < kernel_num_second_layer; j++){
            convolution<<<grid, block>>>(second_pool + (j * in_h * in_w), dF2 + (j * in_h * in_w + (i * in_h * in_w * kernel_num_third_layer)), dZ0 + (i * 1 * 1), KERNEL_DIM, KERNEL_DIM, padding, stride_c);
        }
    }

    float *dA3;
    out_h = 5;
    out_w = 5;
    cudaMalloc((void **)&dA3, sizeof(float) * out_h * out_w * kernel_num_second_layer);
    block = {};
    grid = {};
    










    free(kernels_first_layer);
    free(kernels_second_layer);
    free(kernels_third_layer);
    free(fc_first_layer);
    free(fc_second_layer);
    free(prediction);
    free(img);

    cudaFree(kernels_first_layer_dev);
    cudaFree(kernels_second_layer_dev);
    cudaFree(kernels_third_layer_dev);
    cudaFree(fc_first_layer_dev);
    cudaFree(fc_second_layer_dev);

    cudaFree(img_dev);
    cudaFree(first_conv);
    cudaFree(first_pool);
    cudaFree(second_conv);
    cudaFree(second_pool);
    cudaFree(third_conv);
    cudaFree(second_fc);
    cudaFree(third_fc);

    return 0;
}