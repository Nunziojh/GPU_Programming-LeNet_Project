#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

__global__ void convolution(float *in, float *out, float *kernel, int new_h, int new_w, int padding, int stride, int kernel_h, int kernel_w){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < new_w && idy < new_h){

        int i, j;
        int r = kernel_h / 2;
        int c = kernel_w / 2;

        float tmp = 0;
        float val;

        int new_idx = idx * stride + c - padding;
        int new_idy = idy * stride + r - padding;

        for(i = -r; i <= r; i++){
            for(j = -c; j <= c; j++){
                val = ((new_idy + i) < 0 || (new_idy + i) >= new_h || (new_idx + j) < 0 || (new_idx + j) >= new_w) ? 0 : in[(new_idy + i) * new_w + new_idx + j];
                tmp += kernel[(r-i) * kernel_w + (c-j)] * val;
            }
        }
        out[idy * new_w + idx] = tmp;
    }
}

__global__ void tanh(float *in, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < w && idy < h){

        in[idy * w + idx] = 1.0 / (1.0 + exp(in[idy * w + idx]));
    }
}

__host__ void debug_print(float *matrice_dev, int w, int h, int c){
    float *tmp = (float *)malloc(sizeof(float) * w * h * c);
    cudaMemcpy(tmp, matrice_dev, sizeof(float) * w * h * c, cudaMemcpyDeviceToHost);

    for(int i = 0; i < c; i++){
        for(int j = 0; j < h; j++){
            for(int k = 0; k < w; k++){
                printf("%f ", tmp[(i * h + j) * w + k]);
            }
            printf("\n");
        }
        printf("----\n");
    }

    free(tmp);
    return;
}


int main(){
    srand(time(NULL));

    int padding = 0;
    int stride_c = 1;
    int in_h = 32;
    int in_w = 32;
    int kernel_h = 5;
    int kernel_w = 5;
    int kernel_number = 6;
    int out_h = (in_h + 2 * padding - kernel_h) / stride_c + 1;
    int out_w = (in_w + 2 * padding - kernel_w) / stride_c + 1;


    dim3 block = {(unsigned int)out_w, (unsigned int)out_h};
    dim3 grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};

    float *img = (float *)malloc(sizeof(float) * in_h * in_w);
    float *kernel = (float *)malloc(sizeof(float) * kernel_h * kernel_w * kernel_number);
    for(int i = 0; i < in_h*in_w; i++) img[i] = (float)(rand() % 800) / (float)800;
    for(int i = 0; i < kernel_h*kernel_w*kernel_number; i++) kernel[i] = (float)rand() / (float)RAND_MAX;

    float *img_dev, *first_conv, *kernel_dev;
    cudaMalloc((void **)&img_dev, sizeof(float) * in_h * in_w);
    cudaMalloc((void **)&first_conv, sizeof(float) * out_h * out_w * kernel_number);
    cudaMalloc((void **)&kernel_dev, sizeof(float) * kernel_h * kernel_w * kernel_number);

    cudaMemcpy(img_dev, img, sizeof(float) * in_h * in_w , cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_dev, kernel, sizeof(float) * kernel_h * kernel_w * kernel_number, cudaMemcpyHostToDevice);

    printf("INIZIO---------\n");
    printf("img:\n");
    debug_print(img_dev, in_w, in_h, 1);
    printf("kernel:\n");
    debug_print(kernel_dev, kernel_w, kernel_h, kernel_number);
    printf("conv:\n");
    debug_print(first_conv, out_w, out_h, kernel_number);

    for(int i = 0; i < 6; i++){
        printf("iterazione: %d\n", i);
        convolution<<<grid, block>>>(img_dev, first_conv + (i * out_h * out_w), kernel_dev + (i * kernel_h * kernel_w), out_h, out_w, padding, stride_c, kernel_h, kernel_w);
        debug_print(first_conv, out_w, out_h, kernel_number);

        printf("\n\n");
        tanh<<<grid, block>>>(first_conv + (i * out_h * out_w), out_w, out_h);
        debug_print(first_conv, out_w, out_h, kernel_number);
    }

    printf("FINE-----\n");
    printf("img:\n");
    debug_print(img_dev, in_w, in_h, 1);
    printf("kernel:\n");
    debug_print(kernel_dev, kernel_w, kernel_h, kernel_number);
    printf("conv:\n");
    debug_print(first_conv, out_w, out_h, kernel_number);

    free(img);
    free(kernel);
    cudaFree(img_dev);
    cudaFree(kernel_dev);
    cudaFree(first_conv);
}