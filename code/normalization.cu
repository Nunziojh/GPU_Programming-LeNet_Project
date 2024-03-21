#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "gpu_functions.h"
#include <float.h>

__host__ void mean_max_min(float *matrix, float *mean, float *max, float *min, int w, int h, int c){
    float tmp;
    float sum = 0;
    for(int i = 0; i < c; i++){
        for(int j = 0; j < h; j++){
            for(int k = 0; k < w; k++){
                tmp = matrix[k + j * w + i * h * w];
                if(tmp < *min) *min = tmp;
                if(tmp > *max) *max = tmp;
                sum += tmp;
            }
        }
    }
    *mean = sum / (float)(w * h * c);
}

__host__ void mean_normalization(float *matrix_dev, int w, int h, int c){
    float mean = 0;
    float max = -FLT_MAX, min = FLT_MAX;
    float *matrix_host = (float *)malloc(sizeof(float) * w * h * c);
    cudaMemcpy(matrix_host, matrix_dev, sizeof(float) * w * h * c, cudaMemcpyDeviceToHost);
    mean_max_min(matrix_host, &mean, &max, &min, w, h, c);
    printf("\t\t\t\t\t%e, %e, %e\n", mean, max, min);
    dim3 block{(unsigned int)w, (unsigned int)h};
    subtraction_scalar_parametric<<<c, block>>>(matrix_dev, mean, w, h);
    matrix3D_scalar_product<<<c, block>>>(matrix_dev, (2.0 / (max - min)), w, h, 0);
    free(matrix_host);
}


int main(){

    

    /*float *matrice = (float *)malloc(sizeof(float) * 28 * 28 * 6);
    for(int i = 0; i < 4704; i++) matrice[i] = i;

    for(int i = 0; i < 6; i++){
        for(int j = 0; j < 28; j++){
            for(int k = 0; k < 28; k++){
                printf("%f ", matrice[i * 28 * 28 + j * 28 + k]);
            }
            printf("\n");
        }
        printf("----------------\n");
    }*/

    /*float mean = 0;
    float max = -FLT_MAX, min = FLT_MAX;
    mean_max_min(matrice, &mean, &max, &min, 10, 10, 6);
    printf("\t\t\t\t\t%e, %e, %e\n", mean, max, min);*/


    /*float *matrice_dev;
    cudaMalloc((void **)&matrice_dev, sizeof(float) * 4704);
    cudaMemcpy(matrice_dev, matrice, sizeof(float) * 4704, cudaMemcpyHostToDevice);

    mean_normalization(matrice_dev, 28, 28, 6);*/

    /*
    subtraction_scalar_parametric<<<6, {10, 10}>>>(matrice_dev, 299.5, 10, 10);
    matrix3D_scalar_product<<<6, {10, 10}>>>(matrice_dev, 2.0/599.0, 10, 10, 0);*/
    /*cudaMemcpy(matrice, matrice_dev, sizeof(float) * 4704, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 6; i++){
        for(int j = 0; j < 28; j++){
            for(int k = 0; k < 28; k++){
                printf("%f ", matrice[i * 28 * 28 + j * 28 + k]);
            }
            printf("\n");
        }
        printf("----------------\n");
    }

    free(matrice);
    cudaFree(matrice_dev);*/

    return 0;
}