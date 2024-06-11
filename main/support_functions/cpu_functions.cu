#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include "..\\leNet.h"

__host__ void debug_print(float *matrice_dev, int w, int h, int c, int n, const char *filename){

	FILE *fp;
	if(filename != NULL){
		if((fp = fopen(filename, "a")) == NULL){
			fprintf(stderr, "FILE NON TROVATO\n");
			exit(1);
		}
	}
	else{
		fp = stdout;
	}
    float *tmp = (float *)malloc(sizeof(float) * w * h * c * n);
    cudaMemcpy(tmp, matrice_dev, sizeof(float) * w * h * c * n, cudaMemcpyDeviceToHost);

    for(int l = 0; l < n; l++){
        for(int i = 0; i < c; i++){
            for(int j = 0; j < h; j++){
                for(int k = 0; k < w; k++){
                    fprintf(fp, "%e ",tmp[(((k + j * w) + i * (h * w)) + l * (h * w * c))]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "----\n");
        }
        fprintf(fp, "\n##########\n--------------------\n##########\n");
    }

	if(fp != stdout){
		fclose(fp);
	}

    free(tmp);
    return;
}

__host__ void save_parameter(float *param, int w, int h, int c, int n, FILE *fp){
    float *tmp = (float *)malloc(sizeof(float) * w * h * c * n);
    cudaMemcpy(tmp, param, sizeof(float) * w * h * c * n, cudaMemcpyDeviceToHost);

    for(int l = 0; l < n; l++){
        for(int i = 0; i < c; i++){
            for(int j = 0; j < h; j++){
                for(int k = 0; k < w; k++){
                    fprintf(fp, "%e ",tmp[(((k + j * w) + i * (h * w)) + l * (h * w * c))]);
                }
                fprintf(fp, "\n");
            }
        }
    }

    free(tmp);
    return;
}

__host__ void load_parameter(float *output, FILE *fp){
    char str[100];
    int t;
    fscanf(fp, "%s %d", str, &t);
    int w, h, c, n;
    fscanf(fp, "%d %d %d %d\n", &w, &h, &c, &n);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < c; j++){
            for(int k = 0; k < h; k++){
                for(int l = 0; l < w; l++){
                    fscanf(fp, "%e", &(output[(((l + k * w) + j * (h * w)) + i * (h * w * c))]));
                }
            }
        }
    }
}

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
    dim3 block{(unsigned int)w, (unsigned int)h};
    subtraction_scalar_parametric<<<c, block>>>(matrix_dev, mean, w, h);
    matrix3D_scalar_product<<<c, block>>>(matrix_dev, (2.0 / (max - min)), w, h);
    free(matrix_host);
}

__host__ void save_img(char *name, float *image){
    char file_name[100];
    FILE *fp;
    int x, y;

    if (name[0] == '\0') {
        printf("output file name (*.pgm) : ");
        scanf("%s", file_name);
    } else strcpy(file_name, name);

    if ( (fp=fopen(file_name, "wb"))==NULL ) {
        printf("could not open file\n");
        exit(1);
    }

    int i;
    fputs("P5\n", fp);
    fputs("# Created by Image Processing\n", fp);
    fprintf(fp, "%d %d\n", 32, 32);
    fprintf(fp, "%d\n", 255);
    for (y=0; y<32; y++){
        for (x=0; x<32; x++){
            i = image[y * 32 + x] * 255;
            fputc(i, fp);
        }
    }
    fclose(fp);
    printf("Image was saved successfully\n");
}

__host__ void load_example_to_device(mnist_data data, float *img_dev, float *target){
    float *tmp = (float *)malloc(sizeof(float) * 32 * 32);

    for(int i = 0; i < 32; i++){
        for(int j = 0; j < 32; j++){
            if(i < 2 || i > 29 || j < 2 || j > 29){
                tmp[i * 32 + j] = 0;
            }
            else{
                tmp[i * 32 + j] = (float)(data.data[i - 2][j - 2]);
            }
        }
    }

    for(int i = 0; i < 10; i++) target[i] = 0;
    target[data.label] = 1;
    cudaMemcpy(img_dev, tmp, sizeof(float) * 32 * 32, cudaMemcpyHostToDevice);

    free(tmp);
}