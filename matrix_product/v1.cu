#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>
#include <time.h>

__global__ void filterFunct(float *m1, float *m2, float *res, int h, int k, int w){

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
/*
	if(row >= 0 && row < h && col >= 0 && col < w){
		float result = 0;		
		for(int i = 0; i < k; i++){
			result += m1[row * k + i] * m2[k * i + col];
		}

		//__syncthreads();
		res[row * w + col] = result;
	}*/
	printf("(%d, %d)\n", row, col);
}
	
int main(void){

	srand(time(NULL));

	int m1_h = 5;
	int m1_w = 3;
	int m2_h = m1_w;
	int m2_w = 2;

	float *h_m1 = (float *)malloc(sizeof(float) * m1_w * m1_h);
	float *h_m2 = (float *)malloc(sizeof(float) * m2_w * m2_h);
	float *h_res = (float *)malloc(sizeof(float) * m2_w * m1_h);

	for(int i = 0; i < m1_h * m1_h; i++) h_m1[i] = (float)(rand() % 1000) / 100;
	for(int i = 0; i < m2_h * m2_h; i++) h_m2[i] = (float)(rand() % 1000) / 100;

	unsigned int thread_num = 16;
	unsigned int block_num_x = m2_w / thread_num +1;
	unsigned int block_num_y = m1_h / thread_num +1;

	float *dev_m1, *dev_m2, *dev_res;
	cudaMalloc((void **)&dev_m1, sizeof(float) * m1_w * m1_h);
	cudaMalloc((void **)&dev_m2, sizeof(float) * m2_w * m2_h);
	cudaMalloc((void **)&dev_res, sizeof(float) * m2_w * m1_h);

	cudaMemcpy(dev_m1, h_m1, sizeof(float) * m1_w * m1_h, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_m2, h_m2, sizeof(float) * m2_w * m2_h, cudaMemcpyHostToDevice);

	//printf("Ok");
	filterFunct<<<{block_num_x, block_num_y}, {thread_num, thread_num}>>>(dev_m1, dev_m2, dev_res, m1_h, m1_w, m2_w);

	cudaMemcpy(h_res, dev_res, sizeof(float) * m1_h * m2_w, cudaMemcpyDeviceToHost);

	for(int i = 0; i < m1_h; i++){
		for(int j = 0; j < m1_w; j++){
			printf("%.3f ", h_m1[j + i * m1_w]);
		}
		printf("\n");
	}
	printf("\n");
	for(int i = 0; i < m2_h; i++){
		for(int j = 0; j < m2_w; j++){
			printf("%.3f ", h_m2[j + i * m2_w]);
		}
		printf("\n");
	}
	printf("\n");
	for(int i = 0; i < m1_h; i++){
		for(int j = 0; j < m2_w; j++){
			printf("%.3f ", h_res[j + i * m2_w]);
		}
		printf("\n");
	}
	printf("\n");

	cudaFree(dev_m1);
	cudaFree(dev_m2);
	cudaFree(dev_res);

	free(h_m1);
	free(h_m2);
	free(h_res);

	return 0;
}