#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "img.h"

#define FILTER_DIMENSION 9			// square of the side
#define MAX 4080
#define MIN 0

__constant__ int filter[FILTER_DIMENSION];

__global__ void filterFunct(unsigned char *img, unsigned char *res, int height, int width, int padding){

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if(row > 0 && row < (height - 1) && col > 0 && col < (width - 1)){

		int result = 0;
		int filter_offset = (int)sqrt(atoi(FILTER_DIMENSION));
		
		for(int i = 0; i < FILTER_DIMENSION; i++){
			result += filter[i] * img[(row + (i / filter_offset) - 1) * width + (col + (i % filter_offset) - 1)];
		}

		//__syncthreads();

		res[((row - 1) * (width - (2 * padding))) + (col - 1)] = ((float)(result) - MIN) / (MAX - MIN) * 255;		//(x - min) / (max - min) * 255
	}
}

void addPadd(GrayImage *img, int padding){

	int new_height = img->height + 2 * padding, new_width = img->width + 2 * padding;

	unsigned char *new_data = (unsigned char *) malloc(sizeof(unsigned char) * new_height * new_width);

	int i = 0, j = 0, offset;
	for(i = 0; i < (new_width * padding); i++) new_data[i] = 0;
	for(i = 0; i < img->height; i++){
		offset = (padding + i) * new_width;
		for(j = 0; j < padding; j++) new_data[offset + j] = 0;
		for(j = 0; j < img->width; j++) new_data[offset + padding + j] = img->data[i * img->width + j];
		for(j = 0; j < padding; j++) new_data[offset + padding + img->width + j] = 0;
	}
	for(i = 0; i < (new_width * padding); i++) new_data[(padding + img->height) * new_width + i] = 0;

	img->height = new_height;
	img->width = new_width;
	unsigned char *tmp = img->data;
	img->data = new_data;
	free(tmp);
}
	
int main(int argc, char **argv){

	if(argc != 4){
		printf("%s input_file(.pgm) threas_per_block file_out(.pgm)\n", argv[0]);
		return 1;
	}
	if(atoi(argv[2]) > 1024) {
		printf("Threads number must be a square of 2 and less or equal to 1024\n");
		return 1;
	}

	int filtro[FILTER_DIMENSION] = {3, 1, 3, 1, 0, 1, 3, 1, 3};

	int padding = (int)(sqrt(FILTER_DIMENSION) / 2);
	cudaMemcpyToSymbol(filter, filtro, sizeof(int) * FILTER_DIMENSION);

	GrayImage *h_img = readPGM(argv[1]);
	GrayImage *h_res = createPGM(h_img->width, h_img->height);
	addPadd(h_img, padding);

	int size_padded = sizeof(unsigned char) * h_img->width * h_img->height;
	int size = sizeof(unsigned char) * h_res->width * h_res->height;

	unsigned int thread_num = (int)sqrt(atoi(argv[2]));
	unsigned int block_num_x = (h_img->width % thread_num == 0) ? h_img->width / thread_num : (h_img->width / thread_num + 1);
	unsigned int block_num_y = (h_img->height % thread_num == 0) ? h_img->height / thread_num : (h_img->height / thread_num + 1);

	unsigned char *dev_img, *dev_res;
	cudaMalloc((void **)&dev_img, size_padded);
	cudaMalloc((void **)&dev_res, size);
	cudaMemcpy(dev_img, h_img->data, size_padded, cudaMemcpyHostToDevice);

	filterFunct<<<{block_num_x, block_num_y} , {thread_num, thread_num}>>>(dev_img, dev_res, h_img->height, h_img->width, padding);

	cudaMemcpy(h_res->data, dev_res, size, cudaMemcpyDeviceToHost);

	writePGM(argv[3], h_res);

	destroyPGM(h_img);
	destroyPGM(h_res);

	cudaFree(dev_img);
	cudaFree(dev_res);

	return 0;
}