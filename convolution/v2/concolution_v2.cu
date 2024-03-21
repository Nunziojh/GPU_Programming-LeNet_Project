/*
	Convolution function with padding and no stride using CUDA.
	In this versione the kernle is defined and saved in a constant global variable then moved into the constant memory
	of the device. A pad is added directly by the kernel function. The input matrix is copied into the global memory of the device.
	The kernel function read the values directly from the global memory, save the temporary summation into a register and then save the result into the global memory.
*/


#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// -------------------------------------------
// Define the filter adjusting the filter side dimension accordingly

const int filter_side_dim = 3;
const float filter_h[filter_side_dim*filter_side_dim] = {1, 2, 1, 2, 0.5, -0.1, 1, 2, 0};
__constant__ float filter_d[filter_side_dim*filter_side_dim];


// -------------------------------------------
// Define the width and height of the input matrix

const int height = 10000;
const int width = 5000;

// -------------------------------------------
// Define the number of thread to use

const int thread_num = 32;


// -------------------------------------------
// Kernel function to calculate the convolution

__global__ void filterFunct(float *input, float *output, int height, int width, int padding, int filter_dim){

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if(row >= 0 && row < height && col >= 0 && col < width){

		float result = 0;
		int r, c;

		for(int i = 0; i < (filter_dim * filter_dim); i++){
			r = row + (i / filter_dim) - 1;
			c = col + (i % filter_dim) - 1;
			if(r < 0 || r >= height || c < 0 || c >= width) result += 0;
			else result += filter_d[i] * input[r * width + c];
		}

		output[(row * width + col)] = result;
	}
}
	
int main(){

	// Padding needed in order to keep the dimension untouched
	int padding = filter_side_dim / 2;

	// Copy of the filter from the contant memoty of the host to the constant memory of the device
	cudaMemcpyToSymbol(filter_d, filter_h, sizeof(float) * filter_side_dim * filter_side_dim, 0, cudaMemcpyHostToDevice);

	// Allocating the space for the input and output matrix, treated as vector, on the host
	int size = sizeof(float) * width * height;
	float *input_h = (float *) malloc(size);
	float *output_h = (float *) malloc(size);

	// Initializing the input matrix
	srand(time(NULL));
	for(int i = 0; i < height * width; i++) input_h[i] = (float)rand() / RAND_MAX * 2 - 1;

	// Printing the input
	/*for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++) printf("%.3f ", input_h[i*width+j]);
		printf("\n");
	}
	printf("\n");*/

	// Allocating the space for the input and output matrix on the device
	float *input_d, *output_d;
	cudaMalloc((void **)&input_d, size);
	cudaMalloc((void **)&output_d, size);

	// Define the number of blocks to use accordingly to the number of threads per block defined
	unsigned int block_num_x = width / thread_num + 1;			// The x block's coordinates refer to the colums of the matrix
	unsigned int block_num_y = height / thread_num + 1;			// Tht y block's coordinetes refer to the rows of the matrix

	// Initializing the events to meaure the time required by the kernel
	cudaEvent_t start_c, end_c;
	cudaEventCreate(&start_c);
	cudaEventCreate(&end_c);
	cudaEventRecord(start_c, 0);

	// Copy the input matrix from host to devide
	cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);

	// Calling the kernel function
	filterFunct<<<{block_num_x, block_num_y} , {thread_num, thread_num}>>>(input_d, output_d, height, width, padding, filter_side_dim);

	// Copy the result from device to host
	cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);

	// Recording the end event
	cudaEventRecord(end_c, 0);
	cudaEventSynchronize(end_c);

	// Calculating and printing the elapsed time
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start_c, end_c);
	printf("Input (padded): %d x %d\nKernel: %d x %d\nGrid: %d x %d block(s)\nBlock: %d x %d thread(s)\nTime (in milliseconds): %f\n", width, height, filter_side_dim, filter_side_dim, block_num_x, block_num_y, thread_num, thread_num, elapsedTime);

	// Printing the result
	/*for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++) printf("%.3f ", output_h[i*width+j]);
		printf("\n");
	}*/

	// Freeing the host memory
	free(input_h);
	free(output_h);

	// Freeing jthe device memory
	cudaFree(input_d);
	cudaFree(output_d);

	return 0;
}