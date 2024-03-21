/*
	Max pooling calculated using CUDA.
	The input matrix is copied into the global memory of the device.
	The kernel function read the values directly from the global memory, save the temporary summation into a register and then save the result into the global memory.
*/


#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// -------------------------------------------
// Define the kernel size
const int kernel_size = 2;

// -------------------------------------------
// Define the width and height of the input matrix

const int height = 10;			// Constrain: the dimensions must be a multiple of the kernel size
const int width = 10;				

// -------------------------------------------
// Define the number of thread to use

const int thread_num = 32;


// -------------------------------------------
// Kernel function to calculate the convolution

__global__ void MaxPoolingFunct(int *input, int *output, int height, int width, int kernel_dim, int offset){

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if(row >= 0 && row < height && col >= 0 && col < width){

		int index;

		int *in = &input[row * 2 * width * kernel_size + col * 2];
		int result = in[0];

		for(int i = 1; i < (kernel_dim * kernel_dim); i++){
			index = i + offset * (i/kernel_dim);
			result = max(result, in[index]);
		}

		output[(row * width + col)] = result;
	}
}
	
int main(){

	// Allocating the space for the input and output matrix, treated as vector, on the host
	int input_size = sizeof(int) * width * height;
	int *input_h = (int *) malloc(input_size);
	int output_size = sizeof(int) * (width / kernel_size) * (height / kernel_size);
	int *output_h = (int *) malloc(output_size);

	// Initializing the input matrix
	srand(time(NULL));
	for(int i = 0; i < height * width; i++) input_h[i] = rand() % 100 - 50;

	// Printing the input
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++) printf("%d ", input_h[i*width+j]);
		printf("\n");
	}
	printf("\n");

	// Allocating the space for the input and output matrix on the device
	int *input_d, *output_d;
	cudaMalloc((void **)&input_d, input_size);
	cudaMalloc((void **)&output_d, output_size);

	// Define the number of blocks to use accordingly to the dimension of the output and the number of threads per block defined
	unsigned int block_num_x = width / kernel_size / thread_num + 1;			// The x block's coordinates refer to the colums of the matrix
	unsigned int block_num_y = height / kernel_size / thread_num + 1;			// Tht y block's coordinetes refer to the rows of the matrix

	// Initializing the events to meaure the time required by the kernel
	cudaEvent_t start_c, end_c;
	cudaEventCreate(&start_c);
	cudaEventCreate(&end_c);
	cudaEventRecord(start_c, 0);

	// Copy the input matrix from host to devide
	cudaMemcpy(input_d, input_h, input_size, cudaMemcpyHostToDevice);

	// Calling the kernel function
	MaxPoolingFunct<<<{block_num_x, block_num_y} , {thread_num, thread_num}>>>(input_d, output_d, height / kernel_size, width / kernel_size, kernel_size, width - kernel_size);

	// Copy the result from device to host
	cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost);

	// Recording the end event
	cudaEventRecord(end_c, 0);
	cudaEventSynchronize(end_c);

	// Calculating and printing the elapsed time
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start_c, end_c);
	printf("Input: %d x %d\nPooling kernel: %d x %d\nGrid: %d x %d block(s)\nBlock: %d x %d thread(s)\nTime (in milliseconds): %f\n\n", width, height, kernel_size, kernel_size, block_num_x, block_num_y, thread_num, thread_num, elapsedTime);

	// Printing the result
	for(int i = 0; i < height / kernel_size; i++){
		for(int j = 0; j < width / kernel_size; j++) printf("%d ", output_h[i*width/kernel_size+j]);
		printf("\n");
	}

	// Freeing the host memory
	free(input_h);
	free(output_h);

	// Freeing jthe device memory
	cudaFree(input_d);
	cudaFree(output_d);

	return 0;
}