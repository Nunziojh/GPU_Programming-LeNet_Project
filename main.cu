#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define DIM 100

__global__ void kernel_function(int *in_vec){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    in_vec[idx * 5 + idy + idz * 25] = in_vec[idx * 5 + idy + idz * 25] * 2;
}

int main(int argc, char **argv){

    dim3 grid = {2};
    dim3 block = {5, 5, 2};

    int *memDev;
    int *input_vect = (int *) malloc(sizeof(int) * DIM);

    for(int i = 0; i < DIM; i++) input_vect[i] = i;

    
    cudaMalloc((void **)&memDev, DIM * sizeof(int));

    cudaMemcpy(memDev, input_vect, sizeof(int) * DIM, cudaMemcpyHostToDevice);

    kernel_function<<<grid, block>>>(memDev);

    cudaMemcpy(input_vect, memDev, sizeof(int) * DIM, cudaMemcpyDeviceToHost);

    for(int i = 0; i < DIM; i++) printf("%d\n", input_vect[i]);

    free(input_vect);
    cudaFree(memDev);

    return 0;

}