#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void tanh(float *in, int w, int h, int d){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if(idx < w && idy < h && idz < d){
        int index = (idz * h + idy) * w + idx;

        float val = in[index];
        float e = expf(2 * val);
        in[index] = (e - 1) / (e + 1);
    }
}

__global__ void exponential(float *in, int len){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < len){
        in[idx] = expf(in[idx]);
    }
}

__global__ void subtraction(float *out, float *in1, float*in2, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < dim){
        out[idx] = in1[idx] - in2[idx]; 
    }
}

__global__ void scalar_subtraction(float *out, float *in, int w, int h, int d){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if(idx < w && idy < h && idz < d){
        int index = (idz * h + idy) * w + idx;
        out[index] = 1 - in[index];
    }
}

__global__ void subtraction_scalar_parametric(float *io, float scalar, int dim){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < dim){
        io[idx] = io[idx] - scalar;
    }
}

// __global__ void transpose(float *out, float *in, int w_out, int h_out){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;

//     if(idx < w_out && idy < h_out){
//         out[idy * w_out + idx] = in[idx * h_out + idy];
//     }
// }

// __global__ void clean_vector(float *m, int dim){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if(idx < dim){
//         m[idx] = 0;
//     }
// }