#include "monolithic_shared.h"

__global__ void convolution_3D_shared(float *input, float *kernel, float *output, int in_width, int in_height, int in_depth, int ker_width, int ker_height, int ker_depth, int ker_number, int out_width, int out_height, int out_depth) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oz = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float s_m[];
    float *data = &(s_m[0]);
    float *filter = &(s_m[in_depth * in_height * in_width]);

    if(threadIdx.x < in_width && threadIdx.y < in_height && threadIdx.z < in_depth)
    {
        for(int i = threadIdx.z; i < in_depth; i+=blockDim.z)
        {
            data[(i * in_height + oy) * in_width + ox] = input[(i * in_height + oy) * in_width + ox];
        }
    }

    if(threadIdx.x < ker_width && threadIdx.y < ker_height && threadIdx.z < ker_number)
    {
        for(int i = 0; i < KERNEL_Z; i++)
        {
            filter[threadIdx.z * (KERNEL_X * KERNEL_Y * KERNEL_Z) + i * (KERNEL_X * KERNEL_Y) + threadIdx.y * KERNEL_X + threadIdx.x] = kernel[oz * (KERNEL_X * KERNEL_Y * KERNEL_Z) + i * (KERNEL_X * KERNEL_Y) + threadIdx.y * KERNEL_X + threadIdx.x];
        }
        /*for(int j = 0; j < blockDim.z && (j + blockIdx.z * blockDim.z) < ker_number; j++)
        {
            for(int i = threadIdx.z; i < ker_depth; i+=blockDim.z)
            {
                filter[((j * ker_depth + i) * ker_height + threadIdx.y) * ker_width + threadIdx.x] = kernel[(((j + blockIdx.z * blockDim.z) * ker_depth + i) * ker_height + threadIdx.y) * ker_width + threadIdx.x];
            }
        }*/
    }

    __syncthreads();

    if (ox < out_width && oy < out_height && oz < out_depth)
    {
        float sum = 0.0f;
        int kz, ky, kx;
        float in_val, ker_val;
        filter += (threadIdx.z * ker_depth * ker_height * ker_width);

        for(kz = 0; kz < ker_depth; kz++)
        {
            for(ky = 0; ky < ker_height; ky++)
            {
                for(kx = 0; kx < ker_width; kx++)
                {
                    if((ox + kx) >= 0 && (ox + kx) < in_width && (oy + ky) >= 0 && (oy + ky) < in_height){
                        in_val = data[(oy + ky) * in_width + ox + kx];
                        ker_val = filter[(ker_width * ker_height - 1) - (ky * ker_width + kx)];
                        sum += in_val * ker_val;
                    }
                }
            }
            data += (in_height * in_width);
            filter += (ker_height * ker_width);
        }
        output[(oz * out_height * out_width) + oy * out_width + ox] = sum;
    }
}

__global__ void convolution_forNOutChannels_shared(float *in, float *kernel, float *out, int in_w, int in_h, int in_d, int kernel_w, int kernel_h, int kernel_d, int kernel_n, int out_w, int out_h, int out_d, int out_n)
{
    /***
     * Il numero di threads viene definito in base alle dimensioni di uscita.
     * La dimensione x del blocco viene calcolata come il prodotto tra le mensione x e y dell'uscita.
     * La dimensione y del blocco è pari alla dimensione z dell'uscita.
     * La dimensione z del blocco è pari al numero di matrici di uscita.
     * Capire bene quanti sono i threads da assegnare ad ogni parte del blocco.
    */
    int oxy = blockIdx.x * blockDim.x + threadIdx.x;
    int oz = blockIdx.y * blockDim.y + threadIdx.y;
    int on = blockIdx.z * blockDim.z + threadIdx.z;

    int ox = oxy % in_h;
    int oy = oxy / in_w;

    extern __shared__ float s_m[];
    float *data = &(s_m[0]);
    float *filter = &(s_m[in_w * in_h]);

    if(ox < in_w && oy < in_h && oz < in_d)
    {
        data[oy * in_w + ox] = in[(oz * in_h + oy) * in_w + ox];
    }

    if(ox < kernel_w && oy < kernel_h && on < kernel_n)
    {
        filter[(threadIdx.z * kernel_h + oy) * kernel_w + ox] = kernel[(on * kernel_h + oy) * kernel_w + ox];
    }

    __syncthreads();

    if (ox < out_w && oy < out_h && oz < out_d && on < out_n)
    {
        float sum = 0.0f;
        int ky, kx;
        float in_val, ker_val;
        // kernel += (on * kernel_d * kernel_h * kernel_w);
        filter += (threadIdx.z * kernel_d * kernel_h * kernel_w);
        //in += (oz * in_h * in_w);

        for(ky = 0; ky < kernel_h; ky++)
        {
            for(kx = 0; kx < kernel_w; kx++)
            {
                if((ox + kx) >= 0 && (ox + kx) < in_w && (oy + ky) >= 0 && (oy + ky) < in_h){
                    // in_val = in[(oy + ky) * in_w + ox + kx];
                    in_val = data[(oy + ky) * in_w + ox + kx];
                    // ker_val = kernel[(kernel_w * kernel_h - 1) - (ky * kernel_w + kx)];
                    ker_val = filter[(kernel_w * kernel_h - 1) - (ky * kernel_w + kx)];
                    sum += in_val * ker_val;
                }
            }
        }
        out[(((on * out_d) + oz) * out_h + oy) * out_w + ox] = sum;
    }
}