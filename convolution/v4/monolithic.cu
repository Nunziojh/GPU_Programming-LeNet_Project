#include "monolithic.h"

__global__ void convolution_3D(float *input, float *kernel, float *output, int in_width, int in_height, int in_depth, int ker_width, int ker_height, int ker_depth, int out_width, int out_height, int out_depth) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x; //Numero di threads definiti in base alle dimensioni di uscita
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ox < out_width && oy < out_height && oz < out_depth)
    {
        float sum = 0.0f;
        int kz, ky, kx;
        float in_val, ker_val;
        kernel += (oz * ker_depth * ker_height * ker_width);

        int new_ox = ox;
        int new_oy = oy;

        for(kz = 0; kz < ker_depth; kz++)
        {
            for(ky = 0; ky < ker_height; ky++)
            {
                for(kx = 0; kx < ker_width; kx++)
                {
                    if((new_ox + kx) >= 0 && (new_ox + kx) < in_width && (new_oy + ky) >= 0 && (new_oy + ky) < in_height){
                        in_val = input[(new_oy + ky) * in_width + new_ox + kx];
                        ker_val = kernel[(ker_width * ker_height - 1) - (ky * ker_width + kx)];
                        sum += in_val * ker_val;
                    }
                }
            }
            input += (in_height * in_width);
            kernel += (ker_height * ker_width);
        }
        output[(oz * out_height * out_width) + oy * out_width + ox] = sum;
    }
}

__global__ void convolution_forNOutChannels(float *in, float *kernel, float *out, int in_w, int in_h, int in_d, int kernel_w, int kernel_h, int kernel_d, int out_w, int out_h, int out_d, int out_n)
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

    int ox = oxy % out_h;
    int oy = oxy / out_w;

    if (ox < out_w && oy < out_h && oz < out_d && on < out_n)
    {
        float sum = 0.0f;
        int ky, kx;
        float in_val, ker_val;
        kernel += (on * kernel_d * kernel_h * kernel_w);
        in += (oz * in_h * in_w);

        for(ky = 0; ky < kernel_h; ky++)
        {
            for(kx = 0; kx < kernel_w; kx++)
            {
                if((ox + kx) >= 0 && (ox + kx) < in_w && (oy + ky) >= 0 && (oy + ky) < in_h){
                    in_val = in[(oy + ky) * in_w + ox + kx];
                    ker_val = kernel[(kernel_w * kernel_h - 1) - (ky * kernel_w + kx)];
                    sum += in_val * ker_val;
                }
            }
        }
        out[(((on * out_d) + oz) * out_h + oy) * out_w + ox] = sum;
        //out[(on * out_d * out_h * out_w) + (oz * out_h * out_w) + oy * out_w + ox] = sum;
    }
}

__global__ void full_Convolution(float *input, float *kernel, float *output, int in_width, int in_height, int in_depth, int ker_width, int ker_height, int ker_depth, int ker_number, int out_width, int out_height, int out_depth, int padding) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x; //Numero di threads definiti in base alle dimensioni di uscita
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ox < out_width && oy < out_height && oz < out_depth)
    {
        float sum = 0.0f;
        int kz, ky, kx;
        float in_val, ker_val;
        input += (oz * in_height * in_width);

        int new_ox = ox - padding;
        int new_oy = oy - padding;

        for(kz = 0; kz < ker_number; kz++)
        {
            for(ky = 0; ky < ker_height; ky++)
            {
                for(kx = 0; kx < ker_width; kx++)
                {
                    if((new_ox + kx) >= 0 && (new_ox + kx) < in_width && (new_oy + ky) >= 0 && (new_oy + ky) < in_height){
                        in_val = input[(new_oy + ky) * in_width + kx + new_ox];
                        ker_val = kernel[(ker_height * ker_width - 1) - (ky * ker_width + kx)];
                        //ker_val = kernel[((new_oy + ky) * ker_width + new_ox + kx)];
                        sum += in_val * ker_val;
                    }
                }
            }
            input += (in_height * in_width * in_depth);
            kernel += (ker_height * ker_width * ker_depth);
        }
        output[(oz * out_height * out_width) + oy * out_width + ox] = sum;
    }
}