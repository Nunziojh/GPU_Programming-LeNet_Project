#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * Versione monolitica della convoluzione 3D.
 * Permette di colcorare la covoluzione tra una matrice d'ingresso e N matrici kernel, supponendo che abbiano tutte la stessa profondità.
 * I(i, i, b, c) * K(k, k, b, n) = O(o, o, n, 1)
 * Questa funzione viene utilizzata solamente nella fase di foreward della nostra rete.
 * 
 * La dimensione del blocco sugli assi x, y e z è pari al minimo tra 10 e le dimensioni spaziali della matrice d'uscita,
 * in modo da non eccedere i 1000 thread per blocco, lavorando con matrici quadrate abbiamo deciso di mantenere la struttura del blocco quadrata
 * a discapito di alcuni thread inutilizzati per ogni blocco. La dimensione della griglia è tale da coprire tutta la matrice
 * d'uscita nel caso in cui questa ecceda 10 nelle tre dimensioni.
 * 
 * Dopo aver shiftato l'indice di inizio del kernel da usare in base alla z della matrice di uscita su cui stiamo lavorando, 
 * iteriamo con tre cicli annidati per tutta la matrice kernel. Incrementiamo un registro contatore con il prodotto tra un valore dell'ingresso e uno del kernel.
 * Alla fine assegnamo il valore alla matrice d'uscita. Controlliamo di non eccedere le diensioni della matrice d'ingresso solo sugli assi x e y perché sul terzo asse
 * abbiamo lo stesso numero di valori.
*/

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

        for(kz = 0; kz < ker_depth; kz++)
        {
            for(ky = 0; ky < ker_height; ky++)
            {
                for(kx = 0; kx < ker_width; kx++)
                {
                    if((ox + kx) >= 0 && (ox + kx) < in_width && (oy + ky) >= 0 && (oy + ky) < in_height){
                        in_val = input[(oy + ky) * in_width + ox + kx];
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

/**
 * Versione monolitica della convoluzione 3D.
 * Permette di colcorare la covoluzione tra una matrice d'ingresso e N matrici kernel.
 * I(i, i, a, b) * K(k, k, 1, n) = O(o, o, a, n)
 * Questa funzione viene utilizzata solamente nella fase di backward per calcolare le derivate dei kernel della nostra rete.
 * 
 * Usiamo l'asse x del blocco per iterare lungo gli assi x e y dell'uscita, l'asse y viene usato iterare lungo l'asse z dell'uscita e l'asse
 * z del blocco viene usato per itarere su tutte le matrici d'uscita.
 * La dimensione del blocco è quindi pari a: min(10, out_width * out_height), min(10, out_depth) e min(10, out_number),
 * in modo da non eccedere i 1000 thread per blocco, lavorando con matrici quadrate abbiamo deciso di mantenere la struttura del blocco quadrata
 * a discapito di alcuni thread inutilizzati per ogni blocco. La dimensione della griglia è tale da coprire tutta la matrice
 * d'uscita nel caso in cui questa ecceda 10 nelle quattro dimensioni.
 * 
 * Dopo aver shiftato l'indice di inizio del kernel e dell'ingresso, 
 * iteriamo con due cicli annidati per tutta la matrice kernel. Usiamo solamente due cicli annidati, non considerando la profondità del kernel
 * perché nel nostro caso il kernel è sempre profondo uno sull'asse z. Incrementiamo un registro contatore con il prodotto tra un valore dell'ingresso e uno del kernel.
 * Alla fine assegnamo il valore alla matrice d'uscita. Controlliamo di non eccedere le diensioni della matrice d'ingresso solo sugli assi x e y perché sul terzo asse
 * abbiamo lo stesso numero di valori.
*/

__global__ void convolution_forNOutChannels(float *in, float *kernel, float *out, int in_w, int in_h, int in_d, int kernel_w, int kernel_h, int kernel_d, int out_w, int out_h, int out_d, int out_n)
{
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
    }
}

/**
 * Versione monolitica della convoluzione 3D.
 * Permette di colcorare la covoluzione tra una matrice d'ingresso e N matrici kernel.
 * I(i, i, a, b) * K(k, k, 1, b) = O(o, o, a, 1)
 * Questa funzione viene utilizzata solamente nella fase di backward per calcolare le derivate rispetto ai valori di ingresso delle convoluzioni.
 * 
 * La dimensione del blocco sugli assi x, y e z è pari al minimo tra 10 e le dimensioni spaziali della matrice d'uscita,
 * in modo da non eccedere i 1000 thread per blocco, lavorando con matrici quadrate abbiamo deciso di mantenere la struttura del blocco quadrata
 * a discapito di alcuni thread inutilizzati per ogni blocco. La dimensione della griglia è tale da coprire tutta la matrice
 * d'uscita nel caso in cui questa ecceda 10 nelle tre dimensioni.
 * 
 * Una volta definiti gli indici d'inizio di lavoro della matrice di ingresso, considerando il padding, iteraimo sul numero di kernel e sulle due dimensioni spaziali x e y,
 * non consideriamo l'asse z perché per la nostra applicazione i kernel sono sempre profondi 1. Controllando di non eccedere le dimensioni dell'ingresso calcoliamo il prodotto
 * tra i valori del kernel e quelli dell'ingresso.
*/

__global__ void full_Convolution(float *input, float *kernel, float *output, int in_width, int in_height, int in_depth, int ker_width, int ker_height, int ker_depth, int ker_number, int out_width, int out_height, int out_depth, int padding) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x; //Numero di threads definiti in base alle dimensioni di uscita
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ox < out_width && oy < out_height && oz < out_depth)
    {
        float sum = 0.0f;
        int kn, ky, kx;
        float in_val, ker_val;
        input += (oz * in_height * in_width);

        int new_ox = ox - padding;
        int new_oy = oy - padding;

        for(kn = 0; kn < ker_number; kn++)
        {
            for(ky = 0; ky < ker_height; ky++)
            {
                for(kx = 0; kx < ker_width; kx++)
                {
                    if((new_ox + kx) >= 0 && (new_ox + kx) < in_width && (new_oy + ky) >= 0 && (new_oy + ky) < in_height){
                        in_val = input[(new_oy + ky) * in_width + kx + new_ox];
                        ker_val = kernel[(ker_height * ker_width - 1) - (ky * ker_width + kx)];
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