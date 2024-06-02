#include "shared.h"

/**
 * Versione con shared memory della convoluzione.
 * Permette di colcorare la covoluzione tra due matrici quadrate bidimensionali.
 * 
 * La dimensione del blocco sugli assi x e y è pari al minimo tra le dimensioni spaziali della matrice d'uscita, della matrice d'ingresso e 32,
 * in modo da non eccedere i 1024 thread massimi per blocco. La dimensione della griglia è tale da coprire tutta la matrice
 * d'uscita/ingresso nel caso in cui questa ecceda 32 nelle due dimensioni.
 * Inquesto caso consideriamo anche la dimensione d'ingresso nel calcolo della dimensione del blocco peché questo deve essere sufficientemente
 * grande da permettere di copiare la matrice quadrata d'ingresso sulla memoria condivisa.
 * 
 * Definiamo due buffer associandoli alla memoria globale in modo da potervi accedere più facilmente poi dopo copiamo i valori nella memoria 
 * condivisa sia l'ingresso che il kernel.
 * 
 * Dopo aver shiftato l'indice di inizio della matrice d'ingresso a seconda del thread in cui ci troviamo
 * iteriamo con due cicli annidati per tutta la matrice kernel. Incrementiamo un registro contatore con il prodotto tra un valore dell'ingresso e uno del kernel.
 * Alla fine assegnamo il valore alla matrice d'uscita con un incremento e non tramite un'assegnazione in modo da poter usare questa funzione anche per matrici
 * a tre dimensioni.
 * Questa soluzione non funzione in tutte le configurazioni delle matrici d'ingresso/uscita perch* non controlla
 * di trovarsi in una posizione valida all'interno della matrice d'ingresso.
 * 
 * Per matrici a tre dimensioni iteriamo la chiamata a questa fuznione im modo da calcolare tutte le facce della matrice d'uscita.
 * 
 * A seconda della profondità della matrice d'uscita e del numero di iterazioni richieste questa funzione viene chiamata un numero necessario di volte.
 * 
 * Prima di lavorare su una matrice quindi è necessario riazzerarne i valori.
*/

__global__ void convolution_shared(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim/*, int padding_f, int stride_f*/){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ float s_m[];
    float *data = &s_m[0];
    float *filter = &s_m[in_dim * in_dim];

    int index_data = idy * in_dim + idx;
    if(idx < in_dim && idy < in_dim){
        data[index_data] = in[index_data];
    }

    int index_filter = idy * kernel_dim + idx;
    int offset = kernel_dim * kernel_dim - 1;
    if(idx < kernel_dim && idy < kernel_dim){
        filter[index_filter] = kernel[offset - index_filter];
    }

    __syncthreads();

    if(idx < out_dim && idy < out_dim){

        int i, j;

        float tmp = 0;
        float val;

        data += idy * in_dim + idx;

        for(i = 0; i < kernel_dim; i++){
            for(j = 0; j < kernel_dim; j++){
                val = data[j];
                tmp += filter[j] * val;
            }
            filter += kernel_dim;
            data += in_dim;
        }
        out[idy * out_dim + idx] += tmp;
    }
}