#include <cuda_runtime.h>
#include <stdlib.h>

/**
 * Versione base della convoluzione.
 * Permette di colcorare la covoluzione tra due matrici quadrate bidimensionali.
 * 
 * La dimensione del blocco sugli assi x e y è pari al minimo tra le dimensioni spaziali della matrice d'uscita e 32, in modo da non eccedere i 1024
 * thread massimi per blocco. La dimensione della griglia è tale da coprire tutta la matrice d'uscita nel caso in cui questa ecceda 32 nelle due dimensioni.
 * 
 * Dopo aver calcolato l'indice di partenza relativo alla matrice di partenza calcolato considerando il padding (non consideriamo lo stride perché nel nostro
 * contesto non lo usiamo mai) iteriamo con due cicli annidati per tutta la matrice kernel controllando ad ogni passo di essere all'interno della matrice di 
 * input. Se la condizione è rispettata allora incrementiamo un registro contatore con il prodotto tra un valore dell'ingresso e uno del kernel.
 * Alla fine assegnamo il valore alla matrice d'uscita con un incremento e non tramite un'assegnazione in modo da poter usare questa funzione anche per matrici
 * a tre dimensioni.
 * 
 * Per matrici a tre dimensioni iteriamo la chiamata a questa fuznione im modo da calcolare tutte le facce della matrice d'uscita.
 * 
 * A seconda della profondità della matrice d'uscita e del numero di iterazioni richieste questa funzione viene chiamata un numero necessario di volte.
 * 
 * Prima di lavorare su una matrice quindi è necessario riazzerarne i valori.
*/

__global__ void convolution_base(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx < out_dim && idy < out_dim){

        int i, j;

        float tmp = 0;
        float val;

        int new_idx = idx - padding_f;
        int new_idy = idy - padding_f;

        for(i = 0; i < kernel_dim; i++){
            for(j = 0; j < kernel_dim; j++){
                val = ((new_idy + i) < 0 || (new_idy + i) >= in_dim || (new_idx + j) < 0 || (new_idx + j) >= in_dim) ? 0 : in[(new_idy + i) * in_dim + new_idx + j];
                tmp += kernel[(kernel_dim * kernel_dim - 1) - (i * kernel_dim + j)] * val;
            }
        }
        out[idy * out_dim + idx] += tmp;
    }
}