#include "optimized.h"

/**
 * Versione ottimizzata dell'uso della memoria condivisa.
 * Permette di colcorare la covoluzione tra due matrici quadrate bidimensionali.
 * 
 * La dimensione del blocco sugli assi x e y è pari al minimo tra 32 e le dimensioni spaziali della matrice d'uscita e della matrice d'ingresso più due volte il padding,
 * in modo da non eccedere i 1024 thread massimi per blocco. La dimensione della griglia è tale da coprire tutta la matrice
 * d'uscita/ingresso + padding nel caso in cui questa ecceda 32 nelle due dimensioni.
 * In questo caso consideriamo anche la dimensione d'ingresso e due volte il padding nel calcolo della dimensione del blocco peché questo deve essere sufficientemente
 * grande da permettere di copiare la matrice quadrata d'ingresso sulla memoria condivisa.
 * 
 * Definiamo due buffer associandoli alla memoria globale in modo da potervi accedere più facilmente poi dopo copiamo i valori nella memoria 
 * condivisa sia l'ingresso che il kernel.
 * 
 * La dimensione della memoria condivisa che ospita l'ingresso possiede attorno una corona di dimensine pari al padding in modo da poter rimuovere il controllo
 * sugli indici all'interno del doppio ciclo for e quindi velocizzare l'esecuzine al prezzo di più memoria condivisa utilizzata.
 * 
 * Dopo aver shiftato l'indice di inizio della matrice d'ingresso a seconda del thread in cui ci troviamo
 * iteriamo con due cicli annidati per tutta la matrice kernel. Incrementiamo un registro contatore con il prodotto tra un valore dell'ingresso e uno del kernel.
 * Alla fine assegnamo il valore alla matrice d'uscita con un incremento e non tramite un'assegnazione in modo da poter usare questa funzione anche per matrici
 * a tre dimensioni.
 * 
 * Per matrici a tre dimensioni iteriamo la chiamata a questa fuznione im modo da calcolare tutte le facce della matrice d'uscita.
 * 
 * A seconda della profondità della matrice d'uscita e del numero di iterazioni richieste questa funzione viene chiamata un numero necessario di volte.
 * 
 * Prima di lavorare su una matrice quindi è necessario riazzerarne i valori.
*/

__global__ void convolution_optimized(float *in, float *out, float *kernel, int in_dim, int out_dim, int kernel_dim, int padding_f){ //in_dim = dimensioneffettiva dell'ingresso
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ float s_m[];
    float *filter = &s_m[0];
    float *data = &s_m[kernel_dim * kernel_dim]; //bisogna allocare abbastanza spazio da contenere sia l'ingresso che il padding associato all'ingresso

    int new_in_dim = in_dim + 2 * padding_f;
    if(idx < new_in_dim && idy < new_in_dim){
        data[idy * new_in_dim + idx] = 0;
    }
    if(idx < in_dim && idy < in_dim){
        data[(idy + padding_f) * new_in_dim + idx + padding_f] = in[idy * in_dim + idx];
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

        data += idy * new_in_dim + idx;

        for(i = 0; i < kernel_dim; i++){
            for(j = 0; j < kernel_dim; j++){
                val = data[j];
                tmp += filter[j] * val;
            }
            filter += kernel_dim;
            data += new_in_dim;
        }
        out[idy * out_dim + idx] += tmp;
    }
}