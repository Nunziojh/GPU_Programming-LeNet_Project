#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "gpu_functions.h"
//#include "mnist.h"


__host__ void debug_print(float *matrice_dev, int w, int h, int c, int n){
    float *tmp = (float *)malloc(sizeof(float) * w * h * c * n);
    cudaMemcpy(tmp, matrice_dev, sizeof(float) * w * h * c * n, cudaMemcpyDeviceToHost);

    for(int l = 0; l < n; l++){
        for(int i = 0; i < c; i++){
            for(int j = 0; j < h; j++){
                for(int k = 0; k < w; k++){
                    printf("%e ",tmp[(((k + j * w) + i * (h * w)) + l * (h * w * c))]);
                }
                printf("\n");
            }
            printf("----\n");
        }
        printf("\n##########\n--------------------\n##########\n");
    }

    free(tmp);
    return;
}

int main(){
    srand(2);

    /***
     * Definizione dei parametri della rete.
    */
    const unsigned int padding = 0;
    const unsigned int stride_c = 1;
    const unsigned int stride_p = 2;
    const unsigned int kernel_num_first_layer = 6;
    const unsigned int kernel_num_second_layer = 16;
    const unsigned int kernel_num_third_layer = 120;
    const unsigned int fc_first_dim = 120;
    const unsigned int fc_second_dim = 84;
    const unsigned int fc_third_dim = 10;
    const unsigned int m = 1;        //batch size

    /*
        Definizione delle variabili generiche.
    */
    dim3 block, grid;
    int unsigned in_h = 32;
    int unsigned in_w = 32;
    int unsigned out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
    int unsigned out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;
    int unsigned h_2, w_2;
    int unsigned padding_full_conv;
    float summation;
    float loss;

    /****
     * Definizione, allocazione e inizializzazione randomica dei kernel e delle matrici dei pesi.
    */
    float *kernels_first_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer);
    float *kernels_second_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer);
    float *kernels_third_layer = (float *) malloc(sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer);
    float *fc_first_layer = (float *) malloc(sizeof(float) * fc_second_dim * fc_first_dim);         // W1 (84 x 120)
    float *fc_second_layer = (float *) malloc(sizeof(float) * fc_third_dim * fc_second_dim);        // W2 (10 x 84)
    float *prediction = (float *) malloc(sizeof(float) * fc_third_dim);
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer; i++) kernels_first_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer; i++) kernels_second_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer; i++) kernels_third_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < fc_first_dim * fc_second_dim; i++) fc_first_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < fc_second_dim * fc_third_dim; i++) fc_second_layer[i] = (float)rand() / (float)RAND_MAX;

    /***
     * Definizione e allocazione delle matrici e dei target in ingresso.
    */
    float target[10] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    float *img = (float *) malloc(sizeof(float) * in_h * in_w);
    // TODO: Aggiungere il padding all'immagine di ingresso (28 x 28) -> (32 x 32)
    for(int i = 0; i < in_w * in_w; i++) img[i] = i;

    /***
     * Definizione e allocazione dei kernel e delle matrici su device.
    */
    float *kernels_first_layer_dev, *first_conv, *first_pool;
    float *kernels_second_layer_dev, *second_conv, *second_pool;
    float *kernels_third_layer_dev, *third_conv;
    float *fc_first_layer_dev, *second_fc;
    float *fc_second_layer_dev, *third_fc;
    float *img_dev;
    float *prediction_dev, *target_dev;

    float *dZ2, *dW2;
    float *dZ1, *gdZ1, *dW1;
    float *dZ0, *gdZ0;
    float *dF2, *dA3;
    float *dP1, *dA2;
    float *dC1, *dA1;
    float *dF1, *dP0, *dA0;
    float *dC0, *dF0;
    
    cudaMalloc((void **)&kernels_first_layer_dev, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer);
    cudaMalloc((void **)&first_conv, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void **)&first_pool, sizeof(float) * 14 * 14 *6);

    cudaMalloc((void **)&kernels_second_layer_dev, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer);
    cudaMalloc((void **)&second_conv, sizeof(float) * 10 * 10 * 16);
    cudaMalloc((void **)&second_pool, sizeof(float) * 5 * 5 * 16);
    
    cudaMalloc((void **)&kernels_third_layer_dev, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer);
    cudaMalloc((void **)&third_conv, sizeof(float) * 1 * 1 * 120 * m);

    cudaMalloc((void **)&fc_first_layer_dev, sizeof(float) * fc_second_dim * fc_first_dim);
    cudaMalloc((void **)&second_fc, sizeof(float) * 84 * m);
    cudaMalloc((void **)&fc_second_layer_dev, sizeof(float) * fc_second_dim * fc_third_dim);
    cudaMalloc((void **)&third_fc, sizeof(float) * 10 * m);

    cudaMalloc((void **)&img_dev, sizeof(float) * in_w * in_w);
    
    cudaMalloc((void **)&prediction_dev, sizeof(float) * fc_third_dim);
    cudaMalloc((void **)&target_dev, sizeof(float) * fc_third_dim);

    cudaMalloc((void **)&dZ2, sizeof(float) * fc_third_dim * m);
    cudaMalloc((void **)&dW2, sizeof(float) * fc_third_dim * fc_second_dim);
    cudaMalloc((void **)&dZ1, sizeof(float) * fc_second_dim * m);
    cudaMalloc((void **)&gdZ1, sizeof(float) * fc_second_dim * m);
    cudaMalloc((void **)&dW1, sizeof(float) * fc_second_dim * fc_first_dim);
    cudaMalloc((void **)&dZ0, sizeof(float) * fc_first_dim * m);
    cudaMalloc((void **)&gdZ0, sizeof(float) * fc_first_dim * m);
    cudaMalloc((void **)&dF2, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer);
    cudaMalloc((void **)&dA3, sizeof(float) * 5 * 5 * kernel_num_second_layer);
    cudaMalloc((void **)&dP1, sizeof(float) * 5 * 5 * kernel_num_second_layer);
    cudaMalloc((void **)&dA2, sizeof(float) * 10 * 10 * kernel_num_second_layer);
    cudaMalloc((void **)&dC1, sizeof(float) * 10 * 10 * kernel_num_second_layer);
    cudaMalloc((void **)&dA1, sizeof(float) * 14 * 14 * kernel_num_first_layer);
    cudaMalloc((void **)&dF1, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer);
    cudaMalloc((void **)&dP0, sizeof(float) * 14 * 14 * kernel_num_first_layer);
    cudaMalloc((void **)&dA0, sizeof(float) * 28 * 28 * kernel_num_first_layer);
    cudaMalloc((void **)&dC0, sizeof(float) * 28 * 28 * kernel_num_first_layer);
    cudaMalloc((void **)&dF0, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer);

    /****
     * Copia dei valori dei parametri e dell'immagine di ingresso sul dispositivo.
    */
    cudaMemcpy(kernels_first_layer_dev, kernels_first_layer, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(kernels_second_layer_dev, kernels_second_layer, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer * kernel_num_second_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(kernels_third_layer_dev, kernels_third_layer, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(fc_first_layer_dev, fc_first_layer, sizeof(float) * fc_first_dim * fc_second_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(fc_second_layer_dev, fc_second_layer, sizeof(float) * fc_second_dim * fc_third_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(img_dev, img, sizeof(float) * in_w * in_h, cudaMemcpyHostToDevice);

    /*----------------------------------------------------------------------------------------------------------------------------
        La successiva riga di codice serve a normalizzare i valori nell'intervallo [0, 1] così che la prima
        attivazione della tanh non debba lavorare con valori altissimo e dia problemi di calcolo.
        Questo lavoro di normalizzazione viene fatto in automatico nei livelli successivi per via della
        presenza della funzione tanh che riporta tutti i valori nell'intervallo [-1, 1].
        Per ora questa prima normalizzazione non sottrae la media ma divide solo per la deiazione standard.    
    */
    matrix_scalar_product<<<{1, 1}, {32, 32}>>>(img_dev, (float)(1.0 / (float)(1024)), 32, 32, 0);


    /****
     * Inizio del ciclo per fare apprendimento. L'indice da usare è il numero di epoche
     * per le quali si vuole addestrare la rete.
    */
    for(int epoch = 0; epoch < 4; epoch++){
        for(int batch_dim = 0; batch_dim < 1; batch_dim++){

            /****
             * Calcolo del primo layer convolutivo con la relativa funzione di attivazion tanh
             * Partiamo dall'immagine in ingresso salvata nella variabile 'img_dev' con dimensioni (32 x 32)
             * e ne facciamo la convoluzione valid, (0 di padding e 1 di stride) con 6 kernel di dimensioni (5 x 5)
             * salvati, nella variabile 'kernels_first_layer_dev', come vettore.
            */
            in_h = 32;
            in_w = 32;
            out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
            out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
            for(int i = 0; i < kernel_num_first_layer; i++){
                convolution<<<grid, block>>>(img_dev, first_conv + (i * out_h * out_w), kernels_first_layer_dev + (i * KERNEL_DIM * KERNEL_DIM), out_w, out_h, padding, stride_c);
                tanh<<<grid, block>>>(first_conv + (i * out_h * out_w), out_w, out_h, 0);
            }

            /****
             * Calcoliamo il primo layer di Average Pooling con la relativa funzione di attivazione tanh.
             * Partiamo dal risultato del livello precedente salvato nella variabile 'first_conv', come vettore di dimensioni
             * (28 x 28 x 6), e applichiamo il filtro e otteniamo un risultato, salvato nella variabile 'first_pool', di dimensione
             * (14 x 14 x 6), sempre memorizzandolo come un vettore.
            */
            in_h = out_h;
            in_w = out_w; 
            out_h = (in_h - POOLING_WINDOW_SIZE) / stride_p + 1;
            out_w = (in_w - POOLING_WINDOW_SIZE) / stride_p + 1;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
            for(int i = 0; i < kernel_num_first_layer; i++){
                avg_pooling<<<grid, block>>>(first_conv + (i * in_h * in_w), first_pool + (i * out_h * out_w), in_h, in_w, out_h, out_w, stride_p);
                tanh<<<grid, block>>>(first_pool + (i * out_h * out_w), out_w, out_h, 0);
            }

            /****
             * Calcoliamo il secondo layer convolutivo a partire dall'uscita del layer di pooling precedente,
             * le dimensioni di ingresso sono (14 x 14 x 6), usiamo 16 kernel di dimensioni (5 x 5) e otteniamo
             * un valore di uscita, salvato nella variabile second_conv di dimensione (10 x 10 x 16).
            */
            in_h = out_h;
            in_w = out_w;
            out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
            out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
            // Prima di calcolare i nuovi valori ripuliamo la matrice second_conv perché la convolutin3D sovrascrive i valori già presenti inmemoria.
            clean_vector<<<((out_h * out_w * kernel_num_second_layer) / 1024 + 1), 1024>>>(second_conv, out_h * out_w * kernel_num_second_layer);
            for(int i = 0; i < kernel_num_second_layer; i++){
                for(int j = 0; j < kernel_num_first_layer; j++){
                    convolution3D<<<grid, block>>>(first_pool + (j * in_h * in_w), second_conv + (i * out_h * out_w), kernels_second_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer)), out_h, out_w, padding, stride_c, KERNEL_DIM);
                }
                tanh<<<grid, block>>>(second_conv + (i * out_h * out_w), out_w, out_h, 0);
            }

            //debug_print(kernels_second_layer_dev, 5, 5, 6, 1);

            /****
             * Calcoliamo il secondo layer di Average Pooling partendo da una matrice di dimensini (10 x 10 x 16)
             * e otteniamo una matrice di dimensioni (5 x 5 x 16) che salviamo come vettore in second_pool.
            */
            in_h = out_h;
            in_w = out_w;
            out_h = (in_h - POOLING_WINDOW_SIZE) / stride_p + 1;
            out_w = (in_w - POOLING_WINDOW_SIZE) / stride_p + 1;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
            for(int i = 0; i < kernel_num_second_layer; i++){
                avg_pooling<<<grid, block>>>(second_conv + (i * in_h * in_w), second_pool + (i * out_h * out_w), in_h, in_w, out_h, out_w, stride_p);
                tanh<<<grid, block>>>(second_pool + (i * out_h * out_w), out_w, out_h, 0);
            }

            /****
             * Calcoliamo il terzo layer convolutivo a partire dall'uscita del layer di pooling precedente,
             * le dimensioni di ingresso sono (5 x 5 x 16), usiamo 120 kernel di dimensioni (5 x 5) e otteniamo
             * un valore di uscita, salvato nella variabile third_conv di dimensione (1 x 1 x 120).
            */
            in_h = out_h;
            in_w = out_w;
            out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
            out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
            // Prima di calcolare i nuovi valori ripuliamo la matrice second_conv perché la convolutin3D sovrascrive i valori già presenti inmemoria.
            clean_vector<<<((out_h * out_w * kernel_num_third_layer) / 1024), 1024>>>(third_conv, out_h * out_w * kernel_num_third_layer);
            for(int i = 0; i < kernel_num_third_layer; i++){
                for(int j = 0; j < kernel_num_second_layer; j++){
                    convolution3D<<<grid, block>>>(second_pool + (j * in_h * in_w), third_conv + (i * out_h * out_w), kernels_third_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer)), out_h, out_w, padding, stride_c, KERNEL_DIM);
                }
                tanh<<<grid, block>>>(third_conv + (i * out_h * out_w), out_w, out_h, 0);
            }

            /****
             * A partire dalla matrice di dimensini (120 x m) ottenuta dall'ultimo layer convolutivo, calcoliamo il primo
             * livello di Fully Connected usando come matrice di pesi la variabile 'fc_first_layer_dev' di dimensioni
             * (84 x 120). Otteniamo una matrice risultato di dimensioni (84 x m).
            */
            in_h = fc_first_dim;
            in_w = m;
            out_h = fc_second_dim;
            out_w = m;
            block = {(unsigned int)m, (unsigned int)fc_second_dim};
            grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
            matrix_product<<<grid, block>>>(fc_first_layer_dev, third_conv, second_fc, m, fc_second_dim, fc_first_dim);
            tanh<<<grid, block>>>(second_fc, 1, fc_second_dim, 0);

            /****
             * A partire dalla matrice di dimensini (84 x m) ottenuta al livello FC precedente, calcoliamo il secondo
             * livello di Fully Connected usando come matrice di pesi la variabile 'fc_second_layer_dev' di dimensioni
             * (10 x 84). Otteniamo una matrice risultato di dimensioni (10 x m).
            */
            in_h = fc_second_dim;
            in_w = m;
            out_h = fc_third_dim;
            out_w = m;
            block = {(unsigned int)m, (unsigned int)fc_third_dim};
            grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
            matrix_product<<<grid, block>>>(fc_second_layer_dev, second_fc, third_fc, m, fc_third_dim, fc_second_dim);

            /****
             * Calcoliamo la funzione di Loss calcolando prima gli esponenziali sul dispositivo e poi portiamo i valori
             * sull'host per poter terminare il calcolo della Softmax facendo la sommatoria di tutti i valori ottenuti
             * per poi dividere ogni elemento del vettore per il valore calcolato.
            */
            exponential<<<grid, block>>>(third_fc, fc_third_dim);
            cudaMemcpy(prediction, third_fc, sizeof(float) * fc_third_dim, cudaMemcpyDeviceToHost);
            summation = 0.0;
            for(int i = 0; i < fc_third_dim; i++) summation += prediction[i];
            for(int i = 0; i < fc_third_dim; i++) {
                prediction[i] = prediction[i] / summation;
                printf("\t\t%f\n", prediction[i]);
            }
            //for(int i = 0; i < fc_third_dim; i++) printf("%2.2f\n", prediction[i]);

            /****
             * Calcoliamo il valore della Loss.
            */
            loss = 0.0;
            for(int i = 0; i < fc_third_dim; i++) loss += target[i] * log(prediction[i]);
            loss = -loss;
            printf("Loss = %e\n", loss);

            // Inizio della BackPropagation





            // TODO: CONTROLLARE LA CORRETTEZZA DEL CALCOLO CHE VA DAL VALORE DELLA LOSS A DZ2
            cudaMemcpy(prediction_dev, prediction, sizeof(float) * fc_third_dim, cudaMemcpyHostToDevice);
            cudaMemcpy(target_dev, target, sizeof(float) * fc_third_dim, cudaMemcpyHostToDevice);

            //float *dZ2;                             // Da moltiplicare per il numero di immagini usate nel batch
            block = {(unsigned int)fc_third_dim};
            grid = {(unsigned int)(block.x / 1024 + 1)};
            subtraction<<<grid, block>>>(dZ2, prediction_dev, target_dev, fc_third_dim);
            // FINE CONTROLLO


            //debug_print(dZ2, 1, 10, 1, 1);


            /****
             * Calcoliamo la derivata dei pesi tra il secondo e il terzo livello FC.
             * dW2 = dZ2 * A1^T
             * dW2 (10 x 84)
             * dZ2 (10 x m) Derivata della funzione di Loss fino a questo punto
             * A1 (84 x m) Attivazioni del livello precedente
            */
            in_h = fc_third_dim;
            in_w = m;
            out_h = fc_third_dim;
            out_w = fc_second_dim;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            matrix_product_transpose<<<grid, block>>>(dZ2, second_fc, dW2, out_w, out_h, in_w);
            matrix_scalar_product<<<grid, block>>>(dW2, (1.0 / m), out_w, out_h, 0);
        
            /****
             * Calcoliamo la derivata delle uscite del secondo layer FC.
             * dA1 = W2^T * dZ2
             * dZ1 = dA1 .* (1 - A1^2) --> Derivata della tangente iperbolica
             * dA1 (84 x m)
             * W2 (10 x 84)
             * dZ2 (10 x m)
             * A1 (84 x m) Attivazioni del livello precedente
            */
            in_h = fc_third_dim;
            in_w = m;
            out_h = fc_second_dim;
            out_w = m;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            matrix_transpose_product<<<grid, block>>>(fc_second_layer_dev, dZ2, dZ1, out_w, out_h, in_h);
            matrix_dot_product<<<grid, block>>>(second_fc, second_fc, gdZ1, out_w, out_h);
            scalar_subtraction<<<grid, block>>>(gdZ1, gdZ1, out_w, out_h);
            matrix_dot_product<<<grid, block>>>(dZ1, gdZ1, dZ1, out_w, out_h);

            /****
             * Calcoliamo la derivata dei pesi tra il primo e il secondo livello FC.
             * dW1 = dZ1 * A0^T
             * dW1 (84 x 120)
             * dZ1 (84 x m) Derivata della funzione di Loss fino a questo punto
             * A0 (120 x m) La matrice delle attivazioni dell'ultimo layer convolutivo raccolte per colonne di altezza 120
            */
            in_h = fc_second_dim;
            in_w = m;
            out_h = fc_second_dim;
            out_w = fc_first_dim;
            block = {(unsigned int)32, (unsigned int)32};
            grid = {(unsigned int)(out_w / 32 + 1), (unsigned int)(out_h / 32 + 1)};
            matrix_product_transpose<<<grid, block>>>(dZ1, third_conv, dW1, out_w, out_h, in_w);
            matrix_scalar_product<<<grid, block>>>(dW1, (1.0 / m), out_w, out_h, 0);

            /****
             * Calcoliamo la derivata delle uscite del secondo layer FC.
             * dA0 = W1^T * dZ1
             * dZ0 = dA0 .* (1 - A0^2) --> Derivata della tangente iperbolica
             * dA0 (120 x m)
             * W1 (84 x 120)
             * dZ1 (84 x m)
             * A0 (120 x m) La matrice delle attivazioni dell'ultimo layer convolutivo raccolte per colonne di altezza 120
            */
            in_h = fc_second_dim;
            in_w = m;
            out_h = fc_first_dim;
            out_w = m;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            matrix_transpose_product<<<grid, block>>>(fc_first_layer_dev, dZ1, dZ0, out_w, out_h, in_h);
            matrix_dot_product<<<grid, block>>>(third_conv, third_conv, gdZ0, out_w, out_h);
            scalar_subtraction<<<grid, block>>>(gdZ0, gdZ0, out_w, out_h);
            matrix_dot_product<<<grid, block>>>(dZ0, gdZ0, dZ0, out_w, out_h);

            /****
             * Calcoliamo la derivata del terzo gruppo di kernel che di dimensioni (5 x 5 x 16), di cui
             * ne abbiamo 120.
             * La convoluzione, mantenendo fisso il canale dell'uscita dZ0, itera su tutti i canali dell'ingresso.
             * Aggiorniamo il valore di tutte le derivate dei filtri.
             * Utilizziamo la funzione convolution3D che ci permette di aggiornare i valori sulla matrice di output
             * sommandoli invece di sovrascriverli.
            */
            in_h = 5;
            in_w = 5;
            h_2 = 1;
            w_2 = 1;
            out_h = KERNEL_DIM;
            out_w = KERNEL_DIM;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            clean_vector<<<(5 * 5 * 16 * 120 / 1024 + 1), 1024>>>(dF2, 5 * 5 * 16 * 120); // 5 * 5 * 16 * 120 = 48000 -> dimensione dei kernel tra il secondo e il terso layer convolutivo, di cui ne abbiamo 120 
            for(int i = 0; i < kernel_num_third_layer; i++){
                for(int j = 0; j < kernel_num_second_layer; j++){
                    convolution3D<<<grid, block>>>(second_pool + (j * in_h * in_w), dF2 + (j * out_w * out_h + (i * out_w * out_h * kernel_num_second_layer)), dZ0 + (i * h_2 * w_2), KERNEL_DIM, KERNEL_DIM, padding, stride_c, h_2);
                }
            }

            /****
             * Cacoliamo la Full Convolution tra i kernel della terza convoluzione e dZ0 che è
             * la derivata, rispetto alla Loss, calcolata fino all'uscita della terza convoluzione.
             * h_2 e w_2 si riferiscono alle dimensioni spaziali della matrice dZ0.
             * Per calcolare le dimensine di uscita usiamo la formual standard.
             * Iteriamo su tutti i canali dei kernel, mantenendo fisso il canale di dZ0 per un intero ciclo
             * di j. Utilizziamo la funzione convolution3D che ci permette di aggiornare i valori di dA3.
            */
            in_h = KERNEL_DIM;
            in_w = KERNEL_DIM;
            h_2 = 1;
            w_2 = 1;
            padding_full_conv = h_2 - 1;
            out_h = (in_h + 2 * padding_full_conv - h_2) / stride_c + 1;
            out_w = (in_w + 2 * padding_full_conv - w_2) / stride_c + 1;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            clean_vector<<<1, 400>>>(dA3, 5 * 5 * 16); //5 * 5 * 16 = 400
            for(int i = 0; i < kernel_num_third_layer; i++){
                for(int j = 0; j < kernel_num_second_layer; j++){
                    convolution3D<<<grid, block>>>(kernels_third_layer_dev + (j * in_h * in_w + (i * in_h * in_w * kernel_num_second_layer)), dA3 + (j * out_h * out_w), dZ0 + (i * h_2 * w_2), out_h, out_w, padding_full_conv, stride_c, h_2);
                }
            }

            /****
             * Calcoliamo la derivata delle uscite del secondo layer di Pooling.
             * Iteriamo su tutti canali singolarmente e per ognuno calcoliamo
             * la derivata della funzione tangente iperbolica.
             * dP1 = dA3 * (1 - A3^2)
             * Le dimensioni sono (5 x 5)
            */
            in_h = 5;
            in_w = 5;
            out_h = in_h;
            out_w = in_w;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            for(int i = 0; i < kernel_num_second_layer; i++){
                matrix_dot_product<<<grid, block>>>(second_pool + (i * in_h * in_w), second_pool + (i * in_h * in_w), dP1, out_h, out_w);
                scalar_subtraction<<<grid, block>>>(dP1 + (i * in_w * in_h), dP1 + (i * in_w * in_h), out_w, out_h);
                matrix_dot_product<<<grid, block>>>(dP1 + (i * in_h * in_w), dA3 + (i * in_h * in_w), dP1 + (i * in_h * in_w), out_w, out_h);
            }

            /****
             * Derivata rispetto all'uscita del secondo layer di Pooling.
             * Ogni valore della matrici di ingresso dP1 viene moltiplicato per il valore
             * proporzionato rispetto a tutti i valori all'interno della regione di Pooling.
             * La matrice ottenuta corrisponde alla matrice delle derivate della funzione di Loss
             * rispetto agli ingressi.
            */
            in_h = 5;
            in_w = 5;
            out_h = in_h * 2;
            out_w = in_w * 2;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            for(int i = 0; i < kernel_num_second_layer; i++){
                inverse_avg_pooling<<<grid, block>>>(dP1 + (i * in_h * in_w), dA2 + (i * out_h * out_w), second_conv + (i * out_h * out_w), in_w, in_h, out_w, out_h, stride_p);
            }

            /****
             * Calcoliamo la derivata delle uscite del secondo layer di Convoluzione.
             * Iteriamo su tutti canali singolarmente e per ognuno calcoliamo
             * la derivata della funzione tangente iperbolica.
             * dC1 = dA2 * (1 - A2^2)
             * Le dimensioni sono (10 x 10)
            */
            in_h = 10;
            in_w = 10;
            out_h = in_h;
            out_w = in_w;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            for(int i = 0; i < kernel_num_second_layer; i++){
                matrix_dot_product<<<grid, block>>>(second_conv + (i * in_w * in_h), second_conv + (i * in_w * in_h), dC1, out_w, out_h);
                scalar_subtraction<<<grid, block>>>(dC1 + (i * in_h * in_w), dC1 + (i * in_h * in_w), out_w, out_h);
                matrix_dot_product<<<grid, block>>>(dC1 + (i * in_h * in_w), dA2 + (i * in_h * in_w), dC1 + (i * in_h * in_w), out_w, out_h);
            }

            /****
             * Cacoliamo la Full Convolution tra i kernel della seconda convoluzione, F1, e dC1 che è
             * la derivata, rispetto alla Loss, calcolata fino all'uscita della seconda convoluzione.
             * h_2 e w_2 si riferiscono alle dimensioni spaziali della matrice dC1.
             * Per calcolare le dimensine di uscita usiamo la formual standard.
             * Iteriamo su tutti i canali dei kernel, mantenendo fisso il canale di dC1 per un intero ciclo
             * di j. Utilizziamo la funzione convolution3D che ci permette di aggiornare i valori di dA1.
            */
            in_h = KERNEL_DIM;
            in_w = KERNEL_DIM;
            h_2 = 10;
            w_2 = 10;
            padding_full_conv = h_2 - 1;
            out_h = (in_h + 2 * padding_full_conv - h_2) / stride_c + 1;
            out_w = (in_w + 2 * padding_full_conv - w_2) / stride_c + 1;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            clean_vector<<<2, 1024>>>(dA1, 14 * 14 * 6);// 14 * 14 * 6 = 1176, servono due blocchi, provare ad usarne due da 600 invece che da 1024.
            for(int i = 0; i < kernel_num_second_layer; i++){
                for(int j = 0; j < kernel_num_first_layer; j++){
                    convolution3D<<<grid, block>>>(kernels_second_layer_dev + (j * in_h * in_w + (i * in_h * in_w * kernel_num_first_layer)), dA1 + (j * out_h * out_w), dC1 + (i * h_2 * w_2), out_h, out_w, padding_full_conv, stride_c, h_2);
                }
            }

            /****
             * Calcoliamo la derivata del secondo gruppo di kernel di dimensioni (5 x 5 x 6), di cui
             * ne abbiamo 16.
             * La convoluzione, mantenendo fisso il canale dell'uscita dC1, itera su tutti i canali dell'ingresso.
             * Aggiorniamo il valore di tutte le derivate dei filtri.
             * Utilizziamo la funzione convolution3D che ci permette di aggiornare i valori sulla matrice di output
             * sommandoli invece di sovrascriverli.
            */
            in_h = 14;
            in_w = 14;
            h_2 = 10;
            w_2 = 10;
            out_h = KERNEL_DIM;
            out_w = KERNEL_DIM;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            clean_vector<<<3, 1024>>>(dF1, 5 * 5 * 6 * 16);// 5 * 5 * 6 * 16 = 2400, servono tre blocchi, provare ad usarne tre da 800 invece che da 1024.
            for(int i = 0; i < kernel_num_second_layer; i++){
                for(int j = 0; j < kernel_num_first_layer; j++){
                    convolution3D<<<grid, block>>>(first_pool + (j * in_h * in_w), dF1 + (j * out_h * out_w + (i * out_h * out_w * kernel_num_first_layer)), dC1 + (i * h_2 * w_2), out_w, out_h, padding, stride_c, h_2);
                }
            }

            /****
             * Calcoliamo la derivata delle uscite del secondo layer di Pooling.
             * Iteriamo su tutti canali singolarmente e per ognuno calcoliamo
             * la derivata della funzione tangente iperbolica.
             * dP0 = dA1 * (1 - A1^2)
             * Le dimensioni sono (14 x 14)
            */
            in_h = 14;
            in_w = 14;
            out_h = in_h;
            out_w = in_w;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            for(int i = 0; i < kernel_num_first_layer; i++){
                matrix_dot_product<<<grid, block>>>(first_pool + (i * in_h * in_w), first_pool + (i * in_h * in_w), dP0, out_w, out_h);
                scalar_subtraction<<<grid, block>>>(dP0 + (i * in_h * in_w), dP0 + (i * in_h * in_w), out_w, out_h);
                matrix_dot_product<<<grid, block>>>(dP0 + (i * in_h * in_w), dA1 + (i * in_h * in_w), dP0 + (i * in_h * in_w), out_w, out_h);
            }

            /****
             * Derivata rispetto all'uscita del primo layer di Pooling.
             * Ogni valore della matrici di ingresso dP0 viene moltiplicato per il valore
             * proporzionato rispetto a tutti i valori all'interno della regione di Pooling.
             * La matrice ottenuta, dA0, corrisponde alla matrice delle derivate della funzione di Loss
             * rispetto agli ingressi.
            */
            in_h = 14;
            in_w = 14;
            out_h = in_h * 2;
            out_w = in_w * 2;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            for(int i = 0; i < kernel_num_second_layer; i++){
                inverse_avg_pooling<<<grid, block>>>(dP0 + (i * in_h * in_w), dA0 + (i * out_w * out_h), first_conv + (i * out_w * out_h), in_w, in_h, out_w, out_h, stride_p);
            }
        
            /****
             * Calcoliamo la derivata delle uscite del primo layer di Convoluzione.
             * Iteriamo su tutti canali singolarmente e per ognuno calcoliamo
             * la derivata della funzione tangente iperbolica.
             * dC0 = dA0 * (1 - A0^2)
             * Le dimensioni sono (28 x 28)
            */
            in_h = 28;
            in_w = 28;
            out_h = in_h;
            out_w = in_w;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            for(int i = 0; i < kernel_num_first_layer; i++){
                matrix_dot_product<<<grid, block>>>(first_conv + (i * in_w * in_h), first_conv + (i * in_w * in_h), dC0, out_w, out_h);
                scalar_subtraction<<<grid, block>>>(dC0 + (i * in_w * in_h), dC0 + (i * in_w * in_h), out_w, out_h);
                matrix_dot_product<<<grid, block>>>(dC0 + (i * in_w * in_h), dA0 + (i * in_w * in_h), dC0 + (i * in_w * in_h), out_w, out_h);
            }

            /****
             * Calcoliamo la derivata del secondo gruppo di kernel di dimensioni (5 x 5), di cui
             * ne abbiamo 6.
             * La convoluzione, mantenendo fisso il canale dell'uscita dC0, itera su tutti i canali dell'ingresso.
             * Aggiorniamo il valore di tutte le derivate dei filtri.
             * Utilizziamo la funzione convolution3D che ci permette di aggiornare i valori sulla matrice di output
             * sommandoli invece di sovrascriverli.
            */
            in_h = 32;
            in_w = 32;
            h_2 = 28;
            w_2 = 28;
            out_h = KERNEL_DIM;
            out_w = KERNEL_DIM;
            block = {(unsigned int)out_w, (unsigned int)out_h};
            grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            clean_vector<<<1, 150>>>(dF0, 5 * 5 * 6);
            for(int i = 0; i < kernel_num_first_layer; i++){
                convolution3D<<<grid, block>>>(img_dev, dF0 + (i * out_w * out_h * kernel_num_first_layer), dC0 + (i * w_2 * h_2), out_w, out_h, padding, stride_c, h_2);
            }
        }

        debug_print(dF1, 5, 5, 6, 1);

        /*-------------------------------
            Fine calcolo delle derivate.
            Inizio aggiornamento dei parametri
        */
        block = {1024};
        grid = {(5 * 5 * 6) / 1024 + 1};
        matrix_scalar_product<<<grid, block>>>(dF0, LEARNING_RATE, 5 * 5 * 6, 1, 0);
        subtraction<<<grid, block>>>(kernels_first_layer_dev, kernels_first_layer_dev, dF0, 5 * 5 * 6);

        block = {1024};
        grid = {(5 * 5 * 6 * 16) / 1024 + 1};
        matrix_scalar_product<<<grid, block>>>(dF1, LEARNING_RATE, 5 * 5 * 6 * 16, 1, 0);
        subtraction<<<grid, block>>>(kernels_second_layer_dev, kernels_second_layer_dev, dF1, 5 * 5 * 6 * 16);

        block = {1024};
        grid = {(5 * 5 * 6 * 120) / 1024 + 1};
        matrix_scalar_product<<<grid, block>>>(dF2, LEARNING_RATE, 5 * 5 * 6 * 120, 1, 0);
        subtraction<<<grid, block>>>(kernels_third_layer_dev, kernels_third_layer_dev, dF2, 5 * 5 * 6 * 120);

        block = {1024};
        grid = {(84 * 120) / 1024 + 1};
        matrix_scalar_product<<<grid, block>>>(dW1, LEARNING_RATE, 84 * 120, 1, 0);
        subtraction<<<grid, block>>>(fc_first_layer_dev, fc_first_layer_dev, dW1, 84 * 120);

        block = {1024};
        grid = {(10 * 84) / 1024 + 1};
        matrix_scalar_product<<<grid, block>>>(dW2, LEARNING_RATE, 10 * 84, 1, 0);
        subtraction<<<grid, block>>>(fc_second_layer_dev, fc_second_layer_dev, dW2, 10 * 84);

    }

    free(kernels_first_layer);
    free(kernels_second_layer);
    free(kernels_third_layer);
    free(fc_first_layer);
    free(fc_second_layer);
    free(prediction);
    free(img);

    

    cudaFree(kernels_first_layer_dev); //
    cudaFree(kernels_second_layer_dev); //
    cudaFree(kernels_third_layer_dev); //
    cudaFree(fc_first_layer_dev); //
    cudaFree(fc_second_layer_dev); //

    cudaFree(img_dev); //
    cudaFree(first_conv); //
    cudaFree(first_pool); //
    cudaFree(second_conv); //
    cudaFree(second_pool); //
    cudaFree(third_conv); //
    cudaFree(second_fc); //
    cudaFree(third_fc); //

    cudaFree(prediction_dev);
    cudaFree(target_dev);

    cudaFree(dZ2);
    cudaFree(dW2);
    cudaFree(dZ1);
    cudaFree(gdZ1);
    cudaFree(dW1);
    cudaFree(dZ0);
    cudaFree(gdZ0);
    cudaFree(dF2);
    cudaFree(dA3);
    cudaFree(dP1);
    cudaFree(dA2);
    cudaFree(dC1);
    cudaFree(dA1);
    cudaFree(dF1);
    cudaFree(dP0);
    cudaFree(dA0);
    cudaFree(dC0);
    cudaFree(dF0);

    return 0;
}