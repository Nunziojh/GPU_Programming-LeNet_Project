#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "gpu_functions.h"

int main(){
    srand(time(NULL));

    /***
     * Definizione dei parametri della rete.
    */
    int padding = 0;
    const int stride_c = 1;
    const int stride_p = 2;
    const int kernel_num_first_layer = 6;
    const int kernel_num_second_layer = 16;
    const int kernel_num_third_layer = 120;
    const int fc_first_dim = 120;
    const int fc_second_dim = 84;
    const int fc_third_dim = 10;
    const int m = 1;        //batch size

    /*
        Definizione delle variabili generiche.
    */
    dim3 block, grid;
    int in_h = 32;
    int in_w = 32;
    int out_h = (in_h + 2 * padding - KERNEL_DIM) / stride_c + 1;
    int out_w = (in_w + 2 * padding - KERNEL_DIM) / stride_c + 1;
    int h_2, w_2;
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
    for(int i = 0; i < in_w * in_w; i++) img[i] = i;

    /***
     * Definizione e allocazione dei kernel e delle matrici su device.
    */
    float *kernels_first_layer_dev, *first_conv, *first_pool;
    float *kernels_second_layer_dev, *second_conv, *second_pool;
    float *kernels_third_layer_dev, *third_conv;
    float *fc_first_layer_dev, *second_fc;
    float *fc_second_layer_dev, *third_fc;
    float *img_dev, *prediction_dev, *target_dev;

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
    cudaMalloc((void **)&third_conv, sizeof(float) * 1 * 1 * 120);

    cudaMalloc((void **)&fc_first_layer_dev, sizeof(float) * fc_second_dim * fc_first_dim);
    cudaMalloc((void **)&second_fc, sizeof(float) * 84);
    cudaMalloc((void **)&fc_second_layer_dev, sizeof(float) * fc_second_dim * fc_third_dim);
    cudaMalloc((void **)&third_fc, sizeof(float) * 10);

    cudaMalloc((void **)&img_dev, sizeof(float) * in_w * in_w);
    cudaMalloc((void **)&prediction_dev, sizeof(float) * 10);
    cudaMalloc((void **)&target_dev, sizeof(float) * 10);

    cudaMalloc((void **)&dZ2, sizeof(float) * fc_third_dim);
    cudaMalloc((void **)&dW2, sizeof(float) * fc_third_dim * fc_second_dim);
    cudaMalloc((void **)&dZ1, sizeof(float) * fc_second_dim);
    cudaMalloc((void **)&gdZ1, sizeof(float) * fc_second_dim);
    cudaMalloc((void **)&dW1, sizeof(float) * fc_second_dim * fc_first_dim);
    cudaMalloc((void **)&dZ0, sizeof(float) * fc_first_dim);
    cudaMalloc((void **)&gdZ0, sizeof(float) * fc_first_dim);
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
    cudaMemcpy(kernels_first_layer_dev, kernels_first_layer, sizeof(float) * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer * kernel_num_third_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(fc_first_layer_dev, fc_first_layer, sizeof(float) * fc_first_dim * fc_second_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(fc_second_layer_dev, fc_second_layer, sizeof(float) * fc_second_dim * fc_third_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(img_dev, img, sizeof(float) * in_w * in_h, cudaMemcpyHostToDevice);

    /****
     * Inizio del ciclo per fare apprendimento. L'indice da usare è il numero di epoche
     * per le quali si vuole addestrare la rete.
    */
    for(int rrr = 0; rrr < 5; rrr++){

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
            convolution<<<grid, block>>>(img_dev, first_conv + (i * out_h * out_w), kernels_first_layer_dev + (i * KERNEL_DIM * KERNEL_DIM), out_h, out_w, padding, stride_c);
            tanh<<<grid, block>>>(first_conv + (i * out_h * out_w), out_w, out_h);
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
            tanh<<<grid, block>>>(first_pool + (i * out_h * out_w), out_w, out_h);
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
        clean_vector<<<((out_h * out_w * kernel_num_second_layer) / 1024), 1024>>>(second_conv, out_h * out_w * kernel_num_second_layer);
        for(int i = 0; i < kernel_num_second_layer; i++){
            for(int j = 0; j < kernel_num_first_layer; j++){
                convolution3D<<<grid, block>>>(first_pool + (j * in_h * in_w), second_conv + (i * out_h * out_w), kernels_second_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer)), out_h, out_w, padding, stride_c);
            }
            tanh<<<grid, block>>>(second_conv + (i * out_h * out_w), out_w, out_h);
        }

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
            tanh<<<grid, block>>>(second_pool + (i * out_h * out_w), out_w, out_h);
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
                convolution3D<<<grid, block>>>(second_pool + (j * in_h * in_w), third_conv + (i * out_h * out_w), kernels_third_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer)), out_h, out_w, padding, stride_c);
            }
            tanh<<<grid, block>>>(third_conv + (i * out_h * out_w), out_w, out_h);
        }



        /*
            TODO
            raccogliare tutte le uscite calcolate fino ad ora e raggrupparle in una matrice (120 * m) per
            velocizzare i calcoli successivi.
        
        */




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
        tanh<<<grid, block>>>(second_fc, fc_second_dim, 1);

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
        exponential<<<grid, block>>>(second_fc, fc_third_dim);
        cudaMemcpy(prediction, fc_second_layer_dev, sizeof(float) * fc_third_dim, cudaMemcpyDeviceToHost);
        summation = 0.0;
        for(int i = 0; i < fc_third_dim; i++) summation += prediction[i];
        for(int i = 0; i < fc_third_dim; i++) prediction[i] = prediction[i] / summation;
        for(int i = 0; i < fc_third_dim; i++) printf("%2.2f\n", prediction[i]);

        /****
         * Calcoliamo il valore della Loss.
        */
        loss = 0.0;
        for(int i = 0; i < fc_third_dim; i++) loss += target[i] * log(prediction[i]);
        loss = -loss;
        printf("Loss = %f\n", loss);

        // Inizio della BackPropagation

    // TODO: CONTROLLARE LA CORRETTEZZA DEL CALCOLO CHE VA DAL VALORE DELLA LOSS A DZ2
    float *prediction_dev, *target_dev;
    cudaMalloc((void **)&prediction_dev, sizeof(float) * fc_third_dim);
    cudaMalloc((void **)&target_dev, sizeof(float) * fc_third_dim);
    cudaMemcpy(prediction_dev, prediction, sizeof(float) * fc_third_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(target_dev, target, sizeof(float) * fc_third_dim, cudaMemcpyHostToDevice);

    float *dZ2;
    cudaMalloc((void **)&dZ2, sizeof(float) * fc_third_dim);            // Da moltiplicare per il numero di immagini usate nel batch
    block = {(unsigned int)fc_third_dim};
    grid = {(unsigned int)(block.x / 1024 +1)};
    subtraction<<<grid, block>>>(dZ2, prediction_dev, target_dev, fc_third_dim);
    // FINE CONTROLLO

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
        matrix_product_transpose<<<grid, block>>>(dZ2, second_fc, dW2, out_h, out_w, m);
    
    float *dZ1;
    cudaMalloc((void **)&dZ1, sizeof(float) * fc_second_dim * 1);            //Da moltiplicare per il numero di immagini usate nel batch
    block = {1, (unsigned int)fc_second_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_transpose_product<<<grid, block>>>(fc_second_layer_dev, dZ2, dZ1, 1, fc_second_dim, fc_third_dim);
    float *gdZ1;
    cudaMalloc((void **)&gdZ1, sizeof(float) * fc_second_dim * 1);
    block = {1, (unsigned int)fc_second_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_dot_product<<<grid, block>>>(second_fc, second_fc, gdZ1, 1, fc_second_dim);
    scalar_subtraction<<<grid, block>>>(gdZ1, gdZ1, 1, fc_second_dim);
    matrix_dot_product<<<grid, block>>>(dZ1, gdZ1, dZ1, 1, fc_second_dim);

    float *dW1;
    cudaMalloc((void **)&dW1, sizeof(float) * fc_second_dim * fc_first_dim);
    block = {(unsigned int)fc_first_dim, (unsigned int)fc_second_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_product_transpose<<<grid, block>>>(dZ1, third_conv, dW1, fc_first_dim, fc_second_dim, 1);

    float *dZ0;
    cudaMalloc((void **)&dZ0, sizeof(float) * fc_first_dim * 1);            //Da moltiplicare per il numero di immagini usate nel batch
    block = {1, (unsigned int)fc_first_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_transpose_product<<<grid, block>>>(fc_first_layer_dev, dZ1, dZ0, 1, fc_first_dim, fc_second_dim);
    float *gdZ0;
    cudaMalloc((void **)&gdZ0, sizeof(float) * fc_first_dim * 1);
    block = {1, (unsigned int)fc_first_dim};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    matrix_dot_product<<<grid, block>>>(third_conv, third_conv, gdZ0, 1, fc_first_dim);
    scalar_subtraction<<<grid, block>>>(gdZ0, gdZ0, 1, fc_first_dim);
    matrix_dot_product<<<grid, block>>>(dZ0, gdZ0, dZ0, 1, fc_first_dim);

    float *dF2;
    out_h = 5;
    out_w = 5;
    cudaMalloc((void **)&dF2, sizeof(float) * out_h * out_w * kernel_num_second_layer * kernel_num_third_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_third_layer; i++){
        for(int j = 0; j < kernel_num_second_layer; j++){
            convolution<<<grid, block>>>(second_pool + (j * in_h * in_w), dF2 + (j * in_h * in_w + (i * in_h * in_w * kernel_num_third_layer)), dZ0 + (i * 1 * 1), KERNEL_DIM, KERNEL_DIM, padding, stride_c);
        }
    }

    float *dA3;
    in_h = KERNEL_DIM;
    in_w = KERNEL_DIM;
    h_2 = 1;
    w_2 = 1;
    padding = h_2 - 1;
    /*
         out_h = (in_h + 2 * padding - kernel) / stride + 1
         padding = dimensione del secondo elemento - 1
         kernel_size = dimensione del secondo elemento
         stride = 1
    */
    out_h = (in_h + 2 * padding - h_2) / stride_c + 1;
    out_w = (in_w + 2 * padding - w_2) / stride_c + 1;
    cudaMalloc((void **)&dA3, sizeof(float) * out_h * out_w * kernel_num_second_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_third_layer; i++){
        for(int j = 0; j < kernel_num_second_layer; j++){
            convolution3D<<<grid, block>>>(kernels_third_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer)), dA3 + (j * out_h * out_w), dZ0 + (i * h_2 * w_2), out_h, out_w, padding, stride_c);
        }
    }

    float *dP1;
    cudaMalloc((void **)&dP1, sizeof(float) * 5 * 5 * kernel_num_second_layer);
    block = {1, (unsigned int)fc_second_dim};       //Da sistemare
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_second_layer; i++){
        matrix_dot_product<<<grid, block>>>(second_pool + (i * 5 * 5), second_pool + (i * 5 * 5), dP1, 5, 5);
        scalar_subtraction<<<grid, block>>>(dP1 + (i * 5 * 5), dP1 + (i * 5 * 5), 5, 5);
        matrix_dot_product<<<grid, block>>>(dP1, dA3, dP1, 5, 5);
    }

    float *dA2;
    cudaMalloc((void **)&dA2, sizeof(float) * 10 * 10 * kernel_num_second_layer);
    block = {5, 5};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_second_layer; i++){
        inverse_avg_pooling<<<grid, block>>>(dP1 + (i * 5 * 5), dA2 + (i * 10 * 10), second_conv + (i * 10 * 10), 5, 5, 10, 10, stride_p);
    }

    float *dC1;
    cudaMalloc((void **)&dC1, sizeof(float) * 10 * 10 * kernel_num_second_layer);
    block = {1, (unsigned int)fc_second_dim};       //Da sistemare
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_second_layer; i++){
        matrix_dot_product<<<grid, block>>>(second_conv + (i * 10 * 10), second_conv + (i * 10 * 10), dC1, 10, 10);
        scalar_subtraction<<<grid, block>>>(dC1 + (i * 10 * 10), dC1 + (i * 10 * 10), 10, 10);
        matrix_dot_product<<<grid, block>>>(dC1, dA2, dC1, 10, 10);
    }

    float *dA1;
    in_h = KERNEL_DIM;
    in_w = KERNEL_DIM;
    h_2 = 10;
    w_2 = 10;
    padding = h_2 - 1;
    /*
         out_h = (in_h + 2 * padding - kernel) / stride + 1
         padding = dimensione del secondo elemento - 1
         kernel_size = dimensione del secondo elemento
         stride = 1
    */
    out_h = (in_h + 2 * padding - h_2) / stride_c + 1;
    out_w = (in_w + 2 * padding - w_2) / stride_c + 1;
    cudaMalloc((void **)&dA1, sizeof(float) * out_h * out_w * kernel_num_first_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_second_layer; i++){
        for(int j = 0; j < kernel_num_first_layer; j++){
            convolution3D<<<grid, block>>>(kernels_second_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer)), dA1 + (j * out_h * out_w), dZ0 + (i * h_2 * w_2), out_h, out_w, padding, stride_c);
        }
    }

    float *dF1;
    out_h = 5;
    out_w = 5;
    cudaMalloc((void **)&dF1, sizeof(float) * out_h * out_w * kernel_num_first_layer * kernel_num_second_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_second_layer; i++){
        for(int j = 0; j < kernel_num_first_layer; j++){
            convolution<<<grid, block>>>(first_pool + (j * in_h * in_w), dF1 + (j * in_h * in_w + (i * in_h * in_w * kernel_num_second_layer)), dC1 + (i * 1 * 1), KERNEL_DIM, KERNEL_DIM, padding, stride_c);
        }
    }

    float *dP0;
    cudaMalloc((void **)&dP0, sizeof(float) * 14 * 14 * kernel_num_first_layer);
    block = {1, (unsigned int)fc_second_dim};       //Da sistemare
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_first_layer; i++){
        matrix_dot_product<<<grid, block>>>(first_pool + (i * 14 * 14), first_pool + (i * 14 * 14), dP0, 14, 14);
        scalar_subtraction<<<grid, block>>>(dP0 + (i * 14 * 14), dP0 + (i * 14 * 14), 14, 14);
        matrix_dot_product<<<grid, block>>>(dP0, dA1, dP0, 14, 14);
    }

    float *dA0;
    cudaMalloc((void **)&dA0, sizeof(float) * 28 * 28 * kernel_num_first_layer);
    block = {14, 14};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_second_layer; i++){
        inverse_avg_pooling<<<grid, block>>>(dP0 + (i * 14 * 14), dA0 + (i * 28 * 28), first_conv + (i * 28 * 28), 14, 14, 28, 28, stride_p);
    }
    
    float *dC0;
    cudaMalloc((void **)&dC0, sizeof(float) * 28 * 28 * kernel_num_first_layer);
    block = {1, (unsigned int)fc_second_dim};       //Da sistemare
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_first_layer; i++){
        matrix_dot_product<<<grid, block>>>(first_conv + (i * 28 * 28), first_conv + (i * 28 * 28), dC0, 28, 28);
        scalar_subtraction<<<grid, block>>>(dC0 + (i * 28 * 28), dC0 + (i * 28 * 28), 28, 28);
        matrix_dot_product<<<grid, block>>>(dC0, dA0, dC0, 28, 28);
    }

    float *dF0;
    out_h = 5;
    out_w = 5;
    cudaMalloc((void **)&dF0, sizeof(float) * out_h * out_w * kernel_num_first_layer);
    block = {(unsigned int)out_w, (unsigned int)out_h};
    grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
    for(int i = 0; i < kernel_num_second_layer; i++){
        convolution<<<grid, block>>>(img_dev, dF0 + (i * 5 * 5), dC0 + (i * 1 * 1), KERNEL_DIM, KERNEL_DIM, padding, stride_c);
    }

    /*-------------------------------
        Fine calcolo delle derivate.
        Inizio aggiornamento dei parametri
    */

    block = {1024};
    grid = {(5 * 5 * 6) / 1024 + 1};
    matrix_scalar_product<<<grid, block>>>(dF0, LEARNING_RATE, 5 * 5 * 6);
    subtraction<<<grid, block>>>(kernels_first_layer_dev, kernels_first_layer_dev, dF0, 5 * 5 * 6);

    block = {1024};
    grid = {(5 * 5 * 6 * 16) / 1024 + 1};
    matrix_scalar_product<<<grid, block>>>(dF1, LEARNING_RATE, 5 * 5 * 6 * 16);
    subtraction<<<grid, block>>>(kernels_second_layer_dev, kernels_second_layer_dev, dF1, 5 * 5 * 6 * 16);

    block = {1024};
    grid = {(5 * 5 * 6 * 120) / 1024 + 1};
    matrix_scalar_product<<<grid, block>>>(dF2, LEARNING_RATE, 5 * 5 * 6 * 120);
    subtraction<<<grid, block>>>(kernels_third_layer_dev, kernels_third_layer_dev, dF2, 5 * 5 * 6 * 120);

    block = {1024};
    grid = {(84 * 120) / 1024 + 1};
    matrix_scalar_product<<<grid, block>>>(dW1, LEARNING_RATE, 84 * 120);
    subtraction<<<grid, block>>>(fc_first_layer_dev, fc_first_layer_dev, dW1, 84 * 120);

    block = {1024};
    grid = {(10 * 84) / 1024 + 1};
    matrix_scalar_product<<<grid, block>>>(dW2, LEARNING_RATE, 10 * 84);
    subtraction<<<grid, block>>>(fc_second_layer_dev, fc_second_layer_dev, dW2, 10 * 84);
   
    
    }









/*
    free(kernels_first_layer);
    free(kernels_second_layer);
    free(kernels_third_layer);
    free(fc_first_layer);
    free(fc_second_layer);
    free(prediction);
    free(img);

    cudaFree(kernels_first_layer_dev);
    cudaFree(kernels_second_layer_dev);
    cudaFree(kernels_third_layer_dev);
    cudaFree(fc_first_layer_dev);
    cudaFree(fc_second_layer_dev);

    cudaFree(img_dev);
    cudaFree(first_conv);
    cudaFree(first_pool);
    cudaFree(second_conv);
    cudaFree(second_pool);
    cudaFree(third_conv);
    cudaFree(second_fc);
    cudaFree(third_fc);

    */

    return 0;
}