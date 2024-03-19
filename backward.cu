#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "gpu_functions.h"

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"


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

__host__ void save_parameter(float *param, int w, int h, int c, int n, FILE *fp){
    float *tmp = (float *)malloc(sizeof(float) * w * h * c * n);
    cudaMemcpy(tmp, param, sizeof(float) * w * h * c * n, cudaMemcpyDeviceToHost);

    for(int l = 0; l < n; l++){
        for(int i = 0; i < c; i++){
            for(int j = 0; j < h; j++){
                for(int k = 0; k < w; k++){
                    fprintf(fp, "%e ",tmp[(((k + j * w) + i * (h * w)) + l * (h * w * c))]);
                }
                fprintf(fp, "\n");
            }
        }
    }

    free(tmp);
    return;
}

__host__ void mean_max_min(float *matrix, float *mean, float *max, float *min, int w, int h, int c){
    float tmp;
    float sum = 0;
    for(int i = 0; i < c; i++){
        for(int j = 0; j < h; j++){
            for(int k = 0; k < w; k++){
                tmp = matrix[k + j * w + i * h * w];
                if(tmp < *min) *min = tmp;
                if(tmp > *max) *max = tmp;
                sum += tmp;
            }
        }
    }
    *mean = sum / (float)(w * h * c);
}

__host__ void mean_normalization(float *matrix_dev, int w, int h, int c){
    float mean = 0;
    float max = -FLT_MAX, min = FLT_MAX;
    float *matrix_host = (float *)malloc(sizeof(float) * w * h * c);
    cudaMemcpy(matrix_host, matrix_dev, sizeof(float) * w * h * c, cudaMemcpyDeviceToHost);
    mean_max_min(matrix_host, &mean, &max, &min, w, h, c);
    //printf("\t\t\t\t\t%e, %e, %e\n", mean, max, min);
    dim3 block{(unsigned int)w, (unsigned int)h};
    subtraction_scalar_parametric<<<c, block>>>(matrix_dev, mean, w, h);
    matrix3D_scalar_product<<<c, block>>>(matrix_dev, (2.0 / (max - min)), w, h, 0);
    free(matrix_host);
}

__host__ void save_img(char *name, float *image){
    char file_name[100];
    FILE *fp;
    int x, y;

    if (name[0] == '\0') {
        printf("output file name (*.pgm) : ");
        scanf("%s", file_name);
    } else strcpy(file_name, name);

    if ( (fp=fopen(file_name, "wb"))==NULL ) {
        printf("could not open file\n");
        exit(1);
    }

    int i;
    fputs("P5\n", fp);
    fputs("# Created by Image Processing\n", fp);
    fprintf(fp, "%d %d\n", 32, 32);
    fprintf(fp, "%d\n", 255);
    for (y=0; y<32; y++){
        for (x=0; x<32; x++){
            i = image[y * 32 + x] * 255;
            fputc(i, fp);
        }
    }
    fclose(fp);
    printf("Image was saved successfully\n");
}

__host__ void load_example_to_device(mnist_data data, float *img_dev, float *target){
    float *tmp = (float *)malloc(sizeof(float) * 32 * 32);

    for(int i = 0; i < 32; i++){
        for(int j = 0; j < 32; j++){
            if(i < 2 || i > 29 || j < 2 || j > 29){
                tmp[i * 32 + j] = 0;
            }
            else{
                tmp[i * 32 + j] = (float)(data.data[i - 2][j - 2]);
            }
        }
    }

    // for(int i = 0; i < 32; i++){
    //     for(int j = 0; j < 32; j++){
    //         printf("%f ", tmp[i * 32 + j]);
    //     }
    //     printf("\n");
    // }
    //save_img(name, tmp);

    //printf("label: %d\n", data.label);

    for(int i = 0; i < 10; i++) target[i] = 0;
    target[data.label] = 1;
    cudaMemcpy(img_dev, tmp, sizeof(float) * 32 * 32, cudaMemcpyHostToDevice);

    free(tmp);
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
    mnist_data *data;
    float target[10];
    unsigned int counter = 0;
    int ret;

    if(ret = mnist_load("./MNIST_Dataset/train-images.idx3-ubyte", "./MNIST_Dataset/train-labels.idx1-ubyte", &data, &counter)){
        printf("Errore: %d\n", ret);
        exit(1);
    }
    else{
        printf("Immagini lette: %d\n", counter);
    }

    //float target[10] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    //float *img = (float *) malloc(sizeof(float) * in_h * in_w);
    // TODO: Aggiungere il padding all'immagine di ingresso (28 x 28) -> (32 x 32)
    //for(int i = 0; i < in_w * in_w; i++) img[i] = i;


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
    //cudaMemcpy(img_dev, img, sizeof(float) * in_w * in_h, cudaMemcpyHostToDevice);

    /*----------------------------------------------------------------------------------------------------------------------------
        La successiva riga di codice serve a normalizzare i valori nell'intervallo [0, 1] così che la prima
        attivazione della tanh non debba lavorare con valori altissimo e dia problemi di calcolo.
        Questo lavoro di normalizzazione viene fatto in automatico nei livelli successivi per via della
        presenza della funzione tanh che riporta tutti i valori nell'intervallo [-1, 1].
        Per ora questa prima normalizzazione non sottrae la media ma divide solo per la deiazione standard.    
    */
    //mean_normalization(img_dev, in_w, in_h, 1);

    FILE *loss_file, *time_file, *prediction_file, *parameter_file;
    char parameter_file_name[20];
    if((loss_file = fopen("loss_plot.txt", "w")) == NULL){
        printf("\"loss_plot.txt\" non torvato\n");
        exit(1);
    }
    if((time_file = fopen("execution_time.txt", "w")) == NULL){
        printf("\"loss_plot.txt\" non trovato\n");
        exit(1);
    }
    if((prediction_file = fopen("predictions.txt", "w")) == NULL){
        printf("\"predictions.txt\" non trovato\n");
        exit(1);
    }

    /****
     * Gestione del tempo
    */
    time_t start_time = time(NULL);
    time_t partial_time;

    /****
     * Inizio del ciclo per fare apprendimento. L'indice da usare è il numero di epoche
     * per le quali si vuole addestrare la rete.
    */
    //char name[100];
    for(int epoch = 0; epoch < 1; epoch++){
        for(int batch_dim = 0; batch_dim < 5; batch_dim++){
            //sprintf(name, "epoch_%d.pgm", epoch);
            load_example_to_device(data[batch_dim], img_dev, target);

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
            }
            mean_normalization(first_conv, out_w, out_h, kernel_num_first_layer);
            //debug_print(first_conv, out_w, out_h, 6, 1);

            for(int i = 0; i < kernel_num_first_layer; i++){
                tanh<<<grid, block>>>(first_conv + (i * out_h * out_w), out_w, out_h, 0);
            }

            //debug_print(first_conv, out_w, out_h, 6, 1);

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

            //debug_print(first_pool, out_w, out_h, 6, 1);

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
            }
            mean_normalization(second_conv, out_w, out_h, kernel_num_second_layer);
            for(int i = 0; i < kernel_num_second_layer; i++){
                tanh<<<grid, block>>>(second_conv + (i * out_h * out_w), out_w, out_h, 0);
            }

            //debug_print(second_conv, out_w, out_h, 16, 1);

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

            //debug_print(second_pool, out_w, out_h, 16, 1);

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
            }
            mean_normalization(third_conv, out_w, out_h, kernel_num_third_layer);
            for(int i = 0; i < kernel_num_third_layer; i++){
                tanh<<<grid, block>>>(third_conv + (i * out_h * out_w), out_w, out_h, 0);
            }

            //debug_print(third_conv, out_w, out_h, 120, 1);

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

            //debug_print(second_fc, 1, 84, 1, 1);

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
            //mean_normalization(third_fc, 1, 10, 1);

            //debug_print(third_fc, 1, 10, 1, 1);

            /****
             * Calcoliamo la funzione di Loss calcolando prima gli esponenziali sul dispositivo e poi portiamo i valori
             * sull'host per poter terminare il calcolo della Softmax facendo la sommatoria di tutti i valori ottenuti
             * per poi dividere ogni elemento del vettore per il valore calcolato.
            */
            exponential<<<1, fc_third_dim>>>(third_fc, fc_third_dim);
            cudaMemcpy(prediction, third_fc, sizeof(float) * fc_third_dim, cudaMemcpyDeviceToHost);
            //debug_print(third_fc, 1, 10, 1, 1);
            summation = 0.0;
            for(int i = 0; i < fc_third_dim; i++) summation += prediction[i];
            for(int i = 0; i < fc_third_dim; i++) {
                prediction[i] = prediction[i] / summation;
                //if(epoch % 20 == 0){printf("\t\t%f\n", prediction[i]);}
                //printf("\t\t%f\n", prediction[i]);
            }
            //for(int i = 0; i < fc_third_dim; i++) printf("%2.2f\n", prediction[i]);

            /****
             * Calcoliamo il valore della Loss.
            */
            loss = 0.0;
            //loss += target[i] * logf(prediction[i];
            for(int i = 0; i < fc_third_dim; i++){
                if(target)
                    loss += target[i] * logf(prediction[i]);
            }
            loss = -loss;
            // if(epoch % 20 == 0){
            //     printf("Loss = %e\n", loss);
            // }
            //printf("Loss = %e\n", loss);
            //fprintf(file_p, "%d %e\n", epoch, loss);


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

            //debug_print(dW2, 84, 10, 1, 1);
        
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

            //debug_print(dZ1, 1, 84, 1, 1);

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

            //debug_print(dW1, 120, 84, 1, 1);

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

            // debug_print(dZ0, 1, 120, 1, 1);

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

            // debug_print(dF2 + (5 * 5 * 16 * 119), 5, 5, 16, 1);

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

            // debug_print(dA3, 5, 5, 16, 1);

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
                matrix_dot_product<<<grid, block>>>(second_pool + (i * in_h * in_w), second_pool + (i * in_h * in_w), dP1 + (i * out_h * out_w), out_h, out_w);
                scalar_subtraction<<<grid, block>>>(dP1 + (i * in_w * in_h), dP1 + (i * in_w * in_h), out_w, out_h);
                matrix_dot_product<<<grid, block>>>(dP1 + (i * in_h * in_w), dA3 + (i * in_h * in_w), dP1 + (i * in_h * in_w), out_w, out_h);
            }

            // debug_print(dP1, 5, 5, 16, 1);

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

            // debug_print(dA2, 10, 10, 16, 1);

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
                matrix_dot_product<<<grid, block>>>(second_conv + (i * in_w * in_h), second_conv + (i * in_w * in_h), dC1 + (i * out_h * out_w), out_w, out_h);
                scalar_subtraction<<<grid, block>>>(dC1 + (i * in_h * in_w), dC1 + (i * in_h * in_w), out_w, out_h);
                matrix_dot_product<<<grid, block>>>(dC1 + (i * in_h * in_w), dA2 + (i * in_h * in_w), dC1 + (i * in_h * in_w), out_w, out_h);
            }

            // debug_print(dC1, 10, 10, 16, 1);

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

            // debug_print(dA1, 14, 14, 6, 1);

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

            // debug_print(dF1, 5, 5, 6, 1);

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
                matrix_dot_product<<<grid, block>>>(first_pool + (i * in_h * in_w), first_pool + (i * in_h * in_w), dP0 + (i * out_h * out_w), out_w, out_h);
                scalar_subtraction<<<grid, block>>>(dP0 + (i * in_h * in_w), dP0 + (i * in_h * in_w), out_w, out_h);
                matrix_dot_product<<<grid, block>>>(dP0 + (i * in_h * in_w), dA1 + (i * in_h * in_w), dP0 + (i * in_h * in_w), out_w, out_h);
            }

            // debug_print(dA1, 14, 14, 6, 1);
            // debug_print(first_pool, 14, 14, 6, 1);
            // debug_print(dP0, 14, 14, 6, 1);

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
                matrix_dot_product<<<grid, block>>>(first_conv + (i * in_w * in_h), first_conv + (i * in_w * in_h), dC0 + (i * out_h * out_w), out_w, out_h);
                scalar_subtraction<<<grid, block>>>(dC0 + (i * in_w * in_h), dC0 + (i * in_w * in_h), out_w, out_h);
                matrix_dot_product<<<grid, block>>>(dC0 + (i * in_w * in_h), dA0 + (i * in_w * in_h), dC0 + (i * in_w * in_h), out_w, out_h);
            }

            
            //debug_print(dC0, 28, 28, 6, 1);

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
            /*for(int i = 0; i < kernel_num_first_layer; i++){
                convolution3D<<<grid, block>>>(img_dev, dF0 + (i * out_w * out_h/* * kernel_num_first_layer), dC0 + (i * w_2 * h_2), out_w, out_h, padding, stride_c, h_2);
            }*/

            //debug_print(img_dev, 32, 32, 1, 1);

            for(int i = 0; i < kernel_num_first_layer; i++){
                convolution3D<<<grid, block>>>(img_dev, dF0 + (i * out_h * out_w), dC0 + (i * h_2 * w_2), out_w, out_h, padding, stride_c, h_2);
            }

            debug_print(dF0, 5, 5, 1, 6);

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

            /****
             * Gestione del salvataggio di:
             * loss
             * parametri
             * predizioni
             * tempo
            */

            if(batch_dim % 5 == 0){
                fprintf(loss_file, "%d\t%e\n", (batch_dim + epoch * 60000), loss);       

                fflush(loss_file);             
            }

            if(batch_dim % 1000 == 0){
                fprintf(prediction_file, "Epoch: %d\tIteration: %d\n", epoch, batch_dim);
                for(int i = 0; i < 10; i++)fprintf(prediction_file, "%.3f\t", prediction[i]);
                fprintf(prediction_file, "\n");
                fflush(prediction_file);

                partial_time = time(NULL);
                fprintf(time_file, "Epoch: %d\tIteration: %d\t\t%02d:%02d\n", epoch, batch_dim, (int)(difftime(partial_time, start_time)) / 60, (int)(difftime(partial_time, start_time)) % 60);
                fflush(time_file);
            }
        }

        sprintf(parameter_file_name, "epoch_%d.txt", epoch);
        if((parameter_file = fopen(parameter_file_name, "w")) == NULL){
            printf("\"%s\" non trovato\n", parameter_file_name);
            exit(1);
        }
        fprintf(parameter_file, "#kernel 1\n");
        fprintf(parameter_file, "5 5 1 6\n");
        save_parameter(kernels_first_layer_dev, 5, 5, 1, 6, parameter_file);
        fprintf(parameter_file, "#kernel 2\n");
        fprintf(parameter_file, "5 5 6 16\n");
        save_parameter(kernels_second_layer_dev, 5, 5, 6, 6, parameter_file);
        fprintf(parameter_file, "#kernel 3\n");
        fprintf(parameter_file, "5 5 16 120\n");
        save_parameter(kernels_third_layer_dev, 5, 5, 16, 120, parameter_file);
        fprintf(parameter_file, "#weights 1\n");
        fprintf(parameter_file, "120 84 1 1\n");
        save_parameter(fc_first_layer_dev, 120, 84, 1, 1, parameter_file);
        fprintf(parameter_file, "#weights 2\n");
        fprintf(parameter_file, "10 84 1 1\n");
        save_parameter(fc_second_layer_dev, 84, 10, 1, 1, parameter_file);

        fclose(parameter_file);
        
    }

    fclose(loss_file);
    fclose(time_file);
    fclose(prediction_file);

    free(kernels_first_layer);
    free(kernels_second_layer);
    free(kernels_third_layer);
    free(fc_first_layer);
    free(fc_second_layer);
    free(prediction);
    
    free(data);

    

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