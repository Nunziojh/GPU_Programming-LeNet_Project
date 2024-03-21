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

__host__ void load_parameter(float *param, FILE *fp){
    int n, w, h, c;
    char *tr;
    fscanf(fp, "#%s\n", tr);
    fscanf(fp, "%d %d %d %d\n", &w, &h, &c, &n);

    double dato;

    float *tmp = (float *)malloc(sizeof(float) * w * h * c * n);
    for(int l = 0; l < n; l++){
        for(int i = 0; i < c; i++){
            for(int j = 0; j < h; j++){
                for(int k = 0; k < w; k++){
                    fscanf(fp, "%lf ", &dato);
                    //fscanf(fp, "%e ", &tmp[(((k + j * w) + i * (h * w)) + l * (h * w * c))]);
                    printf("%e\n", dato);
                }
                fscanf(fp, "\n");
            }
        }
    }

    cudaMemcpy(param, tmp, sizeof(float) * w * h * c * n, cudaMemcpyDeviceToHost);

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

    FILE *param_file;
    if((param_file = fopen("epoch_3.txt", "r")) == NULL){
        printf("File non trovato\n");
        exit(1);
    }

    load_parameter(kernels_first_layer, param_file);
    debug_print(kernels_first_layer, 5, 5, 1, 6);

    return 0;
    load_parameter(kernels_second_layer, param_file);
    load_parameter(kernels_third_layer, param_file);
    load_parameter(fc_first_layer, param_file);
    load_parameter(fc_second_layer, param_file);


    /***
     * Definizione e allocazione delle matrici e dei target in ingresso.
    */
    mnist_data *data;
    float target[10];
    unsigned int counter = 0;
    int ret;

    if(ret = mnist_load("./MNIST_Dataset/t10k-images.idx3-ubyte", "./MNIST_Dataset/t10k-labels.idx1-ubyte", &data, &counter)){
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
        printf("\"execution_time.txt\" non trovato\n");
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

    int prediction_index = 0;
    float max = 0;
    int prediction_counter = 0;
    for(int batch_dim = 0; batch_dim < 10; batch_dim++){
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
            convolution<<<grid, block>>>(img_dev, first_conv + (i * out_h * out_w), kernels_first_layer_dev + (i * KERNEL_DIM * KERNEL_DIM), in_h, out_h, KERNEL_DIM, padding, stride_c);
        }
        //debug_print(first_conv, out_w, out_h, 6, 1);
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
                convolution3D<<<grid, block>>>(first_pool + (j * in_h * in_w), second_conv + (i * out_h * out_w), kernels_second_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_first_layer)), in_h, out_h,KERNEL_DIM, padding, stride_c);
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
                convolution3D<<<grid, block>>>(second_pool + (j * in_h * in_w), third_conv + (i * out_h * out_w), kernels_third_layer_dev + (j * KERNEL_DIM * KERNEL_DIM + (i * KERNEL_DIM * KERNEL_DIM * kernel_num_second_layer)), in_h, out_h, KERNEL_DIM, padding, stride_c);
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
        max = 0;
        for(int i = 0; i < fc_third_dim; i++) summation += prediction[i];
        for(int i = 0; i < fc_third_dim; i++) {
            prediction[i] = prediction[i] / summation;
            if(prediction[i] > max){
                prediction_index = i;
                max = prediction[i];
            }
            //if(epoch % 20 == 0){printf("\t\t%f\n", prediction[i]);}
            //printf("\t\t%f\n", prediction[i]);
        }

        if(target[prediction_index]) prediction_counter++;
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


        /****
         * Gestione del salvataggio di:
         * loss
         * parametri
         * predizioni
         * tempo
        */

        fprintf(loss_file, "%d\t%e\n", batch_dim, loss);       

        fflush(loss_file);

        fprintf(prediction_file, "Iteration: %d\n", batch_dim);
        for(int i = 0; i < 10; i++)fprintf(prediction_file, "%.3f\t", prediction[i]);
        fprintf(prediction_file, "\n");
        fflush(prediction_file);

        partial_time = time(NULL);
        fprintf(time_file, "Iteration: %d\t\t%02d:%02d\n", batch_dim, (int)(difftime(partial_time, start_time)) / 60, (int)(difftime(partial_time, start_time)) % 60);
        fflush(time_file);
        start_time = partial_time;
    }

    fprintf(loss_file, "Valori predetti correttamente: %d\n", prediction_counter);

    printf("Valori predetti correttamente: %d\n", prediction_counter);


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

    cudaFree(prediction_dev);
    cudaFree(target_dev);

    return 0;
}