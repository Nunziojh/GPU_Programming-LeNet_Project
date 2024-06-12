#include "leNet.h"

int main(int argc, char **argv){

    /***
     * Definizione dei parametri della rete.
    */
    const unsigned int padding = 0;
    const unsigned int stride_c = 1;
    const unsigned int stride_p = 2;
    const unsigned int window_size = 2;
    const unsigned int kernel_num_first_layer = 6;
    const unsigned int kernel_num_second_layer = 16;
    const unsigned int kernel_num_third_layer = 120;
    const unsigned int kernel_dim = 5;
    const unsigned int fc_first_dim = 120;
    const unsigned int fc_second_dim = 84;
    const unsigned int fc_third_dim = 10;
    const unsigned int m = 1;        //batch size
    const unsigned int tile_dim = 32;
    
    unsigned int batch_dim;
    unsigned int epoch_dim;

    /*
        Definizione delle variabili generiche.
    */
    dim3 block, grid;
    int unsigned shared_mem_dim;
    int unsigned max_third_dim;
    int unsigned in_h = 32;
    int unsigned in_w = 32;
    int unsigned out_h = (in_h + 2 * padding - kernel_dim) / stride_c + 1;
    int unsigned out_w = (in_w + 2 * padding - kernel_dim) / stride_c + 1;
    int unsigned h_2, w_2;
    int unsigned padding_full_conv;
    float summation;
    float loss;

    /****
     * Definizione e allocazione delle matrici dei pesi.
    */
    float *kernels_first_layer = (float *) malloc(sizeof(float) * kernel_dim * kernel_dim * kernel_num_first_layer);        //5x5x1x6
    float *kernels_second_layer = (float *) malloc(sizeof(float) * kernel_dim * kernel_dim * kernel_num_first_layer * kernel_num_second_layer);     //5x5x6x16
    float *kernels_third_layer = (float *) malloc(sizeof(float) * kernel_dim * kernel_dim * kernel_num_second_layer * kernel_num_third_layer);      //5x5x16x120
    float *fc_first_layer = (float *) malloc(sizeof(float) * fc_second_dim * fc_first_dim);         //84 x 120
    float *fc_second_layer = (float *) malloc(sizeof(float) * fc_third_dim * fc_second_dim);        //10 x 84
    float *prediction = (float *) malloc(sizeof(float) * fc_third_dim);     //10

#ifndef PARAMETER_FROM_FILE

    time_t seed = time(NULL);
    srand(seed);
    FILE *fp;
    if((fp = fopen("random_seed.txt", "w")) == NULL){
        fprintf(stderr, "random_seed.txt non trovato\n");
        exit(1);
    }
    fprintf(fp, "%lld", seed);
    fclose(fp);

    for(int i = 0; i < kernel_dim * kernel_dim * kernel_num_first_layer; i++) kernels_first_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < kernel_dim * kernel_dim * kernel_num_first_layer * kernel_num_second_layer; i++) kernels_second_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < kernel_dim * kernel_dim * kernel_num_second_layer * kernel_num_third_layer; i++) kernels_third_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < fc_first_dim * fc_second_dim; i++) fc_first_layer[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < fc_second_dim * fc_third_dim; i++) fc_second_layer[i] = (float)rand() / (float)RAND_MAX;

#else

    char filename[100];
    printf("Inserire il nome dei file contenente i parametri della rete: ");
    scanf("%s", filename);
    FILE *fp;
    // if((fp = fopen(PARAMETER_FILE, "r")) == NULL){
    if((fp = fopen(filename, "r")) == NULL){
        fprintf(stderr, "File dei parametri: %s non trovato!\nCompila con la direttiva -D PARAMETER_FILE=\"nomefile.txt\"\n", filename);
        exit(1);
    }
    load_parameter(kernels_first_layer, fp);
    load_parameter(kernels_second_layer, fp);
    load_parameter(kernels_third_layer, fp);
    load_parameter(fc_first_layer, fp);
    load_parameter(fc_second_layer, fp);

    fclose(fp);

#endif

    /***
     * Definizione e allocazione delle matrici e dei target in ingresso.
    */
#if defined(TEST) || defined(TRAIN)
    mnist_data *data;
    float target[10];
    unsigned int counter = 0;
    int ret;
#endif

#ifdef TEST

    if(ret = mnist_load(".\\..\\MNIST_Dataset\\t10k-images.idx3-ubyte", ".\\..\\MNIST_Dataset\\t10k-labels.idx1-ubyte", &data, &counter)){
        printf("Errore: %d\n", ret);
        exit(1);
    }
    batch_dim = counter;
    epoch_dim = 1;
    float max;
    int prediction_index = 0;
    int prediction_counter = 0;
    printf("Immagini lette: %d\nDimensione di una epoca: %d\nNumero di epoche: %d\n", counter, batch_dim, epoch_dim);

#elif TRAIN

    if(ret = mnist_load(".\\..\\MNIST_Dataset\\train-images.idx3-ubyte", ".\\..\\MNIST_Dataset\\train-labels.idx1-ubyte", &data, &counter)){
        printf("Errore: %d\n", ret);
        exit(1);
    }
    batch_dim = counter;
    printf("Specificare il nuemro di epoche su cui addestrare la rete: ");
    scanf("%d", &epoch_dim);
    printf("Immagini lette: %d\nDimensione di una epoca: %d\nNumero di epoche: %d\n", counter, batch_dim, epoch_dim);

#elif USAGE

    batch_dim = 1;
    epoch_dim = 1;
    float max;
    int prediction_index = 0;
    int prediction_counter = 0;

#endif

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

#ifdef TRAIN

    float *dZ2, *dW2;
    float *dZ1, *gdZ1, *dW1;
    float *dZ0, *gdZ0;
    float *dF2, *dA3;
    float *dP1, *dA2;
    float *dC1, *dA1;
    float *dF1, *dP0, *dA0;
    float *dC0, *dF0;

#endif
    
    cudaMalloc((void **)&kernels_first_layer_dev, sizeof(float) * kernel_dim * kernel_dim * kernel_num_first_layer);
    cudaMalloc((void **)&first_conv, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void **)&first_pool, sizeof(float) * 14 * 14 *6);

    cudaMalloc((void **)&kernels_second_layer_dev, sizeof(float) * kernel_dim * kernel_dim * kernel_num_first_layer * kernel_num_second_layer);
    cudaMalloc((void **)&second_conv, sizeof(float) * 10 * 10 * 16);
    cudaMalloc((void **)&second_pool, sizeof(float) * 5 * 5 * 16);
    
    cudaMalloc((void **)&kernels_third_layer_dev, sizeof(float) * kernel_dim * kernel_dim * kernel_num_second_layer * kernel_num_third_layer);
    cudaMalloc((void **)&third_conv, sizeof(float) * 1 * 1 * 120 * m);

    cudaMalloc((void **)&fc_first_layer_dev, sizeof(float) * fc_second_dim * fc_first_dim);
    cudaMalloc((void **)&second_fc, sizeof(float) * 84 * m);
    cudaMalloc((void **)&fc_second_layer_dev, sizeof(float) * fc_second_dim * fc_third_dim);
    cudaMalloc((void **)&third_fc, sizeof(float) * 10 * m);

    cudaMalloc((void **)&img_dev, sizeof(float) * in_w * in_w);
    
    cudaMalloc((void **)&prediction_dev, sizeof(float) * fc_third_dim);
    cudaMalloc((void **)&target_dev, sizeof(float) * fc_third_dim);

#ifdef TRAIN

    cudaMalloc((void **)&dZ2, sizeof(float) * fc_third_dim * m);
    cudaMalloc((void **)&dW2, sizeof(float) * fc_third_dim * fc_second_dim);
    cudaMalloc((void **)&dZ1, sizeof(float) * fc_second_dim * m);
    cudaMalloc((void **)&gdZ1, sizeof(float) * fc_second_dim * m);
    cudaMalloc((void **)&dW1, sizeof(float) * fc_second_dim * fc_first_dim);
    cudaMalloc((void **)&dZ0, sizeof(float) * fc_first_dim * m);
    cudaMalloc((void **)&gdZ0, sizeof(float) * fc_first_dim * m);
    cudaMalloc((void **)&dF2, sizeof(float) * kernel_dim * kernel_dim * kernel_num_second_layer * kernel_num_third_layer);
    cudaMalloc((void **)&dA3, sizeof(float) * 5 * 5 * kernel_num_second_layer);
    cudaMalloc((void **)&dP1, sizeof(float) * 5 * 5 * kernel_num_second_layer);
    cudaMalloc((void **)&dA2, sizeof(float) * 10 * 10 * kernel_num_second_layer);
    cudaMalloc((void **)&dC1, sizeof(float) * 10 * 10 * kernel_num_second_layer);
    cudaMalloc((void **)&dA1, sizeof(float) * 14 * 14 * kernel_num_first_layer);
    cudaMalloc((void **)&dF1, sizeof(float) * kernel_dim * kernel_dim * kernel_num_first_layer * kernel_num_second_layer);
    cudaMalloc((void **)&dP0, sizeof(float) * 14 * 14 * kernel_num_first_layer);
    cudaMalloc((void **)&dA0, sizeof(float) * 28 * 28 * kernel_num_first_layer);
    cudaMalloc((void **)&dC0, sizeof(float) * 28 * 28 * kernel_num_first_layer);
    cudaMalloc((void **)&dF0, sizeof(float) * kernel_dim * kernel_dim * kernel_num_first_layer);

#endif

    /****
     * Copia dei valori dei parametri e dell'immagine di ingresso sul dispositivo.
    */
    cudaMemcpy(kernels_first_layer_dev, kernels_first_layer, sizeof(float) * kernel_dim * kernel_dim * kernel_num_first_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(kernels_second_layer_dev, kernels_second_layer, sizeof(float) * kernel_dim * kernel_dim * kernel_num_first_layer * kernel_num_second_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(kernels_third_layer_dev, kernels_third_layer, sizeof(float) * kernel_dim * kernel_dim * kernel_num_second_layer * kernel_num_third_layer, cudaMemcpyHostToDevice);
    cudaMemcpy(fc_first_layer_dev, fc_first_layer, sizeof(float) * fc_first_dim * fc_second_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(fc_second_layer_dev, fc_second_layer, sizeof(float) * fc_second_dim * fc_third_dim, cudaMemcpyHostToDevice);

#ifdef CHECK_PARAMETER_CORRECTNESS

    debug_print(kernels_first_layer_dev, 5, 5, 1, 6);
    debug_print(kernels_second_layer_dev, 5, 5, 6, 16);
    debug_print(kernels_third_layer_dev, 5, 5, 16, 120);
    debug_print(fc_first_layer_dev, 120, 84, 1, 1);
    debug_print(fc_second_layer_dev, 84, 10, 1, 1);

    return 0;

#endif

#ifdef TIME_TEST

    FILE *time_file_test;
    if((time_file_test = fopen("test_time.txt", "w")) == NULL){
        printf("test_time.txt non trovato\n");
        exit(1);
    }

#ifdef __linux__
    struct timeval start, stop;
#else
    struct timespec start, stop;
#endif
    long int u_sec;

#else

    FILE *time_file, *loss_file, *prediction_file, *parameter_file;
    char parameter_file_name[20];
    if((loss_file = fopen("loss_plot.txt", "w")) == NULL){
        printf("loss_plot.txt non torvato\n");
        exit(1);
    }
    if((time_file = fopen("execution_time.txt", "w")) == NULL){
        printf("execution_time.txt non trovato\n");
        exit(1);
    }
    if((prediction_file = fopen("predictions.txt", "w")) == NULL){
        printf("predictions.txt non trovato\n");
        exit(1);
    }

    time_t start_time = time(NULL), partial_time;

#endif

    /****
     * Inizio del ciclo per fare apprendimento. L'indice da usare è il numero di epoche
     * per le quali si vuole addestrare la rete.
    */

    // epoch_dim = 1;
    // batch_dim = 1;
    // float *buffer = (float *)malloc(sizeof(float) * 50000);
    // FILE *parametri_in_ingresso;
    for(int epoch = 0; epoch < epoch_dim; epoch++){
        for(int batch = 0; batch < batch_dim; batch++){

#ifdef TIME_TEST
            
            start_timer(&start);

#endif
#if defined(TEST) || defined(TRAIN)

            load_example_to_device(data[batch], img_dev, target);

#else

            FILE *fp;
            if((fp = fopen("input_img.txt", "r")) == NULL){
                fprintf(stderr, "Immagine (input_ing.txt) non trovata\n");
                exit(1);
            }
            float *img = (float *)malloc(sizeof(float) * 32 * 32);
            for(int i = 0; i < 32 * 32; i++){
                fscanf(fp, "%f", &img[i]);
            }

            fclose(fp);
            cudaMemcpy(img_dev, img, sizeof(float) * 32 * 32, cudaMemcpyHostToDevice);
            free(img);

#endif

            /****
             * Calcolo del primo layer convolutivo con la relativa funzione di attivazion tanh
             * Partiamo dall'immagine in ingresso salvata nella variabile 'img_dev' con dimensioni (32 x 32)
             * e ne facciamo la convoluzione valid, (0 di padding e 1 di stride) con 6 kernel di dimensioni (5 x 5)
             * salvati, nella variabile 'kernels_first_layer_dev', come vettore.
            */
            in_h = 32;
            in_w = 32;
            out_h = (in_h + 2 * padding - kernel_dim) / stride_c + 1;
            out_w = (in_w + 2 * padding - kernel_dim) / stride_c + 1;

            max_third_dim = (1024.0 * 48.0 / sizeof(float) - (in_w * in_h * 1)) / (kernel_dim * kernel_dim * 1);
            shared_mem_dim = (in_w * in_h * 1 + kernel_dim * kernel_dim * 1 * max_third_dim) * sizeof(float);
            block = {(unsigned int)min(32, in_w), (unsigned int)min(32, in_h), (unsigned int)min(max_third_dim, (1024 / (min(32, in_w) * min(32, in_h))))};
            grid = {(unsigned int)ceil(((float)out_w / block.x)), (unsigned int)ceil(((float)out_h / block.y)), (unsigned int)ceil((float)kernel_num_first_layer / block.z)};
            convolution_3D_shared<<<grid, block, shared_mem_dim>>>(img_dev, kernels_first_layer_dev, first_conv, in_w, in_h, 1, kernel_dim, kernel_dim, 1, kernel_num_first_layer, out_w, out_h, kernel_num_first_layer);
            
            mean_normalization(first_conv, out_w, out_h, kernel_num_first_layer);
            
            max_third_dim = (1024 / (min(32, out_w) * min(32, out_h)));
            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_first_layer, max_third_dim)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_first_layer / block.z)};
            tanh<<<grid, block>>>(first_conv, out_w, out_h, kernel_num_first_layer);

            // debug_print(first_conv, 28, 28, 6, 1, "first_conv_new.txt");

            /****
             * Calcoliamo il primo layer di Average Pooling con la relativa funzione di attivazione tanh.
             * Partiamo dal risultato del livello precedente salvato nella variabile 'first_conv', come vettore di dimensioni
             * (28 x 28 x 6), e applichiamo il filtro e otteniamo un risultato, salvato nella variabile 'first_pool', di dimensione
             * (14 x 14 x 6), sempre memorizzandolo come un vettore.
            */


            // if((parametri_in_ingresso = fopen("base_first_conv.txt", "r")) == NULL){printf("errore\n");exit(1);}
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(first_conv, buffer, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(first_conv, 28, 28, 6, 1, "first_conv_copied.txt");


            in_h = out_h;
            in_w = out_w; 
            out_h = (in_h - window_size) / stride_p + 1;
            out_w = (in_w - window_size) / stride_p + 1;

            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_first_layer, (1024 / (min(32, out_w) * min(32, out_h))))};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_first_layer / block.z),};
            avg_pooling_monolithic<<<grid, block>>>(first_conv, first_pool, in_w, in_h, out_w, out_h, kernel_num_first_layer, stride_p, window_size);

            max_third_dim = (1024 / (min(32, out_w) * min(32, out_h)));
            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_first_layer, max_third_dim)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_first_layer / block.z)};
            tanh<<<grid, block>>>(first_pool, out_w, out_h, kernel_num_first_layer);

            // debug_print(first_pool, 14, 14, 6, 1, "first_pool_new.txt");

            /****
             * Calcoliamo il secondo layer convolutivo a partire dall'uscita del layer di pooling precedente,
             * le dimensioni di ingresso sono (14 x 14 x 6), usiamo 16 kernel di dimensioni (5 x 5 x 6) e otteniamo
             * un valore di uscita, salvato nella variabile second_conv di dimensione (10 x 10 x 16).
            */


            // parametri_in_ingresso = fopen("base_first_pool.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(first_pool, buffer, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(first_pool, 14, 14, 6, 1, "first_pool_copied.txt");


            in_h = out_h;
            in_w = out_w;
            out_h = (in_h + 2 * padding - kernel_dim) / stride_c + 1;
            out_w = (in_w + 2 * padding - kernel_dim) / stride_c + 1;

            max_third_dim = (1024.0 * 48.0 / sizeof(float) - (in_w * in_h * kernel_num_first_layer)) / (kernel_dim * kernel_dim * kernel_num_first_layer);
            shared_mem_dim = (in_w * in_h * kernel_num_first_layer + kernel_dim * kernel_dim * kernel_num_first_layer * max_third_dim) * sizeof(float);
            block = {(unsigned int)min(32, in_w), (unsigned int)min(32, in_h), (unsigned int)min(max_third_dim, (1024 / (min(32, in_w) * min(32, in_h))))};
            grid = {(unsigned int)ceil(((float)out_w / block.x)), (unsigned int)ceil(((float)out_h / block.y)), (unsigned int)ceil((float)kernel_num_second_layer / block.z)};
            convolution_3D_shared<<<grid, block, shared_mem_dim>>>(first_pool, kernels_second_layer_dev, second_conv, in_w, in_h, kernel_num_first_layer, kernel_dim, kernel_dim, kernel_num_first_layer, kernel_num_second_layer, out_w, out_h, kernel_num_second_layer);

            mean_normalization(second_conv, out_w, out_h, kernel_num_second_layer);
            
            max_third_dim = (1024 / (min(32, out_w) * min(32, out_h)));
            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_second_layer, max_third_dim)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_second_layer / block.z)};
            tanh<<<grid, block>>>(second_conv, out_w, out_h, kernel_num_second_layer);

            // debug_print(second_conv, 10, 10, 16, 1, "second_conv_new.txt");

            /****
             * Calcoliamo il secondo layer di Average Pooling partendo da una matrice di dimensini (10 x 10 x 16)
             * e otteniamo una matrice di dimensioni (5 x 5 x 16) che salviamo come vettore in second_pool.
            */


            // parametri_in_ingresso = fopen("base_second_conv.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(second_conv, buffer, sizeof(float) * 10 * 10 * 16, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(second_conv, 10, 10, 16, 1, "second_conv_copied.txt");


            in_h = out_h;
            in_w = out_w;
            out_h = (in_h - window_size) / stride_p + 1;
            out_w = (in_w - window_size) / stride_p + 1;

            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_second_layer, (1024 / (min(32, out_w) * min(32, out_h))))};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_second_layer / block.z),};
            avg_pooling_monolithic<<<grid, block>>>(second_conv, second_pool, in_w, in_h, out_w, out_h, kernel_num_second_layer, stride_p, window_size);

            max_third_dim = (1024 / (min(32, out_w) * min(32, out_h)));
            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_second_layer, max_third_dim)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_second_layer / block.z)};
            tanh<<<grid, block>>>(second_pool, out_w, out_h, kernel_num_second_layer);

            // debug_print(second_pool, 5, 5, 16, 1, "second_pool_new.txt");

            /****
             * Calcoliamo il terzo layer convolutivo a partire dall'uscita del layer di pooling precedente,
             * le dimensioni di ingresso sono (5 x 5 x 16), usiamo 120 kernel di dimensioni (5 x 5) e otteniamo
             * un valore di uscita, salvato nella variabile third_conv di dimensione (1 x 1 x 120).
            */


            // parametri_in_ingresso = fopen("base_second_pool.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(second_pool, buffer, sizeof(float) * 5 * 5 * 16, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(second_pool, 5, 5, 16, 1, "second_pool_copied.txt");


            in_h = out_h;
            in_w = out_w;
            out_h = (in_h + 2 * padding - kernel_dim) / stride_c + 1;
            out_w = (in_w + 2 * padding - kernel_dim) / stride_c + 1;

            max_third_dim = (1024.0 * 48.0 / sizeof(float) - (in_w * in_h * kernel_num_second_layer)) / (kernel_dim * kernel_dim * kernel_num_second_layer);
            shared_mem_dim = (in_w * in_h * kernel_num_second_layer + kernel_dim * kernel_dim * kernel_num_second_layer * max_third_dim) * sizeof(float);
            block = {(unsigned int)min(32, in_w), (unsigned int)min(32, in_h), (unsigned int)min(max_third_dim, (1024 / (min(32, in_w) * min(32, in_h))))};
            grid = {(unsigned int)ceil(((float)out_w / block.x)), (unsigned int)ceil(((float)out_h / block.y)), (unsigned int)ceil((float)kernel_num_third_layer / block.z)};
            convolution_3D_shared<<<grid, block, shared_mem_dim>>>(second_pool, kernels_third_layer_dev, third_conv, in_w, in_h, kernel_num_second_layer, kernel_dim, kernel_dim, kernel_num_second_layer, kernel_num_third_layer, out_w, out_h, kernel_num_third_layer);

            mean_normalization(third_conv, out_w, out_h, kernel_num_third_layer);
            
            max_third_dim = (1024 / (min(32, out_w) * min(32, out_h)));
            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_third_layer, max_third_dim)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_third_layer / block.z)};
            tanh<<<grid, block>>>(third_conv, out_w, out_h, kernel_num_third_layer);

            // debug_print(third_conv, 1, 1, 120, 1, "third_conv_new.txt");

            /****
             * A partire dalla matrice di dimensini (120 x m) ottenuta dall'ultimo layer convolutivo, calcoliamo il primo
             * livello di Fully Connected usando come matrice di pesi la variabile 'fc_first_layer_dev' di dimensioni
             * (84 x 120). Otteniamo una matrice risultato di dimensioni (84 x m).
            */


            // parametri_in_ingresso = fopen("base_third_conv.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(third_conv, buffer, sizeof(float) * 120, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(third_conv, 1, 1, 120, 1, "third_conv_copied.txt");
            // debug_print(fc_first_layer_dev, 120, 84, 1, 1, "fc_first_layer_dev_new.txt");


            in_h = fc_first_dim;
            in_w = m;
            out_h = fc_second_dim;
            out_w = m;

            block = {(unsigned int)tile_dim, (unsigned int)tile_dim};
            grid = {(unsigned int)ceil((float)in_h / block.x), (unsigned int)ceil((float)out_h / block.y)};
            shared_mem_dim = tile_dim * tile_dim * 2 * sizeof(float);
            matrix_product_shared<<<grid, block, shared_mem_dim>>>(fc_first_layer_dev, third_conv, second_fc, out_w, out_h, fc_first_dim, tile_dim);
            
            // debug_print(second_fc, 1, 84, 1, 1, "second_fc_new_beforet.txt");

            max_third_dim = (1024 / (min(32, out_w) * min(32, out_h)));
            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(1, max_third_dim)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)1 / block.z)};
            tanh<<<grid, block>>>(second_fc, 1, fc_second_dim, 1);

            // debug_print(second_fc, 1, 84, 1, 1, "second_fc_new.txt");

            /****
             * A partire dalla matrice di dimensini (84 x m) ottenuta al livello FC precedente, calcoliamo il secondo
             * livello di Fully Connected usando come matrice di pesi la variabile 'fc_second_layer_dev' di dimensioni
             * (10 x 84). Otteniamo una matrice risultato di dimensioni (10 x m).
            */


            // parametri_in_ingresso = fopen("base_second_fc.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(second_fc, buffer, sizeof(float) * 84, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(second_fc, 1, 84, 1, 1, "second_fc_copied.txt");


            in_h = fc_second_dim;
            in_w = m;
            out_h = fc_third_dim;
            out_w = m;

            block = {(unsigned int)tile_dim, (unsigned int)tile_dim};
            grid = {(unsigned int)ceil((float)in_h / block.x), (unsigned int)ceil((float)out_h / block.y)};
            shared_mem_dim = tile_dim * tile_dim * 2 * sizeof(float);
            matrix_product_shared<<<grid, block, shared_mem_dim>>>(fc_second_layer_dev, second_fc, third_fc, out_w, out_h, fc_second_dim, tile_dim);

            /****
             * Calcoliamo la funzione di Loss calcolando prima gli esponenziali sul dispositivo e poi portiamo i valori
             * sull'host per poter terminare il calcolo della Softmax facendo la sommatoria di tutti i valori ottenuti
             * per poi dividere ogni elemento del vettore per il valore calcolato.
            */
            exponential<<<1, fc_third_dim>>>(third_fc, fc_third_dim);
            
            
            // parametri_in_ingresso = fopen("base_third_fc.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(third_fc, buffer, sizeof(float) * 10, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(third_fc, 1, 10, 1, 1, "third_fc_copied.txt");
            
            
            cudaMemcpy(prediction, third_fc, sizeof(float) * fc_third_dim, cudaMemcpyDeviceToHost);

            // debug_print(third_fc, 1, 10, 1, 1, "third_fc.txt");


            summation = 0.0;
#if defined(USAGE) || defined(TEST)
            max = -1;
#endif
            for(int i = 0; i < fc_third_dim; i++) summation += prediction[i];
            for(int i = 0; i < fc_third_dim; i++) {
                prediction[i] = prediction[i] / summation;

#if defined(TEST) || defined(USAGE)
                if(prediction[i] > max){
                    max = prediction[i];
                    prediction_index = i;
                }
#endif
            }

#ifdef TEST
            if(target[prediction_index] != 0) prediction_counter++;
#endif
#ifdef TRAIN
            /****
             * Calcoliamo il valore della Loss.
            */
            loss = 0.0;
            for(int i = 0; i < fc_third_dim; i++){
                loss += target[i] * logf(prediction[i]);
            }
            loss = -loss;
            //printf("loss = %f\n", loss);
#endif
#ifdef TRAIN
            // Inizio della BackPropagation
            //---------------------------------------------------------------------------------------------------------------------------------------
            cudaMemcpy(prediction_dev, prediction, sizeof(float) * fc_third_dim, cudaMemcpyHostToDevice);
            cudaMemcpy(target_dev, target, sizeof(float) * fc_third_dim, cudaMemcpyHostToDevice);

            block = {(unsigned int)fc_third_dim};
            grid = {(unsigned int)(block.x / 1024 + 1)};
            subtraction<<<grid, block>>>(dZ2, prediction_dev, target_dev, fc_third_dim);
            
            // cudaDeviceSynchronize();
            // debug_print(dZ2, 1, 10, 1, 1, "dZ2_new.txt");
            // parametri_in_ingresso = fopen("base_dZ2.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dZ2, buffer, sizeof(float) * 10, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dZ2, 1, 10, 1, 1, "dZ2_copied.txt");

            //debug_print(second_fc, 1, 84, 1, 1, "second_fc_beforeu.txt");

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

            block = {(unsigned int)tile_dim, (unsigned int)tile_dim};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y)};
            shared_mem_dim = tile_dim * tile_dim * 2 * sizeof(float);
            matrix_product_transpose_shared<<<grid, block, shared_mem_dim>>>(dZ2, second_fc, dW2, out_w, out_h, in_w, tile_dim);

            // debug_print(dW2, 84, 10, 1, 1, "dW2_new.txt");
            // parametri_in_ingresso = fopen("base_dW2.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dW2, buffer, sizeof(float) * 10 * 84, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dW2, 84, 10, 1, 1, "dW2_copied.txt");
        
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

            block = {(unsigned int)tile_dim, (unsigned int)tile_dim};
            grid = {(unsigned int)ceil((float)in_h / block.x), (unsigned int)ceil((float)out_h / block.y)};
            shared_mem_dim = tile_dim * tile_dim * 2 * sizeof(float);
            matrix_transpose_product_shared<<<grid, block, shared_mem_dim>>>(fc_second_layer_dev, dZ2, dZ1, out_w, out_h, in_h, tile_dim);

            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y)};
            matrix_dot_product<<<grid, block>>>(second_fc, second_fc, gdZ1, out_w, out_h, 1);
            scalar_subtraction<<<grid, block>>>(gdZ1, gdZ1, out_w, out_h, 1);
            matrix_dot_product<<<grid, block>>>(dZ1, gdZ1, dZ1, out_w, out_h, 1);

            // debug_print(dZ1, 1, 84, 1, 1, "dZ1_new.txt");
            // parametri_in_ingresso = fopen("base_dZ1.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dZ1, buffer, sizeof(float) * 84, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dZ1, 1, 84, 1, 1, "dZ1_copied.txt");

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

            block = {(unsigned int)tile_dim, (unsigned int)tile_dim};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y)};
            shared_mem_dim = tile_dim * tile_dim * 2 * sizeof(float);
            matrix_product_transpose_shared<<<grid, block, shared_mem_dim>>>(dZ1, third_conv, dW1, out_w, out_h, in_w, tile_dim);

            // debug_print(dW1, 84, 120, 1, 1, "dW1_new.txt");
            // parametri_in_ingresso = fopen("base_dW1.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dW1, buffer, sizeof(float) * 84 * 120, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dW1, 84, 120, 1, 1, "dW1_copied.txt");

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

            block = {(unsigned int)tile_dim, (unsigned int)tile_dim};
            grid = {(unsigned int)ceil((float)in_h / block.x), (unsigned int)ceil((float)out_h / block.y)};
            shared_mem_dim = tile_dim * tile_dim * 2 * sizeof(float);
            matrix_transpose_product_shared<<<grid, block, shared_mem_dim>>>(fc_first_layer_dev, dZ1, dZ0, out_w, out_h, fc_second_dim, tile_dim);

            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y)};
            matrix_dot_product<<<grid, block>>>(third_conv, third_conv, gdZ0, out_w, out_h, 1);
            scalar_subtraction<<<grid, block>>>(gdZ0, gdZ0, out_w, out_h, 1);
            matrix_dot_product<<<grid, block>>>(dZ0, gdZ0, dZ0, out_w, out_h, 1);

            // debug_print(dZ0, 1, 120, 1, 1, "dZ0_new.txt");
            // parametri_in_ingresso = fopen("base_dZ0.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dZ0, buffer, sizeof(float) * 120, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dZ0, 1, 120, 1, 1, "dZ0_copied.txt");

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
            out_h = kernel_dim;
            out_w = kernel_dim;

            block = {(unsigned int)(in_w * in_h), 1, (unsigned int)(1024 / (in_w * in_h))};
            grid = {(unsigned int)ceil((float)(in_w * in_h) / block.x), (unsigned int)ceil((float)kernel_num_second_layer / block.y), (unsigned int)ceil((float)kernel_num_third_layer / block.z)};
            shared_mem_dim = (in_w * in_h + w_2 * h_2 * 1 * block.z) * sizeof(float);
            convolution_forNOutChannels_shared<<<grid, block, shared_mem_dim>>>(second_pool, dZ0, dF2, in_w, in_h, kernel_num_second_layer, w_2, h_2, 1, kernel_num_third_layer, out_w, out_h, kernel_num_second_layer, kernel_num_third_layer);

            // debug_print(dF2, 5, 5, 16, 120, "dF2_new.txt");
            // parametri_in_ingresso = fopen("base_dF2.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dF2, buffer, sizeof(float) * 5 * 5 * 16 * 120, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dF2, 5, 5, 16, 120, "dF2_copied.txt");
            
            /****
             * Cacoliamo la Full Convolution tra i kernel della terza convoluzione e dZ0 che è
             * la derivata, rispetto alla Loss, calcolata fino all'uscita della terza convoluzione.
             * h_2 e w_2 si riferiscono alle dimensioni spaziali della matrice dZ0.
             * Per calcolare le dimensine di uscita usiamo la formual standard.
             * Iteriamo su tutti i canali dei kernel, mantenendo fisso il canale di dZ0 per un intero ciclo
             * di j. Utilizziamo la funzione convolution3D che ci permette di aggiornare i valori di dA3.
            */
            in_h = kernel_dim;
            in_w = kernel_dim;
            h_2 = 1;
            w_2 = 1;
            padding_full_conv = h_2 - 1;
            out_h = (in_h + 2 * padding_full_conv - h_2) / stride_c + 1;
            out_w = (in_w + 2 * padding_full_conv - w_2) / stride_c + 1;

            block = {(unsigned int)min(10, out_w), (unsigned int)min(10, out_h), (unsigned int)min(10, kernel_num_second_layer)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_second_layer / block.z)};
            full_Convolution<<<grid, block>>>(kernels_third_layer_dev, dZ0, dA3, in_w, in_h, kernel_num_second_layer, w_2, h_2, 1, kernel_num_third_layer, out_w, out_h, kernel_num_second_layer, padding_full_conv);

            // debug_print(dA3, 5, 5, 16, 1, "dA3_new.txt");
            // parametri_in_ingresso = fopen("base_dA3.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dA3, buffer, sizeof(float) * 5 * 5 * 16, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dA3, 5, 5, 16, 1, "dA3_copied.txt");

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
            
            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_second_layer, 1024 / (min(32, out_w) * min(32, out_h)))};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_second_layer / block.z)};
            matrix_dot_product<<<grid, block>>>(second_pool, second_pool, dP1, out_h, out_w, kernel_num_second_layer);
            scalar_subtraction<<<grid, block>>>(dP1, dP1, out_w, out_h, kernel_num_second_layer);
            matrix_dot_product<<<grid, block>>>(dP1, dA3, dP1, out_w, out_h, kernel_num_second_layer);
            
            // debug_print(dP1, 5, 5, 16, 1, "dP1_new.txt");
            // parametri_in_ingresso = fopen("base_dP1.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dP1, buffer, sizeof(float) * 5 * 5 * 16, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dP1, 5, 5, 16, 1, "dP1_copied.txt");

            // block = {(unsigned int)out_w, (unsigned int)out_h};
            // grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            // for(int i = 0; i < kernel_num_second_layer; i++){
            //     matrix_dot_product<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(second_pool + (i * in_h * in_w), second_pool + (i * in_h * in_w), dP1 + (i * out_h * out_w), out_h, out_w);
            //     scalar_subtraction<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(dP1 + (i * in_w * in_h), dP1 + (i * in_w * in_h), out_w, out_h);
            //     matrix_dot_product<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(dP1 + (i * in_h * in_w), dA3 + (i * in_h * in_w), dP1 + (i * in_h * in_w), out_w, out_h);
            // }

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

            block = {(unsigned int)min(32, in_w), (unsigned int)min(32, in_h), (unsigned int)min(kernel_num_second_layer, (1024 / (min(32, in_w) * min(32, in_h))))};
            grid = {(unsigned int)ceil((float)in_w / block.x), (unsigned int)ceil((float)in_h / block.y), (unsigned int)ceil((float)kernel_num_second_layer / block.z),};
            inverse_avg_pooling_monolithic<<<grid, block>>>(dP1, dA2, second_conv, in_w, in_h, out_w, out_h, kernel_num_second_layer, stride_p, window_size);

            // debug_print(dA2, 10, 10, 16, 1, "dA2.txt");
            // debug_print(dA2, 10, 10, 16, 1, "dA2_new.txt");
            // parametri_in_ingresso = fopen("base_dA2.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dA2, buffer, sizeof(float) * 10 * 10 * 16, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dA2, 10, 10, 16, 1, "dA2_copied.txt");

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

            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_second_layer, 1024 / (min(32, out_w) * min(32, out_h)))};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_second_layer / block.z)};
            matrix_dot_product<<<grid, block>>>(second_conv, second_conv, dC1, out_w, out_h, kernel_num_second_layer);
            scalar_subtraction<<<grid, block>>>(dC1, dC1, out_w, out_h, kernel_num_second_layer);
            matrix_dot_product<<<grid, block>>>(dC1, dA2, dC1, out_w, out_h, kernel_num_second_layer);

            // debug_print(dC1, 10, 10, 16, 1, "dC1.txt");
            // debug_print(dC1, 10, 10, 16, 1, "dC1_new.txt");
            // parametri_in_ingresso = fopen("base_dC1.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dC1, buffer, sizeof(float) * 10 * 10 * 16, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dC1, 10, 10, 16, 1, "dC1_copied.txt");

            // block = {(unsigned int)out_w, (unsigned int)out_h};
            // grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            // for(int i = 0; i < kernel_num_second_layer; i++){
            //     matrix_dot_product<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(second_conv + (i * in_w * in_h), second_conv + (i * in_w * in_h), dC1 + (i * out_h * out_w), out_w, out_h);
            //     scalar_subtraction<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(dC1 + (i * in_h * in_w), dC1 + (i * in_h * in_w), out_w, out_h);
            //     matrix_dot_product<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(dC1 + (i * in_h * in_w), dA2 + (i * in_h * in_w), dC1 + (i * in_h * in_w), out_w, out_h);
            // }

            /****
             * Cacoliamo la Full Convolution tra i kernel della seconda convoluzione, F1, e dC1 che è
             * la derivata, rispetto alla Loss, calcolata fino all'uscita della seconda convoluzione.
             * h_2 e w_2 si riferiscono alle dimensioni spaziali della matrice dC1.
             * Per calcolare le dimensine di uscita usiamo la formual standard.
             * Iteriamo su tutti i canali dei kernel, mantenendo fisso il canale di dC1 per un intero ciclo
             * di j. Utilizziamo la funzione convolution3D che ci permette di aggiornare i valori di dA1.
            */
            in_h = kernel_dim;
            in_w = kernel_dim;
            h_2 = 10;
            w_2 = 10;
            padding_full_conv = h_2 - 1;
            out_h = (in_h + 2 * padding_full_conv - h_2) / stride_c + 1;
            out_w = (in_w + 2 * padding_full_conv - w_2) / stride_c + 1;

            block = {(unsigned int)min(10, out_w), (unsigned int)min(10, out_h), (unsigned int)min(10, kernel_num_first_layer)};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_first_layer / block.z)};
            full_Convolution<<<grid, block>>>(kernels_second_layer_dev, dC1, dA1, in_w, in_h, kernel_num_first_layer, w_2, h_2, 1, kernel_num_second_layer, out_w, out_h, kernel_num_first_layer, padding_full_conv);

            // debug_print(dA1, 14, 14, 6, 1, "dA1.txt");
            // debug_print(dA1, 14, 14, 6, 1, "dA1_new.txt");
            // parametri_in_ingresso = fopen("base_dA1.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dA1, buffer, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dA1, 14, 14, 6, 1, "dA1_copied.txt");

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
            out_h = kernel_dim;
            out_w = kernel_dim;

            block = {(unsigned int)(in_w * in_h), 1, (unsigned int)(1024 / (in_w * in_h))};
            grid = {(unsigned int)ceil((float)(in_w * in_h) / block.x), (unsigned int)ceil((float)kernel_num_first_layer / block.y), (unsigned int)ceil((float)kernel_num_second_layer / block.z)};
            shared_mem_dim = (in_w * in_h + w_2 * h_2 * 1 * block.z) * sizeof(float);
            convolution_forNOutChannels_shared<<<grid, block, shared_mem_dim>>>(first_pool, dC1, dF1, in_w, in_h, kernel_num_first_layer, w_2, h_2, 1, kernel_num_second_layer, out_w, out_h, kernel_num_first_layer, kernel_num_second_layer);

            // debug_print(dF1, 5, 5, 6, 16, "dF1_new.txt");
            // parametri_in_ingresso = fopen("base_dF1.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dF1, buffer, sizeof(float) * 5 * 5 * 6 * 16, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dF1, 5, 5, 6, 16, "dF1_copied.txt");

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

            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_first_layer, 1024 / (min(32, out_w) * min(32, out_h)))};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_first_layer / block.z)};
            matrix_dot_product<<<grid, block>>>(first_pool, first_pool, dP0, out_w, out_h, kernel_num_first_layer);
            scalar_subtraction<<<grid, block>>>(dP0, dP0, out_w, out_h, kernel_num_first_layer);
            matrix_dot_product<<<grid, block>>>(dP0, dA1, dP0, out_w, out_h, kernel_num_first_layer);

            // debug_print(dP0, 14, 14, 6, 1, "dP0_new.txt");
            // parametri_in_ingresso = fopen("base_dP0.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dP0, buffer, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dP0, 14, 14, 6, 1, "dP0_copied.txt");

            // block = {(unsigned int)out_w, (unsigned int)out_h};
            // grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            // for(int i = 0; i < kernel_num_first_layer; i++){
            //     matrix_dot_product<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(first_pool + (i * in_h * in_w), first_pool + (i * in_h * in_w), dP0 + (i * out_h * out_w), out_w, out_h);
            //     scalar_subtraction<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(dP0 + (i * in_h * in_w), dP0 + (i * in_h * in_w), out_w, out_h);
            //     matrix_dot_product<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(dP0 + (i * in_h * in_w), dA1 + (i * in_h * in_w), dP0 + (i * in_h * in_w), out_w, out_h);
            // }

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

            block = {(unsigned int)min(32, in_w), (unsigned int)min(32, in_h), (unsigned int)min(kernel_num_second_layer, (1024 / (min(32, in_w) * min(32, in_h))))};
            grid = {(unsigned int)ceil((float)in_w / block.x), (unsigned int)ceil((float)in_h / block.y), (unsigned int)ceil((float)kernel_num_second_layer / block.z),};
            inverse_avg_pooling_monolithic<<<grid, block>>>(dP0, dA0, first_conv, in_w, in_h, out_w, out_h, kernel_num_second_layer, stride_p, window_size);

            // debug_print(dA0, 28, 28, 6, 1, "dA0_new.txt");
            // parametri_in_ingresso = fopen("base_dA0.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dA0, buffer, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dA0, 28, 28, 6, 1, "dA0_copied.txt");
        
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

            block = {(unsigned int)min(32, out_w), (unsigned int)min(32, out_h), (unsigned int)min(kernel_num_first_layer, 1024 / (min(32, out_w) * min(32, out_h)))};
            grid = {(unsigned int)ceil((float)out_w / block.x), (unsigned int)ceil((float)out_h / block.y), (unsigned int)ceil((float)kernel_num_first_layer / block.z)};
            matrix_dot_product<<<grid, block>>>(first_conv, first_conv, dC0, out_w, out_h, kernel_num_first_layer);
            scalar_subtraction<<<grid, block>>>(dC0, dC0, out_w, out_h, kernel_num_first_layer);
            matrix_dot_product<<<grid, block>>>(dC0, dA0, dC0, out_w, out_h, kernel_num_first_layer);

            // debug_print(dC0, 28, 28, 6, 1, "dC0_new.txt");
            // parametri_in_ingresso = fopen("base_dC0.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dC0, buffer, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dC0, 28, 28, 6, 1, "dC0_copied.txt");
            
            // block = {(unsigned int)out_w, (unsigned int)out_h};
            // grid = {(unsigned int)(block.x / 32 + 1), (unsigned int)(block.y / 32 + 1)};
            // for(int i = 0; i < kernel_num_first_layer; i++){
            //     matrix_dot_product<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(first_conv + (i * in_w * in_h), first_conv + (i * in_w * in_h), dC0 + (i * out_h * out_w), out_w, out_h);
            //     scalar_subtraction<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(dC0 + (i * in_w * in_h), dC0 + (i * in_w * in_h), out_w, out_h);
            //     matrix_dot_product<<<grid, block/* , 0, stream[i%NUMBER_OF_STREAM] */>>>(dC0 + (i * in_w * in_h), dA0 + (i * in_w * in_h), dC0 + (i * in_w * in_h), out_w, out_h);
            // }

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
            out_h = kernel_dim;
            out_w = kernel_dim;

            block = {(unsigned int)(in_w * in_h), 1, (unsigned int)(1024 / (in_w * in_h))};
            grid = {(unsigned int)ceil((float)(in_w * in_h) / block.x), (unsigned int)ceil((float)1 / block.y), (unsigned int)ceil((float)kernel_num_first_layer / block.z)};
            shared_mem_dim = (in_w * in_h + w_2 * h_2 * 1 * block.z) * sizeof(float);
            convolution_forNOutChannels_shared<<<grid, block, shared_mem_dim>>>(img_dev, dC0, dF0, in_w, in_h, 1, w_2, h_2, 1, kernel_num_first_layer, out_w, out_h, 1, kernel_num_first_layer);

            // debug_print(img_dev, 32, 32, 1, 1, "Immagine.txt");
            // // debug_print(dF0, 5, 5, 1, 6, "dF0.txt");
            // debug_print(dF0, 5, 5, 1, 6, "dF0_new.txt");
            // parametri_in_ingresso = fopen("base_dF0.txt", "r");
            // load_parameter(buffer, parametri_in_ingresso);
            // cudaMemcpy(dF0, buffer, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
            // fclose(parametri_in_ingresso);
            // debug_print(dF0, 5, 5, 1, 6, "dF0_copied.txt");

            /*-------------------------------
                Fine calcolo delle derivate.
                Inizio aggiornamento dei parametri
            */
            block = {(unsigned int)min(1024, 5 * 5 * 6)};
            grid = {(unsigned int)ceil((float)(5 * 5 * 6) / block.x)};
            matrix_scalar_product<<<grid, block>>>(dF0, LEARNING_RATE, 5 * 5 * 6);
            subtraction<<<grid, block>>>(kernels_first_layer_dev, kernels_first_layer_dev, dF0, 5 * 5 * 6);

            block = {(unsigned int)min(1024, 5 * 5 * 6 * 16)};
            grid = {(unsigned int)ceil((float)(5 * 5 * 6 * 16) / block.x)};
            matrix_scalar_product<<<grid, block>>>(dF1, LEARNING_RATE, 5 * 5 * 6 * 16);
            subtraction<<<grid, block>>>(kernels_second_layer_dev, kernels_second_layer_dev, dF1, 5 * 5 * 6 * 16);

            block = {(unsigned int)min(1024, 5 * 5 * 16 * 120)};
            grid = {(unsigned int)ceil((float)(5 * 5 * 16 * 120) / block.x)};
            matrix_scalar_product<<<grid, block>>>(dF2, LEARNING_RATE, 5 * 5 * 16 * 120);
            subtraction<<<grid, block>>>(kernels_third_layer_dev, kernels_third_layer_dev, dF2, 5 * 5 * 16 * 120);

            block = {(unsigned int)min(1024, 84 * 120)};
            grid = {(unsigned int)ceil((float)(84 * 120) / block.x)};
            matrix_scalar_product<<<grid, block>>>(dW1, LEARNING_RATE, 84 * 120);
            subtraction<<<grid, block>>>(fc_first_layer_dev, fc_first_layer_dev, dW1, 84 * 120);

            block = {(unsigned int)min(1024, 10 * 84)};
            grid = {(unsigned int)ceil((float)(10 * 84) / block.x)};
            matrix_scalar_product<<<grid, block>>>(dW2, LEARNING_RATE, 10 * 84);
            subtraction<<<grid, block>>>(fc_second_layer_dev, fc_second_layer_dev, dW2, 10 * 84);

            /****
             * Gestione del salvataggio di:
             * loss
             * parametri
             * predizioni
             * tempo
            */

#ifndef TIME_TEST

            if(batch % 5 == 0){
                fprintf(loss_file, "%d\t%e\n", (batch + epoch * 60000), loss);       
                fflush(loss_file);             
            }

            if(batch % 1000 == 0){
                fprintf(prediction_file, "Epoch: %d\tIteration: %d\n", epoch, batch);
                for(int i = 0; i < 10; i++)fprintf(prediction_file, "%.3f\t", prediction[i]);
                fprintf(prediction_file, "\n");
                fflush(prediction_file);

                partial_time = time(NULL);
                fprintf(time_file, "Epoch: %d\tIteration: %d\t\t%02d:%02d\n", epoch, batch, (int)(difftime(partial_time, start_time)) / 60, (int)(difftime(partial_time, start_time)) % 60);
                fflush(time_file);
                start_time = partial_time;
            }

#endif
#endif
#ifdef TIME_TEST

            u_sec = stop_timer(&start, &stop);
            fprintf(time_file_test, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
            fflush(time_file_test);

#endif

        }

#if defined(TRAIN) && !defined(TIME_TEST)

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
        save_parameter(kernels_second_layer_dev, 5, 5, 6, 16, parameter_file);
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

#endif
        
    }

#ifdef TEST

    printf("Valori predetti correttamente, su %d: %d\n", batch_dim, prediction_counter);

#elif USAGE

    printf("Valore predetto: %d\n", prediction_index);

#endif

#ifndef TIME_TEST

    fclose(loss_file);
    fclose(time_file);
    fclose(prediction_file);

#else

    fclose(time_file_test);

#endif

    free(kernels_first_layer);
    free(kernels_second_layer);
    free(kernels_third_layer);
    free(fc_first_layer);
    free(fc_second_layer);
    free(prediction);
    
#if defined(TEST) || defined(TRAIN)

    free(data);

#endif

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

#ifdef TRAIN

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

#endif

    return 0;
}