#include "functions.h"
    
int rounds = 1;

int main(int argc, char **argv)
{
    srand(time(NULL));

    int i, j, r;

    FILE *file_time_base, *file_time_shared, *file_time_optimized, *file_time_monolithic, *file_time_monolithic_shared;
#ifdef DEBUG_PRINT
    FILE *f_res;
#endif
    file_time_base = fopen("time_base.txt", "w");
    file_time_shared = fopen("time_shared.txt", "w");
    file_time_optimized = fopen("time_optimized.txt", "w");
    file_time_monolithic = fopen("time_monolithic.txt", "w");
    file_time_monolithic_shared = fopen("time_monolithic_shared.txt", "w");

#ifdef __linux__
    struct timeval start, stop;
#else
    struct timespec start, stop;
#endif
    long int u_sec;

    float *host_input = (float *)malloc(sizeof(float) * INPUT_X * INPUT_Y * INPUT_Z * INPUT_N);
    float *host_kernel = (float *)malloc(sizeof(float) * KERNEL_X * KERNEL_Y * KERNEL_Z * KERNEL_N);
    float *host_output = (float *)malloc(sizeof(float) * OUT_X * OUT_Y * OUT_Z * OUT_N);

    float *dev_input, *dev_kernel, *dev_output;
    cudaMalloc((void **)&dev_input, INPUT_X * INPUT_Y * INPUT_Z * INPUT_N * sizeof(float));
    cudaMalloc((void **)&dev_kernel, KERNEL_X * KERNEL_Y * KERNEL_Z * KERNEL_N * sizeof(float));
    cudaMalloc((void **)&dev_output, OUT_X * OUT_Y * OUT_Z * OUT_N * sizeof(float));

#ifdef DEBUG_PRINT
    init_values(host_input, host_kernel, dev_input, dev_kernel, dev_output);
    printf("\t---INPUT---\n");
    debug_print(host_input, NULL, NULL, NULL, INPUT_X, INPUT_Y, INPUT_Z, INPUT_N);
    printf("\t---KERNEL---\n");
    debug_print(host_kernel, NULL, NULL, NULL, KERNEL_X, KERNEL_Y, KERNEL_Z, KERNEL_N);
#endif
    
    dim3 block, grid;
    unsigned int shared_mem_dim;

for(r = 0; r < rounds; r++)
{
   /**
    * Versione Base.
    * A seconda della struttura di: input, kernel e output è necessario modificare la struttura della chiamata alla funzione.
   */
#ifdef DEBUG_PRINT
    clean_vector<<<(unsigned int)((OUT_X * OUT_Y * OUT_Z * OUT_N) / min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024) + 1), min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024)>>>(dev_output, OUT_N * OUT_X * OUT_Y * OUT_Z);
#else
    init_values(host_input, host_kernel, dev_input, dev_kernel, dev_output);
#endif
    start_timer(&start);
    cudaProfilerStart();
    block = {(unsigned int)min(32, OUT_X), (unsigned int)min(32, OUT_Y)};
    grid = {(unsigned int)(OUT_X / block.x + 1), (unsigned int)(OUT_Y / block.y + 1)};
    for(i = 0; i < INPUT_N; i++){
        for(j = 0; j < OUT_Z; j++){
            convolution_base<<<grid, block>>>(dev_input + (j * INPUT_X * INPUT_Y + (i * INPUT_X * INPUT_Y * INPUT_Z)), dev_output + (j * OUT_X * OUT_Y), dev_kernel + (i * KERNEL_X * KERNEL_Y), INPUT_X, OUT_X, KERNEL_X, 0);
        }
    }
    cudaDeviceSynchronize();
    cudaProfilerStop();
    u_sec = stop_timer(&start, &stop);
    fprintf(file_time_base, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
#ifdef DEBUG_PRINT
    debug_print(host_output, dev_output, "risultatli_base.txt", f_res, OUT_X, OUT_Y, OUT_Z, OUT_N);
#endif

    /**
     * Versione con Shared Memory
    * A seconda della struttura di: input, kernel e output è necessario modificare la struttura della chiamata alla funzione.
    */
#ifdef DEBUG_PRINT
    clean_vector<<<(unsigned int)((OUT_X * OUT_Y * OUT_Z * OUT_N) / min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024) + 1), min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024)>>>(dev_output, OUT_N * OUT_X * OUT_Y * OUT_Z);
#else
    init_values(host_input, host_kernel, dev_input, dev_kernel, dev_output);
#endif
    start_timer(&start);
    cudaProfilerStart();
    block = {(unsigned int)min(32, max(OUT_X, INPUT_X)), (unsigned int)min(32, max(OUT_Y, INPUT_Y))};
    grid = {(unsigned int)(max(OUT_X, INPUT_X) / block.x + 1), (unsigned int)( max(OUT_Y, INPUT_Y) / block.y + 1)};
    shared_mem_dim = (INPUT_X * INPUT_Y + KERNEL_X * KERNEL_Y) * sizeof(float);
    for(i = 0; i < OUT_Z; i++){
        for(j = 0; j < INPUT_Z; j++){
            convolution_shared<<<grid, block, shared_mem_dim>>>(dev_input + (j * INPUT_X * INPUT_Y), dev_output + (i * OUT_X * OUT_Y), dev_kernel + (j * KERNEL_X * KERNEL_Y + (i * KERNEL_X * KERNEL_Y * KERNEL_Z)), INPUT_X, OUT_X, KERNEL_X);
        }
    }
    cudaDeviceSynchronize();
    cudaProfilerStop();
    u_sec = stop_timer(&start, &stop);
    fprintf(file_time_shared, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
#ifdef DEBUG_PRINT
    debug_print(host_output, dev_output, "risultati_shared.txt", f_res, OUT_X, OUT_Y, OUT_Z, OUT_N);
#endif

    /**
     * Versione Optimized.
    */
#ifdef DEBUG_PRINT
     clean_vector<<<(unsigned int)((OUT_X * OUT_Y * OUT_Z * OUT_N) / min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024) + 1), min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024)>>>(dev_output, OUT_N * OUT_X * OUT_Y * OUT_Z);
#else
    init_values(host_input, host_kernel, dev_input, dev_kernel, dev_output);
#endif
    start_timer(&start);
    cudaProfilerStart();
    block = {(unsigned int)min(32, max(OUT_X, INPUT_X)), (unsigned int)min(32, max(OUT_Y, INPUT_Y))};
    grid = {(unsigned int)(max(OUT_X, INPUT_X) / block.x + 1), (unsigned int)( max(OUT_Y, INPUT_Y) / block.y + 1)};
    shared_mem_dim = (INPUT_X * INPUT_Y + KERNEL_X * KERNEL_Y) * sizeof(float);
    for(i = 0; i < INPUT_N; i++){
        for(j = 0; j < OUT_Z; j++){
            convolution_optimized<<<grid, block, shared_mem_dim>>>(dev_input + (j * INPUT_X * INPUT_Y + (i * INPUT_X * INPUT_Y * INPUT_Z)), dev_output + (j * OUT_X * OUT_Y), dev_kernel + (i * KERNEL_X * KERNEL_Y), INPUT_X, OUT_X, KERNEL_X, 0);
        }
    }
    cudaDeviceSynchronize();
    cudaProfilerStop();
    u_sec = stop_timer(&start, &stop);
    fprintf(file_time_optimized, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
#ifdef DEBUG_PRINT
    debug_print(host_output, dev_output, "risultatli_optimized.txt", f_res, OUT_X, OUT_Y, OUT_Z, OUT_N);
#endif

    /**
     * Versione Monolithic.
     * A seconda della struttura di: input, kernel e output è nesessario selezionare la funzione corretta.
    */
#ifdef DEBUG_PRINT
    clean_vector<<<(unsigned int)((OUT_X * OUT_Y * OUT_Z * OUT_N) / min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024) + 1), min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024)>>>(dev_output, OUT_N * OUT_X * OUT_Y * OUT_Z);
#else
    init_values(host_input, host_kernel, dev_input, dev_kernel, dev_output);
#endif
    start_timer(&start);
    cudaProfilerStart();

    // Da usare nella Foreward
    block = {(unsigned int)min(10, OUT_X), (unsigned int)min(10, OUT_Y), (unsigned int)min(10, OUT_Z)};
    grid = {(unsigned int)ceil((float)OUT_X / block.x), (unsigned int)ceil((float)OUT_Y / block.y), (unsigned int)ceil((float)OUT_Z / block.z)};
    convolution_3D<<<grid, block>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, OUT_X, OUT_Y, OUT_Z);

    // Da usare nella Backward dove le uscite sono dA3 e dA1
    // block = {(unsigned int)min(10, OUT_X), (unsigned int)min(10, OUT_Y), (unsigned int)min(10, OUT_Z)};
    // grid = {(unsigned int)ceil((float)OUT_X / block.x), (unsigned int)ceil((float)OUT_Y / block.y), (unsigned int)(OUT_Z / block.z + 1)};
    // full_Convolution<<<grid, block>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, KERNEL_N, OUT_X, OUT_Y, OUT_Z, PADDING);

    // Da usare nella Backward dove le uscite sono dF2, dF1, dF0
    // block = {(unsigned int)min(10, (OUT_X * OUT_Y)), (unsigned int)min(10, OUT_Z), (unsigned int)min(10, OUT_N)};
    // grid = {(unsigned int)ceil((float)(OUT_X * OUT_Y) / block.x), (unsigned int)ceil((float)OUT_Z / block.y), (unsigned int)(OUT_N / block.z + 1)};
    // convolution_forNOutChannels<<<grid, block>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, OUT_X, OUT_Y, OUT_Z, OUT_N);

    cudaDeviceSynchronize();
    cudaProfilerStop();
    u_sec = stop_timer(&start, &stop);
    fprintf(file_time_monolithic, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
#ifdef DEBUG_PRINT
    debug_print(host_output, dev_output, "risultati_monolithic.txt", f_res, OUT_X, OUT_Y, OUT_Z, OUT_N);
#endif

    /**
     * Versione Monolithic con Shared Memory.
     * A seconda della struttura di: input, kernel e output è necessario selezionare la funzione corretta.
    */
#ifdef DEBUG_PRINT
    clean_vector<<<(unsigned int)((OUT_X * OUT_Y * OUT_Z * OUT_N) / min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024) + 1), min((OUT_X * OUT_Y * OUT_Z * OUT_N), 1024)>>>(dev_output, OUT_N * OUT_X * OUT_Y * OUT_Z);
#else
    init_values(host_input, host_kernel, dev_input, dev_kernel, dev_output);
#endif
    start_timer(&start);
    cudaProfilerStart();
    /***
     * Nella definizione di blocchi griglia, un pirmo approccio potrebbe essere di associare in cascate le risorse ai tre assi a partire dall'asse x
     * poi l'asse y e infine l'asse z. Questo approccio favorisce le prime due dimensioni e quindi riduco il numero di blocchi sull'asse z e di conseguenza
     * faccio meno accessi alla memoria globale per copiare i kernel su quella condivisa.
     * Oppure potrei trattare le due dimensioni spazili con la stessa priosità ma questo porta, potenzialmente ad avere più threads/blocchi sull'asse z e quindi
     * a dover muovere più volte gli stessi dati dai kernel.
     * Tutto questo ragionamento vale solo se si utilizza la memoria condivisa.
     * 
     * Calcolo 1024 / kernel_spatial_dim e il valore ottenuto mi da la dimensione massima suul'asse x, poi dopo aver allocato i threads sull'asse x calcolo 1024 / block.x
     * e ottengo il numero di threads sull'asse y e infine calcolo 1024 / (block.x * block.y) così ottengo un blocco con un volume totale inferiore a 1024 che da
     * precedenza alle dimensioni spaziali, prima x poi y, in modo che ci siano meno spostamenti possibili dalla memoria globale a quella shared di kernel, perché lavorando
     * sulla stessa faccia d'uscita copio solo un kernel per blocco. La memoria condivisa che alloco dipenderà quindi da block.z
    */
    //Da usare nella Foreward
    block = {(unsigned int)min(32, INPUT_X), (unsigned int)min(32, INPUT_Y), (unsigned int)(1024 / (min(32, INPUT_X) * min(32, INPUT_Y)))};
    grid = {(unsigned int)ceil(((float)OUT_X / block.x)), (unsigned int)ceil(((float)OUT_Y / block.y)), (unsigned int)ceil((double)OUT_Z / block.z)};
    shared_mem_dim = (INPUT_X * INPUT_Y * INPUT_Z + KERNEL_X * KERNEL_Z * KERNEL_Z * KERNEL_N) * sizeof(float);
    convolution_3D_shared<<<grid, block, shared_mem_dim>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, KERNEL_N, OUT_X, OUT_Y, OUT_Z);

    // Da usare nella Backward dove le uscite sono dF2, dF1, dF0
    // block = {(unsigned int)(INPUT_X * INPUT_Y), 1, (unsigned int)(1024 / (INPUT_X * INPUT_Y))};
    // grid = {(unsigned int)ceil((float)(INPUT_X * INPUT_Y) / block.x), (unsigned int)ceil((float)OUT_Z / block.y), (unsigned int)ceil((float)OUT_N / block.z)};
    // shared_mem_dim = (INPUT_X * INPUT_Y + KERNEL_X * KERNEL_Y * KERNEL_Z * block.z) * sizeof(float);
    // convolution_forNOutChannels_shared<<<grid, block, shared_mem_dim>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, KERNEL_N, OUT_X, OUT_Y, OUT_Z, OUT_N);
    
    cudaDeviceSynchronize();
    u_sec = stop_timer(&start, &stop);
    fprintf(file_time_monolithic_shared, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));

#ifdef DEBUG_PRINT
    debug_print(host_output, dev_output, "risultati_monolithic_shared.txt", f_res, OUT_X, OUT_Y, OUT_Z, OUT_N);
#endif
}

    free(host_input);
    free(host_kernel);
    free(host_output);
    cudaFree(dev_input);
    cudaFree(dev_kernel);
    cudaFree(dev_output);

    fclose(file_time_base);
    fclose(file_time_shared);
    fclose(file_time_optimized);
    fclose(file_time_monolithic);
    fclose(file_time_monolithic_shared);

    return 0;
}