#include "functions.h"
    
int rounds = 1;

int main(int argc, char **argv)
{
    srand(time(NULL));

    int i, j, r;
    int total_out_dim = OUT_X * OUT_Y * OUT_Z * OUT_N;

    FILE *file_time;
#ifdef DEBUG_PRINT
    FILE *f_res;
#endif
    file_time = fopen("time.txt", "w");

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
    
    dim3 block, grid;
    unsigned int shared_mem_dim;
    int max_third_dim;

for(r = 0; r < rounds; r++)
{
#ifdef DEBUG_PRINT
    clean_vector_dev<<<(unsigned int)(total_out_dim / min(total_out_dim, 1024) + 1), min(total_out_dim, 1024)>>>(dev_output, OUT_N * OUT_X * OUT_Y * OUT_Z);
    clean_vector_host(host_output, OUT_N * OUT_X * OUT_Y * OUT_Z);
    cudaDeviceSynchronize();
#else
    init_values(host_input, host_kernel, host_output, dev_input, dev_kernel, dev_output);
#endif
    start_timer(&start);
    cudaProfilerStart();

#ifdef BASE_F
    block = {(unsigned int)min(32, OUT_X), (unsigned int)min(32, OUT_Y)};
    grid = {(unsigned int)(OUT_X / block.x + 1), (unsigned int)(OUT_Y / block.y + 1)};
    for(i = 0; i < OUT_Z; i++){
        for(j = 0; j < INPUT_Z; j++){
            convolution_base<<<grid, block>>>(dev_input + (j * INPUT_X * INPUT_Y), dev_output + (i * OUT_X * OUT_Y), dev_kernel + (j * KERNEL_X * KERNEL_Y + (i * KERNEL_X * KERNEL_Y * KERNEL_Z)), INPUT_X, OUT_X, KERNEL_X, 0);
        }
    }
#elif BASE_DF
    block = {(unsigned int)min(32, OUT_X), (unsigned int)min(32, OUT_Y)};
    grid = {(unsigned int)(OUT_X / block.x + 1), (unsigned int)(OUT_Y / block.y + 1)};
    for(i = 0; i < OUT_N; i++){
        for(j = 0; j < OUT_Z; j++){
            convolution_base<<<grid, block>>>(dev_input + (j * INPUT_X * INPUT_Y), dev_output + (j * OUT_X * OUT_Y + (i * OUT_X * OUT_Y * OUT_Z)), dev_kernel + (i * KERNEL_X * KERNEL_Y), INPUT_X, OUT_X, KERNEL_X, PADDING);
        }
    }
#elif BASE_DA
    for(i = 0; i < INPUT_N; i++){
        for(j = 0; j < OUT_Z; j++){
            convolution_base<<<grid, block>>>(dev_input + (j * INPUT_X * INPUT_Y + (i * INPUT_X * INPUT_Y * INPUT_Z)), dev_output + (j * OUT_X * OUT_Y), dev_kernel + (i * KERNEL_X * KERNEL_Y), INPUT_X, OUT_X, KERNEL_X, PADDING);
        }
    }
#elif SHARED_F
    block = {(unsigned int)min(32, max(OUT_X, INPUT_X)), (unsigned int)min(32, max(OUT_Y, INPUT_Y))};
    grid = {(unsigned int)(max(OUT_X, INPUT_X) / block.x + 1), (unsigned int)( max(OUT_Y, INPUT_Y) / block.y + 1)};
    shared_mem_dim = (INPUT_X * INPUT_Y + KERNEL_X * KERNEL_Y) * sizeof(float);
    for(i = 0; i < OUT_Z; i++){
        for(j = 0; j < INPUT_Z; j++){
            convolution_shared<<<grid, block, shared_mem_dim>>>(dev_input + (j * INPUT_X * INPUT_Y), dev_output + (i * OUT_X * OUT_Y), dev_kernel + (j * KERNEL_X * KERNEL_Y + (i * KERNEL_X * KERNEL_Y * KERNEL_Z)), INPUT_X, OUT_X, KERNEL_X, PADDING);
        }
    }
#elif SHARED_DF
    block = {(unsigned int)min(32, max(OUT_X, INPUT_X)), (unsigned int)min(32, max(OUT_Y, INPUT_Y))};
    grid = {(unsigned int)(max(OUT_X, INPUT_X) / block.x + 1), (unsigned int)( max(OUT_Y, INPUT_Y) / block.y + 1)};
    shared_mem_dim = (INPUT_X * INPUT_Y + KERNEL_X * KERNEL_Y) * sizeof(float);
    for(i = 0; i < OUT_N; i++){
        for(j = 0; j < OUT_Z; j++){
            convolution_shared<<<grid, block, shared_mem_dim>>>(dev_input + (j * INPUT_X * INPUT_Y), dev_output + (j * OUT_X * OUT_Y + (i * OUT_X * OUT_Y * OUT_Z)), dev_kernel + (i * KERNEL_X * KERNEL_Y), INPUT_X, OUT_X, KERNEL_X, PADDING);
        }
    }
#elif SHARED_DA
    block = {(unsigned int)min(32, max(OUT_X, INPUT_X)), (unsigned int)min(32, max(OUT_Y, INPUT_Y))};
    grid = {(unsigned int)(max(OUT_X, INPUT_X) / block.x + 1), (unsigned int)( max(OUT_Y, INPUT_Y) / block.y + 1)};
    shared_mem_dim = (INPUT_X * INPUT_Y + KERNEL_X * KERNEL_Y) * sizeof(float);
    for(i = 0; i < INPUT_N; i++){
        for(j = 0; j < OUT_Z; j++){
            convolution_shared<<<grid, block, shared_mem_dim>>>(dev_input + (j * INPUT_X * INPUT_Y + (i * INPUT_X * INPUT_Y * INPUT_Z)), dev_output + (j * OUT_X * OUT_Y), dev_kernel + (i * KERNEL_X * KERNEL_Y), INPUT_X, OUT_X, KERNEL_X, PADDING);
        }
    }
#elif OPTIMIZED_F
    block = {(unsigned int)min(32, max(OUT_X, (INPUT_X + 2 * PADDING))), (unsigned int)min(32, max(OUT_Y, (INPUT_Y + 2 * PADDING)))};
    grid = {(unsigned int)ceil((float)max(OUT_X, (INPUT_X + 2 * PADDING)) / block.x), (unsigned int)ceil((float)max(OUT_X, (INPUT_X + 2 * PADDING)) / block.y)};
    shared_mem_dim = ((INPUT_X + 2 * PADDING) * (INPUT_Y + 2 * PADDING) + KERNEL_X * KERNEL_Y) * sizeof(float);
    for(i = 0; i < OUT_Z; i++){
        for(j = 0; j < INPUT_Z; j++){
            convolution_base<<<grid, block, shared_mem_dim>>>(dev_input + (j * INPUT_X * INPUT_Y), dev_output + (i * OUT_X * OUT_Y), dev_kernel + (j * KERNEL_X * KERNEL_Y + (i * KERNEL_X * KERNEL_Y * KERNEL_Z)), INPUT_X, OUT_X, KERNEL_X, 0);
        }
    }
#elif OPTIMIZED_DF
    block = {(unsigned int)min(32, max(OUT_X, (INPUT_X + 2 * PADDING))), (unsigned int)min(32, max(OUT_Y, (INPUT_Y + 2 * PADDING)))};
    grid = {(unsigned int)ceil((float)max(OUT_X, (INPUT_X + 2 * PADDING)) / block.x), (unsigned int)ceil((float)max(OUT_X, (INPUT_X + 2 * PADDING)) / block.y)};
    shared_mem_dim = ((INPUT_X + 2 * PADDING) * (INPUT_Y + 2 * PADDING) + KERNEL_X * KERNEL_Y) * sizeof(float);
    for(i = 0; i < OUT_N; i++){
        for(j = 0; j < OUT_Z; j++){
            convolution_optimized<<<grid, block, shared_mem_dim>>>(dev_input + (j * INPUT_X * INPUT_Y), dev_output + (j * OUT_X * OUT_Y + (i * OUT_X * OUT_Y * OUT_Z)), dev_kernel + (i * KERNEL_X * KERNEL_Y), INPUT_X, OUT_X, KERNEL_X, PADDING);
        }
    }
#elif OPTIMIZED_DA
    block = {(unsigned int)min(32, max(OUT_X, (INPUT_X + 2 * PADDING))), (unsigned int)min(32, max(OUT_Y, (INPUT_Y + 2 * PADDING)))};
    grid = {(unsigned int)ceil((float)max(OUT_X, (INPUT_X + 2 * PADDING)) / block.x), (unsigned int)ceil((float)max(OUT_X, (INPUT_X + 2 * PADDING)) / block.y)};
    shared_mem_dim = ((INPUT_X + 2 * PADDING) * (INPUT_Y + 2 * PADDING) + KERNEL_X * KERNEL_Y) * sizeof(float);
    for(i = 0; i < INPUT_N; i++){
        for(j = 0; j < OUT_Z; j++){
            convolution_optimized<<<grid, block, shared_mem_dim>>>(dev_input + (j * INPUT_X * INPUT_Y + (i * INPUT_X * INPUT_Y * INPUT_Z)), dev_output + (j * OUT_X * OUT_Y), dev_kernel + (i * KERNEL_X * KERNEL_Y), INPUT_X, OUT_X, KERNEL_X, PADDING);
        }
    }
#elif MONOLITHIC_F
    block = {(unsigned int)min(10, OUT_X), (unsigned int)min(10, OUT_Y), (unsigned int)min(10, OUT_Z)};
    grid = {(unsigned int)ceil((float)OUT_X / block.x), (unsigned int)ceil((float)OUT_Y / block.y), (unsigned int)ceil((float)OUT_Z / block.z)};
    convolution_3D<<<grid, block>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, OUT_X, OUT_Y, OUT_Z);
#elif MONOLITHIC_DF
    block = {(unsigned int)min(10, (OUT_X * OUT_Y)), (unsigned int)min(10, OUT_Z), (unsigned int)min(10, OUT_N)};
    grid = {(unsigned int)ceil((float)(OUT_X * OUT_Y) / block.x), (unsigned int)ceil((float)OUT_Z / block.y), (unsigned int)(OUT_N / block.z + 1)};
    convolution_forNOutChannels<<<grid, block>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, OUT_X, OUT_Y, OUT_Z, OUT_N);
#elif MONOLITHIC_DA
    block = {(unsigned int)min(10, OUT_X), (unsigned int)min(10, OUT_Y), (unsigned int)min(10, OUT_Z)};
    grid = {(unsigned int)ceil((float)OUT_X / block.x), (unsigned int)ceil((float)OUT_Y / block.y), (unsigned int)ceil((float)OUT_Z / block.z)};
    full_Convolution<<<grid, block>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, KERNEL_N, OUT_X, OUT_Y, OUT_Z, PADDING);
#elif MONOLITHIC_S_F
    max_third_dim = (1024.0 * 48.0 / sizeof(float) - (INPUT_X * INPUT_Y * INPUT_Z)) / (KERNEL_X * KERNEL_Y * KERNEL_Z);
    shared_mem_dim = (INPUT_X * INPUT_Y * INPUT_Z + KERNEL_X * KERNEL_Y * KERNEL_Z * max_third_dim) * sizeof(float);
    block = {(unsigned int)min(32, INPUT_X), (unsigned int)min(32, INPUT_Y), (unsigned int)min(max_third_dim, (1024 / (min(32, INPUT_X) * min(32, INPUT_Y))))};
    grid = {(unsigned int)ceil(((float)OUT_X / block.x)), (unsigned int)ceil(((float)OUT_Y / block.y)), (unsigned int)ceil((float)OUT_Z / block.z)};
    convolution_3D_shared<<<grid, block, shared_mem_dim>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, KERNEL_N, OUT_X, OUT_Y, OUT_Z);
#elif MONOLITHIC_S_DF
    block = {(unsigned int)(INPUT_X * INPUT_Y), 1, (unsigned int)(1024 / (INPUT_X * INPUT_Y))};
    grid = {(unsigned int)ceil((float)(INPUT_X * INPUT_Y) / block.x), (unsigned int)ceil((float)OUT_Z / block.y), (unsigned int)ceil((float)OUT_N / block.z)};
    shared_mem_dim = (INPUT_X * INPUT_Y + KERNEL_X * KERNEL_Y * KERNEL_Z * block.z) * sizeof(float);
    convolution_forNOutChannels_shared<<<grid, block, shared_mem_dim>>>(dev_input, dev_kernel, dev_output, INPUT_X, INPUT_Y, INPUT_Z, KERNEL_X, KERNEL_Y, KERNEL_Z, KERNEL_N, OUT_X, OUT_Y, OUT_Z, OUT_N);
#else
    fprintf(stderr, "Direttiva non valida!!\n");
    exit(1);
#endif

    cudaDeviceSynchronize();
    cudaProfilerStop();
    u_sec = stop_timer(&start, &stop);
    fprintf(file_time, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
#ifdef DEBUG_PRINT
    debug_print(host_output, dev_output, "results.txt", f_res, OUT_X, OUT_Y, OUT_Z, OUT_N);
#endif
}

    free(host_input);
    free(host_kernel);
    free(host_output);
    cudaFree(dev_input);
    cudaFree(dev_kernel);
    cudaFree(dev_output);

    fclose(file_time);

    return 0;
}