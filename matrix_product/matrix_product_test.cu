#include "matrix_product_functions.h"

int rounds = 1;

int main(int argc, char **argv)
{
    srand(2);

    int i, j, r;

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

    float *host_m1 = (float *)malloc(sizeof(float) * M1_W * M1_H * M1_Z);
    float *host_m2 = (float *)malloc(sizeof(float) * M2_W * M2_H * M2_Z);
    float *host_output = (float *)malloc(sizeof(float) * OUT_W * OUT_H * OUT_Z);

    float *dev_m1, *dev_m2, *dev_output;
    cudaMalloc((void **)&dev_m1, M1_W * M1_H * M1_Z * sizeof(float));
    cudaMalloc((void **)&dev_m2, M2_W * M2_H * M2_Z * sizeof(float));
    cudaMalloc((void **)&dev_output, OUT_W * OUT_H * OUT_Z * sizeof(float));

#ifdef DEBUG_PRINT
    init_values(host_m1, host_m2, dev_m1, dev_m2);
#endif
    
    dim3 block, grid;
    unsigned int shared_mem_dim;

    for(r = 0; r < rounds; r++)
    {
#ifndef DEBUG_PRINT
        init_values(host_m1, host_m2, dev_m1, dev_m2);
#endif
        start_timer(&start);
        cudaProfilerStart();

#ifdef BASE_P
        block = {(unsigned int)min(32, OUT_W), (unsigned int)min(32, OUT_H)};
        grid = {(unsigned int)ceil((float)OUT_W / block.x), (unsigned int)ceil((float)OUT_H / block.y)};
        matrix_product<<<grid, block>>>(dev_m1, dev_m2, dev_output, OUT_W, OUT_H, M1_W);
#elif BASE_TP
        block = {(unsigned int)min(32, OUT_W), (unsigned int)min(32, OUT_H)};
        grid = {(unsigned int)ceil((float)OUT_W / block.x), (unsigned int)ceil((float)OUT_H / block.y)};
        matrix_transpose_product<<<grid, block>>>(dev_m1, dev_m2, dev_output, OUT_W, OUT_H, M1_H);
#elif BASE_PT
        block = {(unsigned int)min(32, OUT_W), (unsigned int)min(32, OUT_H)};
        grid = {(unsigned int)ceil((float)OUT_W / block.x), (unsigned int)ceil((float)OUT_H / block.y)};
        matrix_product_transpose<<<grid, block>>>(dev_m1, dev_m2, dev_output, OUT_W, OUT_H, M1_W);
#elif BASE_DP
        block = {(unsigned int)min(32, OUT_W), (unsigned int)min(32, OUT_H), (unsigned int)min(OUT_Z, 1024 / (min(32, OUT_W) * min(32, OUT_H)))};
        grid = {(unsigned int)ceil((float)OUT_W / block.x), (unsigned int)ceil((float)OUT_H / block.y), (unsigned int)ceil((float)OUT_Z / block.z)};
        matrix_dot_product<<<grid, block>>>(dev_m1, dev_m2, dev_output, OUT_W, OUT_H, OUT_Z);
#elif BASE_SP
        block = {(unsigned int)min(1024, M1_W * M1_H * M1_Z)};
        grid = {(unsigned int)ceil((float)(M1_W * M1_H * M1_Z) / block.x)};
        matrix_scalar_product<<<grid, block>>>(dev_m1, SCALAR, M1_W * M1_H * M1_Z);
#elif BASE_3D_SP//DA RIMUOVERE PERCHE' NON UTILIZZATO
        block = {(unsigned int)min(32, M1_W), (unsigned int)min(32, M1_H), (unsigned int)min(M1_Z, 1024 / (min(32, M1_W) * min(32, M1_H)))};
        grid = {(unsigned int)ceil((float)M1_W / block.x), (unsigned int)ceil((float)M1_H / block.y), (unsigned int)ceil((float)M1_Z / block.z)};
        matrix3D_scalar_product<<<grid, block>>>(dev_m1, (float)SCALAR, M1_W, M1_H, M1_Z);
#elif SHARED_P
        block = {(unsigned int)TILE_DIM, (unsigned int)TILE_DIM};
        grid = {(unsigned int)ceil((float)M1_W / block.x), (unsigned int)ceil((float)OUT_H / block.y)};
        shared_mem_dim = TILE_DIM * TILE_DIM * 2 * sizeof(float);
        matrix_product_shared<<<grid, block, shared_mem_dim>>>(dev_m1, dev_m2, dev_output, OUT_W, OUT_H, M1_W, TILE_DIM);
#elif SHARED_TP
        block = {(unsigned int)TILE_DIM, (unsigned int)TILE_DIM};
        grid = {(unsigned int)ceil((float)OUT_W / block.x), (unsigned int)ceil((float)OUT_H / block.y)}; //DOVREBBE ESSERE OUT_H INVECE DI M1_H
        shared_mem_dim = TILE_DIM * TILE_DIM * 2 * sizeof(float);
        matrix_transpose_product_shared<<<grid, block, shared_mem_dim>>>(dev_m1, dev_m2, dev_output, OUT_W, OUT_H, M1_H, TILE_DIM);
#elif SHARED_PT
        block = {(unsigned int)TILE_DIM, (unsigned int)TILE_DIM};
        grid = {(unsigned int)ceil((float)OUT_W / block.x), (unsigned int)ceil((float)OUT_H / block.y)};
        shared_mem_dim = TILE_DIM * TILE_DIM * 2 * sizeof(float);
        matrix_product_transpose_shared<<<grid, block, shared_mem_dim>>>(dev_m1, dev_m2, dev_output, OUT_W, OUT_H, M1_W, TILE_DIM);
#else
        fprintf(stderr, "Direttiva non valida!!\n");
        exit(1);
#endif

        cudaDeviceSynchronize();
        cudaProfilerStop();
        u_sec = stop_timer(&start, &stop);
        fprintf(file_time, "%02d:%02d:%03d:%03d\n", (int)(u_sec / 60000000), (int)(u_sec / 1000000) % 60, (int)(u_sec / 1000) % 1000, (int)(u_sec % 1000));
#ifdef DEBUG_PRINT
        debug_print(host_output, dev_output, "results.txt", f_res, OUT_W, OUT_H, OUT_Z);
#endif
    }

    free(host_m1);
    free(host_m2);
    free(host_output);
    cudaFree(dev_m1);
    cudaFree(dev_m2);
    cudaFree(dev_output);

    fclose(file_time);

    return 0;
}







