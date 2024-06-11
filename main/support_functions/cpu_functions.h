#ifndef CPUFUNCTIONS_H
#define CPUFUNCTIONS_H

__host__ void debug_print(float *matrice_dev, int w, int h, int c, int n, const char *filename);
__host__ void save_parameter(float *param, int w, int h, int c, int n, FILE *fp);
__host__ void load_parameter(float *output, FILE *fp);
__host__ void mean_max_min(float *matrix, float *mean, float *max, float *min, int w, int h, int c);
__host__ void mean_normalization(float *matrix_dev, int w, int h, int c);
__host__ void save_img(char *name, float *image);
__host__ void load_example_to_device(mnist_data data, float *img_dev, float *target);

#endif