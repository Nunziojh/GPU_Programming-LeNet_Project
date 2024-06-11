#ifndef LENET_H
#define LENET_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <float.h>
#include <string.h>

#ifdef __linux__
#include <sys/time.h>
#else
#include <time.h>
#endif

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#define MNIST_STATIC
#include "support_functions\\mnist.h"

#include "support_functions\gpu_functions.h"
#include "support_functions\cpu_functions.h"
#include "..\convolution\v5\monolithic_shared.h"
#include "..\convolution\v4\monolithic.h"
#include "..\matrix_product\v2\shared.h"
#include "..\matrix_product\v1\base.h"
#include "..\pooling\v3\monolithic.h"

#define LEARNING_RATE 0.5


#endif