#ifndef MP_FUNCTIONS_H
#define MP_FUNCTIONS_H

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __linux__
#include <sys/time.h>
#else
#include <time.h>
#endif

#include "v1\base.h"
#include "v2\shared.h"
#include "support\support_functions.h"

#define M1_W        1
#define M1_H        10
#define M1_Z        1

#define M2_W        1
#define M2_H        84
#define M2_Z        1

#define OUT_W       84
#define OUT_H       10
#define OUT_Z       1

#define SCALAR      3
#define TILE_DIM    32  

#endif