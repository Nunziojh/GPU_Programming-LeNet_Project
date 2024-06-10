#ifndef P_FUNCTIONS_H
#define P_FUNCTIONS_H

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
#include "v2\register.h"
#include "v3\monolithic.h"
#include "support\support_functions.h"

#define M1_W        16
#define M1_H        16
#define M1_Z        4

#define M2_W        32
#define M2_H        32
#define M2_Z        4

#define OUT_W       32
#define OUT_H       32
#define OUT_Z       4

#define STRIDE      2
#define WINDOW_SIZE 2 

#endif