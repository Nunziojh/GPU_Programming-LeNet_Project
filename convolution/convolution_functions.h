#ifndef C_FUNCTIONS_H
#define C_FUNCTIONS_H

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
#include "v3\optimized.h"
#include "v4\monolithic.h"
#include "v5\monolithic_shared.h"
#include "support\support_functions.h"

#define INPUT_X     14
#define INPUT_Y     INPUT_X
#define INPUT_Z     6
#define INPUT_N     1

#define KERNEL_X    10
#define KERNEL_Y    KERNEL_X
#define KERNEL_Z    1
#define KERNEL_N    16

#define PADDING     0

#define OUT_X       5
#define OUT_Y       OUT_X
#define OUT_Z       6
#define OUT_N       16

#endif