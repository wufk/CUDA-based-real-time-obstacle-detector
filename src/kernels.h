#pragma once
#ifndef __KERNELS_H__
#define __KERNELS_H__

typedef float pixel_t;

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sm_20_atomic_functions.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "gpu.h"

struct gpuStixelWorld::Parameters;
struct Stixel;


#endif // !__KERNELS_H__
