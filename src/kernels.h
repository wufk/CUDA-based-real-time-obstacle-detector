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

__global__ void testkernel(pixel_t *d);

__global__ void transposeDisparity(pixel_t *d_disparity, pixel_t *d_transDisparity, int rows, int cols);

__global__ void columnReduction(pixel_t *d_disparity, pixel_t *d_columns, 
	int width, int rows, int cols, int c_cols);

__global__ void columnReductionMean(pixel_t *d_disparity, pixel_t *d_columns,
	int width, int rows, int cols, int c_cols);

#endif // !__KERNELS_H__
