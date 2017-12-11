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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <glm/glm.hpp>

#include "gpu.h"

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


__global__ void transposeDisparity(pixel_t *d_disparity, pixel_t *d_transDisparity, int rows, int cols);

__global__ void columnReduction(pixel_t *d_disparity, pixel_t *d_columns, 
	int width, int rows, int cols, int c_cols);

__global__ void columnReductionMean(pixel_t *d_disparity, pixel_t *d_columns,
	int width, int rows, int cols, int c_cols);

__global__ void kernComputeGroundDisp(float *d_groundDisp, int h, float baseline, float height, 
	float fu, float v0, float sinTilt, float cosTilt);

__global__ void kernComputeNegativeLogDataTermGrd(int h, float *d_groundDisparity, float *d_nLogPGaussian_, float *d_fn_, float *d_nd_cquad_,
	float fv, float tilt, float height, float cf, float sigmaA, float sigmaH, float sigmaD, float dmax, float dmin, float SQRT2, float PI, float pOut, float vhor);

__global__ void kernComputeNegativeLogDataTermObj(int fnmax, float *d_cquad_, float *d_nLogPGaussian_,
	float fu, float basline, float sigamD, float deltaz, float SQRT2, float PI, float pOut, float dmin, float dmax);

__global__ void kernComputeCostsG(int m_h, int m_w, float nLogPUniform_, float *d_costsG, float *d_nLogPGaussian_, float *d_cquad_, float *d_fn_, float *d_disp_colReduced);

__global__ void kernComputeCostsS(int m_w, int m_h, float *d_costsS, float *d_disp_colReduced, float nLogPUniform_, float cquad_, float nLogPGaussian_, float fn_);

__global__ void kernComputeCostsO(int m_w, int m_h, int fnmax, float *d_costsO, float *d_disp_colReduced, float *d_nLogPGaussian_, float *d_cquad_, float nLogPUniform_);

__global__ void kernComputeCostsOO(int m_w, int fnmax, int m_h, float *d_costsO, float *d_disp_colReduced, float *d_nLogPGaussian_, float *d_cquad_, float nLogPUniform_);

__global__ void kernScanCosts(int m_w, int m_h, float *d_costsG);

__global__ void kernScanCostsObj(int m_w, int fnmax, int m_h, float *d_costsO);

__global__ void kerndValid(int m_w, int m_h, float *d_valid);

__global__ void kerndSum(int m_w, int m_h, float *d_sum);

__global__ void kernScan1(int m_w, int m_h, float *data);

__global__ void kernScan2(int m_w, int m_h, int fnmax, float * data);

__global__ void kernScan1shf(int m_w, int m_h, float *data);

__global__ void kernWarpSum(int m_w, int m_h, int fnmax, float *data);

__global__ void KernDP(int m_w, int m_h, int fnmax, float *d_disparity_colReduced, float *d_sum, float *d_valid,
	float *d_costsG, float *d_costsO, float *d_costsS, float *d_costTableG, float *d_costTableO, float *d_costTableS, 
	float *d_dispTableG, float *d_dispTableO, float *d_dispTableS,
	glm::vec2 *d_indexTableG, glm::vec2 *d_indexTableO, glm::vec2 *d_indexTableS,
	float *d_costs0_, float *d_costs1_, float *d_costs2_O_G_, float *d_costs2_O_O_, float *d_costs2_O_S_, float *d_costs2_S_O_, 
	float N_LOG_0_0, float m_vhor
	);
#endif // !__KERNELS_H__
