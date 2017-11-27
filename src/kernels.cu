#include "kernels.h"

__global__ void testkernel(pixel_t * d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	return;
}

//rows and cols are the row number and col number of the UN-transposed matrix
//rows = h, cols = w
__global__ void transposeDisparity(pixel_t * d_input, pixel_t * d_output, int rows, int cols)
{
	// Allocate appropriate shared memory
	__shared__ float mat[32][32 + 1];

	// Compute input and output index
	int bx = blockIdx.x * blockDim.x;     // Compute block offset - this is number of global threads in X before this block
	int by = blockIdx.y * blockDim.y;     // Compute block offset - this is number of global threads in Y before this block
	int i = bx + threadIdx.x;              // Global input x index - Same as previous kernels
	int j = by + threadIdx.y;              // Global input y index - Same as previous kernels

	int ti = by + threadIdx.x;              // Global output x index - remember the transpose
	int tj = bx + threadIdx.y;              // Global output y index - remember the transpose

	if (i < cols && j < rows)
		mat[threadIdx.y][threadIdx.x] = d_input[j * cols + i];

	__syncthreads();

	// Copy data from shared memory to global memory
	// Check for bounds
	if (ti < rows && tj < cols)
		d_output[tj * rows + rows - 1 - ti] = mat[threadIdx.x][threadIdx.y]; // Switch threadIdx.x and threadIdx.y from input read

}

//rows = m_rows = h, cols = m_cols, c_cols = w
//i: w, j:h
__global__ void columnReduction(pixel_t *d_disparity, pixel_t *d_output,
	int width, int rows, int cols, int c_cols)
{
	__shared__ float mat[32][32 + 1];

	int bx = blockIdx.x * blockDim.x;     // Compute block offset - this is number of global threads in X before this block
	int by = blockIdx.y * blockDim.y;     // Compute block offset - this is number of global threads in Y before this block
	int i = bx + threadIdx.x;              // Global input x index - Same as previous kernels
	int j = by + threadIdx.y;              // Global input y index - Same as previous kernels

	if (i >= cols || j >= rows) return;

	int ti = by + threadIdx.x;              // Global output x index - remember the transpose
	int tj = bx + threadIdx.y;              // Global output y index - remember the transpose

	int i_ori = i * width;
	int index_dis = j * cols + i * width;

	//compute mean
	float sum = 0.0f;
	for (int di = 0; di < width; ++di) {
		if (i_ori + di >= cols) continue;
		sum += d_disparity[index_dis + di];
	}
	sum /= width;

	mat[threadIdx.y][threadIdx.x] = sum;

	__syncthreads();

	// Copy data from shared memory to global memory
	// Check for bounds
	if (ti < rows && tj < cols)
		d_output[tj * rows + rows - 1 - ti] = mat[threadIdx.x][threadIdx.y]; // Switch threadIdx.x and threadIdx.y from input read

	//int idx = blockIdx.x * blockDim.x + threadIdx.x;  
	//int jdx = blockIdx.y * blockDim.y + threadIdx.y;

	//if (idx >= rows || jdx >= c_cols) {
	//	return;
	//}

	//int index_in = idx * c_cols + jdx;
	//int index_out = jdx * rows + rows - 1 - idx;

	//int index_dis = idx * cols + jdx * width;
	//int j_original = jdx * width;

	////TODO use median not mean
	//float sum = 0.0f;
	//for (int dj = 0; dj < width; ++dj) {
	//	if (j_original + dj >= cols) continue;
	//	sum += d_disparity[index_dis + dj];
	//}
	//sum /= width;

	//d_output[index_out] = sum;
	////d_output[index_in] = sum;
}

//rows = m_rows = h, x(i) dimension, cols = m_cols, c_cols = w y(j) dimension
__global__ void columnReductionMean(pixel_t * d_disparity, pixel_t * d_output, int width, int rows, int cols, int c_cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int jdx = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= rows || jdx >= c_cols) {
		return;
	}

	int index_in = idx * c_cols + jdx;

	int index_dis = idx * cols + jdx * width;
	int j_original = jdx * width;

	//TODO use median not mean
	float sum = 0.0f;
	for (int dj = 0; dj < width; ++dj) {
		if (j_original + dj >= cols) continue;
		sum += d_disparity[index_dis + dj];
	}
	sum /= width;

	d_output[index_in] = sum;
}
