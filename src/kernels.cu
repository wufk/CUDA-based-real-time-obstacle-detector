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

	int i_disparity = i * width;
	int index_disparity = j * cols + i * width;

	//compute mean
	float sum = 0.0f;
	for (int di = 0; di < width; ++di) {
		if (i_disparity + di >= cols) continue;
		sum += d_disparity[index_disparity + di];
	}
	sum /= width;

	mat[threadIdx.y][threadIdx.x] = sum;

	__syncthreads();

	// Copy data from shared memory to global memory
	// Check for bounds
	if (ti < rows && tj < cols) {
		d_output[tj * rows + rows - 1 - ti] = mat[threadIdx.x][threadIdx.y];
	}
		

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

__global__ void kernComputeGroundDisp(float * d_groundDisp, int h, float baseline, float height, float fu, float v0, float sinTilt, float cosTilt)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= h) return;

	d_groundDisp[h - 1 - idx] = 0.f;
	float val = baseline / height * (fu * sinTilt + (idx - v0) * cosTilt);
	if (val > 0.f) {
		d_groundDisp[h - 1 - idx] = val;
	}

	//groundDisparity[h - 1 - v] = std::max((camera.baseline / camera.height) * (camera.fu * sinTilt + (v - camera.v0) * cosTilt), 0.f);

}

__global__ void kernComputeNegativeLogDataTermGrd(int h, float * d_groundDisparity, float * d_nLogPGaussian_, float * d_fn_, float * d_cquad_, 
	float fv, float tilt, float height, float cf, float sigmaA, float sigmaH, float sigmaD, float dmax, float dmin, float SQRT2, float PI, float pOut, float vhor)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= h) return;

	const float tmp = ((vhor - idx) / fv + tilt) / height;
	const float sigmaR2 = cf * cf * (tmp * tmp * sigmaH * sigmaH + sigmaA * sigmaA);
	const float sigma = sqrtf(sigmaD * sigmaD + sigmaR2);

	const float fn = d_groundDisparity[idx];
	const float ANorm = 0.5f * (erff((dmax - fn) / (SQRT2 * sigma)) - erff((dmin - fn) / (SQRT2 * sigma)));
	d_nLogPGaussian_[idx] = logf(ANorm) + logf(sigma * sqrtf(2.f * PI)) - logf(1.f - pOut);
	d_fn_[idx] = fn;

	// coefficient of quadratic part
	d_cquad_[idx] = 1.f / (2.f * sigma * sigma);


	// Gaussian distribution term
	//const int h = static_cast<int>(groundDisparity.size());
	//nLogPGaussian_.resize(h);
	//cquad_.resize(h);
	//fn_.resize(h);
	//for (int v = 0; v < h; v++)
	//{
	//	const float tmp = ((vhor - v) / camera.fv + camera.tilt) / camera.height;
	//	const float sigmaR2 = cf * cf * (tmp * tmp * sigmaH * sigmaH + sigmaA * sigmaA);
	//	const float sigma = sqrtf(sigmaD * sigmaD + sigmaR2);

	//	const float fn = groundDisparity[v];
	//	const float ANorm = 0.5f * (erff((dmax - fn) / (SQRT2 * sigma)) - erff((dmin - fn) / (SQRT2 * sigma)));
	//	nLogPGaussian_[v] = logf(ANorm) + logf(sigma * sqrtf(2.f * PI)) - logf(1.f - pOut);
	//	fn_[v] = fn;

	//	// coefficient of quadratic part
	//	cquad_[v] = 1.f / (2.f * sigma * sigma);
	//}

	/*
	if (d < 0.f)
	return 0.f;

	return std::min(nLogPUniform_, h_nLogPGaussian_[v] + h_cquad_[v] * (d - h_fn_[v]) * (d - h_fn_[v]));
	*/
}

__global__ void kernComputeNegativeLogDataTermObj(int fnmax, float * d_cquad_, float * d_nLogPGaussian_, float fu, float baseline, float sigmaD, float deltaz, float SQRT2, float PI, float pOut, float dmin, float dmax)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= fnmax) return;

	const float sigmaZ = idx * idx * deltaz / (fu * baseline);
	const float sigma = sqrtf(sigmaD * sigmaD + sigmaZ * sigmaZ);

	const float ANorm = 0.5f * (erff((dmax - idx) / (SQRT2 * sigma)) - erff((dmin - idx) / (SQRT2 * sigma)));
	d_nLogPGaussian_[idx] = logf(ANorm) + logf(sigma * sqrtf(2.f * PI)) - logf(1.f - pOut);

	d_cquad_[idx] = 1.f / (2.f * sigma * sigma);

	//nLogPGaussian_.resize(fnmax);
	//cquad_.resize(fnmax);
	//for (int fn = 0; fn < fnmax; fn++)
	//{
	//	const float sigmaZ = fn * fn * deltaz / (camera.fu * camera.baseline);
	//	const float sigma = sqrtf(sigmaD * sigmaD + sigmaZ * sigmaZ);

	//	const float ANorm = 0.5f * (erff((dmax - fn) / (SQRT2 * sigma)) - erff((dmin - fn) / (SQRT2 * sigma)));
	//	nLogPGaussian_[fn] = logf(ANorm) + logf(sigma * sqrtf(2.f * PI)) - logf(1.f - pOut);

	//	// coefficient of quadratic part
	//	cquad_[fn] = 1.f / (2.f * sigma * sigma);
	//}
}

__global__ void kernComputeCostsG(int m_w, int m_h, float nLogPUniform_, float * d_costsG, float * d_nLogPGaussian_, float * d_cquad_, float * d_fn_, float * d_disp_colReduced)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int jdx = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= m_w || jdx >= m_h) {
		return;
	}
	int pos = idx * m_h + jdx;
	float d = d_disp_colReduced[pos];
	if (d < 0.f) {
		d_costsG[pos] = 0.f;
		return;
	}
	float cost = d_nLogPGaussian_[jdx] + d_cquad_[jdx] * (d - d_fn_[jdx]) * (d - d_fn_[jdx]);
	if (nLogPUniform_ < cost) {
		d_costsG[pos] = nLogPUniform_;
	}
	else {
		d_costsG[pos] = cost;
	}

	/*
	if (d < 0.f)
	return 0.f;

	return std::min(nLogPUniform_, h_nLogPGaussian_[v] + h_cquad_[v] * (d - h_fn_[v]) * (d - h_fn_[v]));
	*/
}

__global__ void kernComputeCostsS(int m_w, int m_h, float * d_costsS, float *d_disp_colReduced,  float nLogPUniform_, float cquad_, float nLogPGaussian_, float fn_)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int jdx = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= m_w || jdx >= m_h) {
		return;
	}
	int pos = idx * m_h + jdx;
	float d = d_disp_colReduced[pos];

	if (d < 0.f) {
		d_costsS[pos] = 0.f;
		return;
	}
	float cost = nLogPGaussian_ + cquad_ * (d - fn_) * (d - fn_);
	if (nLogPUniform_ < cost) {
		d_costsS[pos] = nLogPUniform_;
	}
	else {
		d_costsS[pos] = cost;
	}

	/**
	if (d < 0.f)
	return 0.f;

	return std::min(nLogPUniform_, nLogPGaussian_ + cquad_ * (d - fn_) * (d - fn_));
	*/
}

__global__ void kernComputeCostsO(int m_w, int m_h, int fnmax, float * d_costsO, float * d_disp_colReduced, float * d_nLogPGaussian_, float * d_cquad_, float nLogPUniform_)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int jdx = blockIdx.y * blockDim.y + threadIdx.y;
	int zdx = blockIdx.z * blockDim.z + threadIdx.z;

	if (idx >= m_w || jdx >= m_h || zdx >= fnmax) {
		return;
	}

	int pos = idx * m_h + jdx;
	int val_pos = idx * m_h * fnmax + jdx * fnmax + zdx;

	float d = d_disp_colReduced[pos];
	if (d < 0.f) {
		d_costsO[val_pos] = 0.f;
		return;
	}
	float cost = d_nLogPGaussian_[zdx] + d_cquad_[zdx] * (d - zdx) * (d - zdx);
	if (cost > nLogPUniform_) {
		d_costsO[val_pos] = nLogPUniform_;
	}
	else {
		d_costsO[val_pos] = cost;
	}
	/*
	if (d < 0.f)
	return 0.f;

	return std::min(nLogPUniform_, h_nLogPGaussian_[fn] + h_cquad_[fn] * (d - fn) * (d - fn));
	*/
}

__global__ void kernComputeCostsOO(int m_w, int fnmax, int m_h, float * d_costsO, float * d_disp_colReduced, float * d_nLogPGaussian_, float * d_cquad_, float nLogPUniform_)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //u
	int jdx = blockIdx.y * blockDim.y + threadIdx.y; //fn
	int zdx = blockIdx.z * blockDim.z + threadIdx.z; //v


	if (idx >= m_w || jdx >= fnmax || zdx >= m_h) {
		return;
	}

	int pos = idx * m_h + zdx;
	int val_pos = idx * m_h * fnmax + jdx * m_h + zdx;

	float d = d_disp_colReduced[pos];
	if (d < 0.f) {
		d_costsO[val_pos] = 0.f;
		return;
	}
	float cost = d_nLogPGaussian_[jdx] + d_cquad_[jdx] * (d - jdx) * (d - jdx);
	if (cost > nLogPUniform_) {
		d_costsO[val_pos] = nLogPUniform_;
	}
	else {
		d_costsO[val_pos] = cost;
	}
}

__global__ void kernScanCosts(int m_w, int m_h, float *d_costsG)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= m_w) return;

	thrust::device_ptr<float> thrust_tmpTerm(d_costsG + idx * m_h);
	thrust::device_ptr<float> thrust_costsG(d_costsG + idx * m_h);

	thrust::inclusive_scan(thrust::device, thrust_tmpTerm, thrust_tmpTerm + m_h, thrust_costsG);
}

__global__ void kernScanCostsObj(int m_w, int fnmax, int m_h, float * d_costsO)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //u
	int jdx = blockIdx.y * blockDim.y + threadIdx.y; //fn

	if (idx >= m_w || jdx >= fnmax) {
		return;
	}

	thrust::device_ptr<float> thrust_tmpTerm(d_costsO + idx * m_h * fnmax + jdx * m_h);
	thrust::device_ptr<float> thrust_costsG(d_costsO + idx * m_h * fnmax + jdx * m_h);

	thrust::inclusive_scan(thrust::device, thrust_tmpTerm, thrust_tmpTerm + m_h, thrust_costsG);
	
}
