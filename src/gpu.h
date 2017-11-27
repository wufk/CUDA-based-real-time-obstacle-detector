#pragma once

#include "common.h"
#include "kernels.h"
#include "matrix.h"


#include <algorithm>
#include <numeric>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

//////////////////////////////////////////////////////////////////////////////
// data cost functions
//////////////////////////////////////////////////////////////////////////////

static const float PI = static_cast<float>(M_PI);
static const float SQRT2 = static_cast<float>(M_SQRT2);

class gpuNegativeLogDataTermGrd
{
public:
	using CameraParameters = StixelWorld::CameraParameters;

	gpuNegativeLogDataTermGrd() {

	}

	gpuNegativeLogDataTermGrd(float dmax, float dmin, float sigmaD, float pOut, float pInv, const CameraParameters& camera,
		float* d_groundDisparity, float vhor, float sigmaH, float sigmaA, int h)
	{
		init(dmax, dmin, sigmaD, pOut, pInv, camera, d_groundDisparity, vhor, sigmaH, sigmaA, h);
	}

	inline float operator()(float d, int v) const
	{
		if (d < 0.f)
			return 0.f;

		return std::min(nLogPUniform_, h_nLogPGaussian_[v] + h_cquad_[v] * (d - h_fn_[v]) * (d - h_fn_[v]));
	}

	// pre-compute constant terms
	void init(float dmax, float dmin, float sigmaD, float pOut, float pInv, const CameraParameters& camera,
		float* d_groundDisparity, float vhor, float sigmaH, float sigmaA, int h);

	void destroy();

	float nLogPUniform_;
	float *h_nLogPGaussian_, *h_cquad_, *h_fn_;
	float *d_nLogPGaussian_, *d_cquad_, *d_fn_;
};

class gpuNegativeLogDataTermObj
{
public:
	using CameraParameters = StixelWorld::CameraParameters;

	gpuNegativeLogDataTermObj () {

	}

	gpuNegativeLogDataTermObj(float dmax, float dmin, float sigma, float pOut, float pInv, const CameraParameters& camera, float deltaz)
	{
		init(dmax, dmin, sigma, pOut, pInv, camera, deltaz);
	}

	void destroy();

	inline float operator()(float d, int fn) const
	{
		if (d < 0.f)
			return 0.f;

		return std::min(nLogPUniform_, h_nLogPGaussian_[fn] + h_cquad_[fn] * (d - fn) * (d - fn));
	}

	// pre-compute constant terms
	void init(float dmax, float dmin, float sigmaD, float pOut, float pInv, const CameraParameters& camera, float deltaz);

	float nLogPUniform_;
	float *h_nLogPGaussian_, *h_cquad_;
	float *d_nLogPGaussian_, *d_cquad_;
};

class gpuNegativeLogDataTermSky
{
public:
	gpuNegativeLogDataTermSky() {

	}

	gpuNegativeLogDataTermSky(float dmax, float dmin, float sigmaD, float pOut, float pInv, float fn = 0.f) : fn_(fn)
	{
		init(dmax, dmin, sigmaD, pOut, pInv, fn);
	}

	inline float operator()(float d) const
	{
		if (d < 0.f)
			return 0.f;

		return std::min(nLogPUniform_, nLogPGaussian_ + cquad_ * (d - fn_) * (d - fn_));
	}

	// pre-compute constant terms
	void init(float dmax, float dmin, float sigmaD, float pOut, float pInv, float fn);

	float nLogPUniform_, cquad_, nLogPGaussian_, fn_;
};

//////////////////////////////////////////////////////////////////////////////
// prior cost functions
//////////////////////////////////////////////////////////////////////////////

static const float N_LOG_0_3 = -static_cast<float>(log(0.3));
static const float N_LOG_0_5 = -static_cast<float>(log(0.5));
static const float N_LOG_0_7 = -static_cast<float>(log(0.7));
static const float N_LOG_0_0 = std::numeric_limits<float>::infinity();
static const float N_LOG_1_0 = 0.f;

class gpuNegativeLogPriorTerm
{
public:
	static const int G = 0;
	static const int O = 1;
	static const int S = 2;

	gpuNegativeLogPriorTerm(int h, float vhor, float dmax, float dmin, float b, float fu, float deltaz, float eps,
		float pOrd, float pGrav, float pBlg, const std::vector<float>& groundDisparity)
	{
		init(h, vhor, dmax, dmin, b, fu, deltaz, eps, pOrd, pGrav, pBlg, groundDisparity);
	}

	inline float getO0(int vT) const
	{
		return costs0_(vT, O);
	}
	inline float getG0(int vT) const
	{
		return costs0_(vT, G);
	}
	inline float getS0(int vT) const
	{
		return N_LOG_0_0;
	}

	inline float getOO(int vB, int d1, int d2) const
	{
		return costs1_(vB, O, O) + costs2_O_O_(d2, d1);
	}
	inline float getOG(int vB, int d1, int d2) const
	{
		return costs1_(vB, O, G) + costs2_O_G_(vB - 1, d1);
	}
	inline float getOS(int vB, int d1, int d2) const
	{
		return costs1_(vB, O, S) + costs2_O_S_(d1);
	}

	inline float getGO(int vB, int d1, int d2) const
	{
		return costs1_(vB, G, O);
	}
	inline float getGG(int vB, int d1, int d2) const
	{
		return costs1_(vB, G, G);
	}
	inline float getGS(int vB, int d1, int d2) const
	{
		return N_LOG_0_0;
	}

	inline float getSO(int vB, int d1, int d2) const
	{
		return costs1_(vB, S, O) + costs2_S_O_(d2, d1);
	}
	inline float getSG(int vB, int d1, int d2) const
	{
		return N_LOG_0_0;
	}
	inline float getSS(int vB, int d1, int d2) const
	{
		return N_LOG_0_0;
	}

	void init(int h, float vhor, float dmax, float dmin, float b, float fu, float deltaz, float eps,
		float pOrd, float pGrav, float pBlg, const std::vector<float>& groundDisparity);

	Matrixf costs0_, costs1_;
	Matrixf costs2_O_O_, costs2_O_G_, costs2_O_S_, costs2_S_O_;
};

class gpuStixelWorld : StixelWorld {
public:

	gpuStixelWorld(const Parameters& param, int rows, int cols) {
		param_ = param;
		d_disparity_colReduced = nullptr;
		d_disparity_original = nullptr;
		d_disparity_columns = nullptr;

		m_rows = rows;
		m_cols = cols;
		m_h = rows;
		m_w = m_cols / param_.stixelWidth;

		const CameraParameters& camera = param_.camera;
		const float sinTilt = sinf(camera.tilt);
		const float cosTilt = cosf(camera.tilt);
		m_vhor = m_h - 1 - (camera.v0 * cosTilt - camera.fu * sinTilt) / cosTilt;

		d_disparity_colReduced = nullptr;
		d_disparity_columns = nullptr;

		cudaMalloc((void **)&d_disparity_original, m_rows * m_cols * sizeof(float));
		cudaMalloc((void **)&d_disparity_colReduced, m_h * m_w * sizeof(float));
		cudaMalloc((void **)&d_disparity_columns, m_h * m_w * sizeof(float));
		data = new float[m_h * m_w];

		/* zero copy for colRecued */
		//cudaSetDeviceFlags(cudaDeviceMapHost);
		//cudaHostAlloc((void**)&h_disparity_colReduced, m_h * m_w * sizeof(float), cudaHostAllocMapped);
		//cudaHostGetDevicePointer((void**)&d_disparity_colReduced, (void*)h_disparity_colReduced, 0);
		

		preprocess(camera, sinTilt, cosTilt);
	}

	virtual void compute(const cv::Mat& disp, std::vector<Stixel>& stixels) override;

	void preprocess(const CameraParameters& camera, float sinTilt, float cosTilt);

	//methods
	void destroy();

private:
	Parameters param_;

	int m_rows;
	int m_cols;
	int m_h;
	int m_w;
	int m_vhor;

	float *d_disparity_original;
	float *d_disparity_colReduced;
	float *d_disparity_columns;
	float *d_groundDisp;

	float *h_disparity_colReduced;
	float *data;

	gpuNegativeLogDataTermGrd m_dataTermG;
	gpuNegativeLogDataTermObj m_dataTermO;
	gpuNegativeLogDataTermSky m_dataTermS;
};