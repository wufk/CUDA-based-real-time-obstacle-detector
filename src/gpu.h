#pragma once

#include "common.h"
#include "kernels.h"
#include "matrix.h"


#include <algorithm>
#include <numeric>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

#define BLOCKSIZE 32

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
		float* d_groundDisparity, float vhor, float sigmaH, float sigmaA, int h, int w)
	{
		m_h = h;
		m_w = w;
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
	
	void computeCostsG(float* d_disp_colReduced);

	void destroy();

	int m_h, m_w;
	float nLogPUniform_;
	float *h_nLogPGaussian_, *h_cquad_, *h_fn_;
	float *d_nLogPGaussian_, *d_cquad_, *d_fn_;

	float *d_costsG, *h_costsG;
};

class gpuNegativeLogDataTermObj
{
public:
	using CameraParameters = StixelWorld::CameraParameters;

	gpuNegativeLogDataTermObj () {

	}

	gpuNegativeLogDataTermObj(float dmax, float dmin, float sigma, float pOut, float pInv, const CameraParameters& camera, float deltaz, int w, int h)
	{
		m_h = h;
		m_w = w;
		// Gaussian distribution term
		fnmax = static_cast<int>(dmax);

		init(dmax, dmin, sigma, pOut, pInv, camera, deltaz);
	}
	void computeCostsO(float *d_disp_colReduced);
	void destroy();

	inline float operator()(float d, int fn) const
	{
		if (d < 0.f)
			return 0.f;

		return std::min(nLogPUniform_, h_nLogPGaussian_[fn] + h_cquad_[fn] * (d - fn) * (d - fn));
	}

	// pre-compute constant terms
	void init(float dmax, float dmin, float sigmaD, float pOut, float pInv, const CameraParameters& camera, float deltaz);

	int m_w;
	int m_h;
	int fnmax;

	float nLogPUniform_;
	float *h_nLogPGaussian_, *h_cquad_;
	float *d_nLogPGaussian_, *d_cquad_;
	float *d_costsO, *h_costsO;
};

class gpuNegativeLogDataTermSky
{
public:
	gpuNegativeLogDataTermSky() {

	}

	gpuNegativeLogDataTermSky(float dmax, float dmin, float sigmaD, float pOut, float pInv, int w, int h, float fn = 0.f) : fn_(fn)
	{
		m_h = h;
		m_w = w;
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
	void destroy();
	void computeCostsS(float *d_disparity_colReduced);

	int m_h, m_w;
	float nLogPUniform_, cquad_, nLogPGaussian_, fn_;
	float *d_costsS, *h_costsS;
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

	gpuNegativeLogPriorTerm() {

	}

	gpuNegativeLogPriorTerm(int h, float vhor, float dmax, float dmin, float b, float fu, float deltaz, float eps,
		float pOrd, float pGrav, float pBlg, float* h_groundDisparity)
	{
		fnmax = static_cast<int>(dmax);
		m_h = h;
		init(h, vhor, dmax, dmin, b, fu, deltaz, eps, pOrd, pGrav, pBlg, h_groundDisparity);
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
		float pOrd, float pGrav, float pBlg, float* h_groundDisparity);
	void destroy();

	Matrixf costs0_, costs1_;
	Matrixf costs2_O_O_, costs2_O_G_, costs2_O_S_, costs2_S_O_;

	int m_h, m_w;
	int fnmax;

	float *d_costs0_, *d_costs1_;
	float *d_costs2_O_O_, *d_costs2_O_G_, *d_costs2_O_S_, *d_costs2_S_O_;
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

		//d_disparity_colReduced = nullptr;
		//d_disparity_columns = nullptr;


		//cudaSetDeviceFlags(cudaDeviceMapHost);
		cudaMalloc((void **)&d_disparity_original, m_rows * m_cols * sizeof(float));
		//cudaHostAlloc((void**)&h_disparity_original, m_rows * m_cols * sizeof(float), cudaHostAllocMapped);
		//cudaHostGetDevicePointer((void**)&d_disparity_original, (void*)h_disparity_original, 0);
			
		
		//data = new float[m_h * m_w];
		cudaMalloc((void **)&d_disparity_colReduced, m_h * m_w * sizeof(float));
		cudaMallocHost((void**)&h_disparity_colReduced, m_h * m_w * sizeof(float));
		/* zero copy for colRecued */
		//cudaSetDeviceFlags(cudaDeviceMapHost);
		//cudaHostAlloc((void**)&h_disparity_colReduced, m_h * m_w * sizeof(float), cudaHostAllocMapped);
		//cudaHostGetDevicePointer((void**)&d_disparity_colReduced, (void*)h_disparity_colReduced, 0);
		
		cudaMalloc((void**)&d_sum, m_h * m_w * sizeof(float));
		cudaMalloc((void**)&d_valid, m_h * m_w * sizeof(float));
		cudaMallocHost((void**)&h_sum, m_h * m_w * sizeof(float));
		cudaMallocHost((void**)&h_valid, m_h * m_w * sizeof(float));

		cudaMalloc((void**)&d_costTableG, m_h * m_w * sizeof(float));
		cudaMalloc((void**)&d_costTableS, m_h * m_w * sizeof(float));
		cudaMalloc((void**)&d_costTableO, m_h * m_w * sizeof(float));

		cudaMalloc((void**)&d_dispTableG, m_h * m_w * sizeof(float));
		cudaMalloc((void**)&d_dispTableS, m_h * m_w * sizeof(float));
		cudaMalloc((void**)&d_dispTableO, m_h * m_w * sizeof(float));

		cudaMalloc((void**)&d_indexTableG, m_h * m_w * sizeof(glm::vec2));
		cudaMalloc((void**)&d_indexTableS, m_h * m_w * sizeof(glm::vec2));
		cudaMalloc((void**)&d_indexTableO, m_h * m_w * sizeof(glm::vec2));

		cudaMallocHost((void**)&h_costTableG, m_h * m_w * sizeof(float));
		cudaMallocHost((void**)&h_costTableS, m_h * m_w * sizeof(float));
		cudaMallocHost((void**)&h_costTableO, m_h * m_w * sizeof(float));

		cudaMallocHost((void**)&h_dispTableG, m_h * m_w * sizeof(float));
		cudaMallocHost((void**)&h_dispTableS, m_h * m_w * sizeof(float));
		cudaMallocHost((void**)&h_dispTableO, m_h * m_w * sizeof(float));

		cudaMallocHost((void**)&h_indexTableG, m_h * m_w * sizeof(glm::vec2));
		cudaMallocHost((void**)&h_indexTableS, m_h * m_w * sizeof(glm::vec2));
		cudaMallocHost((void**)&h_indexTableO, m_h * m_w * sizeof(glm::vec2));

		preprocess(camera, sinTilt, cosTilt);
		// this line can be earsed forever cudaMalloc((void **)&d_disparity_columns, m_h * m_w * sizeof(float));
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
	float *d_sum, *d_valid;
	float *d_costTableG, *d_costTableO, *d_costTableS;
	float *d_dispTableG, *d_dispTableO, *d_dispTableS;
	glm::vec2 *d_indexTableG, *d_indexTableO, *d_indexTableS;

	float *h_groundDisp;
	float *h_disparity_original;
	float *h_disparity_colReduced;
	float *h_sum, *h_valid;
	float *h_costTableG, *h_costTableO, *h_costTableS;
	float *h_dispTableG, *h_dispTableO, *h_dispTableS;
	glm::vec2 *h_indexTableG, *h_indexTableO, *h_indexTableS;
	float *data;

	gpuNegativeLogDataTermGrd m_dataTermG;
	gpuNegativeLogDataTermObj m_dataTermO;
	gpuNegativeLogDataTermSky m_dataTermS;
	gpuNegativeLogPriorTerm m_priorTerm;
};