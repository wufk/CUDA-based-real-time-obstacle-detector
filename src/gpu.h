#pragma once

#include "common.h"

class gpuStixelWorld : StixelWorld {
public:
	struct Parameters
	{
		// stixel width
		int stixelWidth;

		// disparity range
		float dmin;
		float dmax;

		// disparity measurement uncertainty
		float sigmaG;
		float sigmaO;
		float sigmaS;

		// camera height and tilt uncertainty
		float sigmaH;
		float sigmaA;

		// outlier rate
		float pOutG;
		float pOutO;
		float pOutS;

		// probability of invalid disparity
		float pInvG;
		float pInvO;
		float pInvS;

		// probability for regularization
		float pOrd;
		float pGrav;
		float pBlg;

		float deltaz;
		float eps;

		// camera parameters
		CameraParameters camera;

		// default settings
		Parameters()
		{
			// stixel width
			stixelWidth = 7;

			// disparity range
			dmin = 0;
			dmax = 64;

			// disparity measurement uncertainty
			sigmaG = 1.5f;
			sigmaO = 1.5f;
			sigmaS = 1.2f;

			// camera height and tilt uncertainty
			sigmaH = 0.05f;
			sigmaA = 0.05f * static_cast<float>(CV_PI) / 180.f;

			// outlier rate
			pOutG = 0.15f;
			pOutO = 0.15f;
			pOutS = 0.4f;

			// probability of invalid disparity
			pInvG = 0.34f;
			pInvO = 0.3f;
			pInvS = 0.36f;

			// probability for regularization
			pOrd = 0.1f;
			pGrav = 0.1f;
			pBlg = 0.001f;

			deltaz = 3.f;
			eps = 1.f;

			// camera parameters
			camera = CameraParameters();
		}
	};

	virtual void compute(const cv::Mat& disp, std::vector<Stixel>& stixels) override;

private:

	Parameters param_;
};