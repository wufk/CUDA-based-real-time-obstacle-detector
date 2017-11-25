#pragma once
#include <opencv2/opencv.hpp>

struct Stixel
{
	int u;
	int vT;
	int vB;
	int width;
	float disp;
};

class StixelWorld
{
public:

	struct CameraParameters
	{
		float fu;
		float fv;
		float u0;
		float v0;
		float baseline;
		float height;
		float tilt;

		// default settings
		CameraParameters()
		{
			fu = 1.f;
			fv = 1.f;
			u0 = 0.f;
			v0 = 0.f;
			baseline = 0.2f;
			height = 1.f;
			tilt = 0.f;
		}
	};

	struct Parameters
	{
		// stixel width
		int stixelWidth;

		// minimum and maximum disparity
		int minDisparity;
		int maxDisparity;

		// camera parameters
		CameraParameters camera;

		// default settings
		Parameters()
		{
			// stixel width
			stixelWidth = 3;

			// maximum disparity
			minDisparity = -1;
			maxDisparity = 64;

			// camera parameters
			camera = CameraParameters();
		}
	};

	virtual void compute(const cv::Mat& disp, std::vector<Stixel>& stixels) = 0;

	std::vector<int> lowerPath;
	std::vector<int> upperPath;

protected:
	Parameters param_;
};
