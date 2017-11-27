#pragma once
#include "common.h"

class cpuStixelWorld : StixelWorld {
public:
	cpuStixelWorld(const Parameters& param) {
		param_ = param;
	}

	std::vector<int> lowerPath;
	std::vector<int> upperPath;

	virtual void compute(const cv::Mat& disp, std::vector<Stixel>& stixels) override;

private:
	float averageDisparity(const cv::Mat& disparity, const cv::Rect& rect, int minDisp, int maxDisp);
private:
	Parameters param_;
};