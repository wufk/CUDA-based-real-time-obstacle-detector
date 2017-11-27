#include "main.h"

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v*(1.f - s);
	tab[2] = v*(1.f - s*h);
	tab[3] = v*(1.f - s*(1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

static cv::Scalar dispToColor(float disp, float maxdisp)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp, maxdisp) / maxdisp);
}

static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
}

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cout << "usage: " << argv[0] << " left-image-format right-image-format camera.xml" << std::endl;
		return -1;
	}

	// stereo sgbm
	const int wsize = 11;
	const int numDisparities = 64;
	const int P1 = 8 * wsize * wsize;
	const int P2 = 32 * wsize * wsize;
	cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(0, numDisparities, wsize, P1, P2,
		0, 0, 0, 0, 0, cv::StereoSGBM::MODE_SGBM_3WAY);

	// input camera parameters
	const cv::FileStorage cvfs(argv[3], CV_STORAGE_READ);
	CV_Assert(cvfs.isOpened());
	const cv::FileNode node(cvfs.fs, NULL);
	StixelWorld::Parameters param;
	param.camera.fu = node["FocalLengthX"];
	param.camera.fv = node["FocalLengthY"];
	param.camera.u0 = node["CenterX"];
	param.camera.v0 = node["CenterY"];
	param.camera.baseline = node["BaseLine"];
	param.camera.height = node["Height"];
	param.camera.tilt = node["Tilt"];
	param.dmin = -1;
	param.dmax = numDisparities;

	cv::Mat dummyRead = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);

	cpuStixelWorld stixelWorld(param);
	gpuStixelWorld t_stixelWorld(param, dummyRead.rows, dummyRead.cols);

	for (int frameno = 1;; frameno++)
	{
		cv::Mat left_img = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat right_img = cv::imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);

		if (left_img.empty() || right_img.empty())
		{
			std::cerr << "imread failed." << std::endl;
			break;
		}

		if (left_img.channels() > 1) {
			cv::cvtColor(left_img, left_img, CV_RGB2GRAY);
		}
		if (right_img.channels() > 1) {
			cv::cvtColor(right_img, right_img, CV_RGB2GRAY);
		}

		CV_Assert(left_img.size() == right_img.size() && left_img.type() == right_img.type());

		switch (left_img.type())
		{
		case CV_8U:
			break;
		case CV_16U:
			// conver to CV_8U
			cv::normalize(left_img, left_img, 0, 255, cv::NORM_MINMAX);
			cv::normalize(right_img, right_img, 0, 255, cv::NORM_MINMAX);
			left_img.convertTo(left_img, CV_8U);
			right_img.convertTo(right_img, CV_8U);
			break;
		default:
			std::cerr << "unsupported image type." << std::endl;
			return -1;
		}

		// calculate dispaliry
		cv::Mat disparity;
		ssgbm->compute(left_img, right_img, disparity);
		disparity.convertTo(disparity, CV_32F, 1. / cv::StereoSGBM::DISP_SCALE);

		// calculate stixels
		std::vector<Stixel> stixels;

		const auto t1 = std::chrono::system_clock::now();

		//stixelWorld.compute(disparity, stixels);
		t_stixelWorld.compute(disparity, stixels);

		const auto t2 = std::chrono::system_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "stixel computation time: " << 1e-3 * duration << "[msec]" << std::endl;

		// draw stixels
		cv::Mat draw;
		cv::cvtColor(left_img, draw, cv::COLOR_GRAY2BGRA);
		cv::imshow("Original", left_img);

		cv::Mat stixelImg = cv::Mat::zeros(left_img.size(), draw.type());
		for (const auto& stixel : stixels)
			drawStixel(stixelImg, stixel, dispToColor(stixel.disp, (float)numDisparities));

		draw = draw + 0.5 * stixelImg;

		cv::imshow("disparity", disparity / numDisparities);
		cv::imshow("stixels", draw);
		cv::imwrite("./stixels.jpg", draw);
		//cv::Mat trans;
		//cv::rotate(draw, draw, 0);
		//cv::flip(draw, trans, 1);
		////cv::imshow("transstixels", trans);
		//cv::imwrite("./transstixles.jpg", trans);

		t_stixelWorld.destroy();
		

		const char c = cv::waitKey(1);
		if (c == 27)
			break;
		if (c == 'p')
			cv::waitKey(0);
	}
}