#include "gpu.h"

#include <stdio.h>


inline static unsigned divup(unsigned n, unsigned div)
{
	return (n + div - 1) / div;
}

void gpuStixelWorld::compute(const cv::Mat & disparity, std::vector<Stixel>& stixels)
{
	CV_Assert(disparity.type() == CV_32F);

	m_rows = disparity.rows;
	m_cols = disparity.cols;

	const int stixelWidth = param_.stixelWidth;
	const int w = m_cols / stixelWidth;
	const int h = m_rows;
	const int fnmax = static_cast<int>(param_.dmax);
	
	float *d_disparity_colReduced = nullptr;
	float *d_disparity_columns = nullptr;

	cudaMalloc((void **)&d_disparity_original, m_rows * m_cols * sizeof(float));
	cudaMalloc((void **)&d_disparity_colReduced, h * w * sizeof(float));
	cudaMalloc((void **)&d_disparity_columns, h * w * sizeof(float));

	float *h_disparity = nullptr;
	if (disparity.isContinuous()) {
		h_disparity = (float*)disparity.data;
	}
	else {
		std::cout << "disparity not continuous\n";
		exit(1);
	}

	cudaMemcpy(d_disparity_original, h_disparity, m_rows * m_cols * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(32, 32, 1);
	int r = divup(h, 32); int c = divup(w, 32);
	dim3 dimGrid(divup(h, dimBlock.x), divup(w, dimBlock.y), 1);

	//columnReductionMean << <dimGrid, dimBlock >> > (d_disparity_original, d_disparity_colReduced, stixelWidth, m_rows, m_cols, w);

	dim3 dd(c, r, 1);
	//transposeDisparity <<< dd, dimBlock >>> (d_disparity_colReduced, d_disparity_columns, h, w);

	columnReduction << <dd, dimBlock >> > (d_disparity_original, d_disparity_colReduced, stixelWidth, m_rows, m_cols, w);

	float *data = new float[h * w];
	cudaMemcpy(data, d_disparity_colReduced, h * w * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(data, d_disparity_columns, h * w * sizeof(float), cudaMemcpyDeviceToHost);
	cv::Mat columns = cv::Mat(w, h, CV_32F, data);

	/* for debug */
	//float *tmp_colums = new float[h * w];
	//cudaMemcpy(tmp_colums, d_disparity_colReduced, h * w * sizeof(float), cudaMemcpyDeviceToHost);
	////cudaMemcpy(tmp_colums, d_disparity_columns, h * w * sizeof(float), cudaMemcpyDeviceToHost);


	//cv::Mat mmat(w, h, cv::DataType<float>::type);
	//for (int v = 0; v < h; v++)
	//{
	//	for (int u = 0; u < w; u++)
	//	{
	//		// compute horizontal median
	//		float sum = 0.0f;
	//		for (int du = 0; du < stixelWidth; du++) {
	//			sum += disparity.at<float>(v, u * stixelWidth + du);
	//		}
	//		float m = sum / stixelWidth;
	//		mmat.at<float>(u, h - 1 - v) = m;
	//	}
	//}


	//float *matdata = (float*)mmat.data;
	//for (int k = 0; k < w * h; k++) {
	//	/*if (k > 10000)*/
	//		std::cout << k << ": " << tmp_colums[k] << " " << matdata[k] << "|";
	//}
	int k = 0;

	// get camera parameters
	const CameraParameters& camera = param_.camera;
	const float sinTilt = sinf(camera.tilt);
	const float cosTilt = cosf(camera.tilt);

	// compute expected ground disparity
	std::vector<float> groundDisparity(h);
	for (int v = 0; v < h; v++)
		groundDisparity[h - 1 - v] = std::max((camera.baseline / camera.height) * (camera.fu * sinTilt + (v - camera.v0) * cosTilt), 0.f);
	const float vhor = h - 1 - (camera.v0 * cosTilt - camera.fu * sinTilt) / cosTilt;

	NegativeLogDataTermGrd dataTermG(param_.dmax, param_.dmin, param_.sigmaG, param_.pOutG, param_.pInvG, camera,
		groundDisparity, vhor, param_.sigmaH, param_.sigmaA);
	NegativeLogDataTermObj dataTermO(param_.dmax, param_.dmin, param_.sigmaO, param_.pOutO, param_.pInvO, camera, param_.deltaz);
	NegativeLogDataTermSky dataTermS(param_.dmax, param_.dmin, param_.sigmaS, param_.pOutS, param_.pInvS);

	const int G = NegativeLogPriorTerm::G;
	const int O = NegativeLogPriorTerm::O;
	const int S = NegativeLogPriorTerm::S;
	NegativeLogPriorTerm priorTerm(h, vhor, param_.dmax, param_.dmin, camera.baseline, camera.fu, param_.deltaz,
		param_.eps, param_.pOrd, param_.pGrav, param_.pBlg, groundDisparity);

	// data cost LUT
	Matrixf costsG(w, h), costsO(w, h, fnmax), costsS(w, h), sum(w, h);
	Matrixi valid(w, h);

	// cost table
	Matrixf costTable(w, h, 3), dispTable(w, h, 3);
	Matrix<cv::Point> indexTable(w, h, 3);

	// process each column
	int u;
	//#pragma omp parallel for
	for (u = 0; u < w; u++)
	{
		//////////////////////////////////////////////////////////////////////////////
		// pre-computate LUT
		//////////////////////////////////////////////////////////////////////////////
		float tmpSumG = 0.f;
		float tmpSumS = 0.f;
		std::vector<float> tmpSumO(fnmax, 0.f);

		float tmpSum = 0.f;
		int tmpValid = 0;

		for (int v = 0; v < h; v++)
		{
			// measured disparity
			const float d = columns.at<float>(u, v);
			//const float d = columns(u, v);

			// pre-computation for ground costs
			tmpSumG += dataTermG(d, v);
			costsG(u, v) = tmpSumG;

			// pre-computation for sky costs
			tmpSumS += dataTermS(d);
			costsS(u, v) = tmpSumS;

			// pre-computation for object costs
			for (int fn = 0; fn < fnmax; fn++)
			{
				tmpSumO[fn] += dataTermO(d, fn);
				costsO(u, v, fn) = tmpSumO[fn];
			}

			// pre-computation for mean disparity of stixel
			if (d >= 0.f)
			{
				tmpSum += d;
				tmpValid++;
			}
			sum(u, v) = tmpSum;
			valid(u, v) = tmpValid;
		}

		//////////////////////////////////////////////////////////////////////////////
		// compute cost tables
		//////////////////////////////////////////////////////////////////////////////
		for (int vT = 0; vT < h; vT++)
		{
			float minCostG, minCostO, minCostS;
			float minDispG, minDispO, minDispS;
			cv::Point minPosG(G, 0), minPosO(O, 0), minPosS(S, 0);

			// process vB = 0
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = sum(u, vT) / std::max(valid(u, vT), 1);
				const int fn = cvRound(d1);

				// initialize minimum costs
				minCostG = costsG(u, vT) + priorTerm.getG0(vT);
				minCostO = costsO(u, vT, fn) + priorTerm.getO0(vT);
				minCostS = costsS(u, vT) + priorTerm.getS0(vT);
				minDispG = minDispO = minDispS = d1;
			}

			for (int vB = 1; vB <= vT; vB++)
			{
				// compute mean disparity within the range of vB to vT
				const float d1 = (sum(u, vT) - sum(u, vB - 1)) / std::max(valid(u, vT) - valid(u, vB - 1), 1);
				const int fn = cvRound(d1);

				// compute data terms costs
				const float dataCostG = vT < vhor ? costsG(u, vT) - costsG(u, vB - 1) : N_LOG_0_0;
				const float dataCostO = costsO(u, vT, fn) - costsO(u, vB - 1, fn);
				const float dataCostS = vT < vhor ? N_LOG_0_0 : costsS(u, vT) - costsS(u, vB - 1);

				// compute priors costs and update costs
				const float d2 = dispTable(u, vB - 1, 1);

#define UPDATE_COST(C1, C2) \
				const float cost##C1##C2 = dataCost##C1 + priorTerm.get##C1##C2(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, C2); \
				if (cost##C1##C2 < minCost##C1) \
				{ \
					minCost##C1 = cost##C1##C2; \
					minDisp##C1 = d1; \
					minPos##C1 = cv::Point(C2, vB - 1); \
				} \

				UPDATE_COST(G, G);
				UPDATE_COST(G, O);
				UPDATE_COST(G, S);
				UPDATE_COST(O, G);
				UPDATE_COST(O, O);
				UPDATE_COST(O, S);
				UPDATE_COST(S, G);
				UPDATE_COST(S, O);
				UPDATE_COST(S, S);
			}

			costTable(u, vT, G) = minCostG;
			costTable(u, vT, O) = minCostO;
			costTable(u, vT, S) = minCostS;

			dispTable(u, vT, G) = minDispG;
			dispTable(u, vT, O) = minDispO;
			dispTable(u, vT, S) = minDispS;

			indexTable(u, vT, G) = minPosG;
			indexTable(u, vT, O) = minPosO;
			indexTable(u, vT, S) = minPosS;
		}
	}

	//////////////////////////////////////////////////////////////////////////////
	// backtracking step
	//////////////////////////////////////////////////////////////////////////////
	for (int u = 0; u < w; u++)
	{
		float minCost = std::numeric_limits<float>::max();
		cv::Point minPos;
		for (int c = 0; c < 3; c++)
		{
			const float cost = costTable(u, h - 1, c);
			if (cost < minCost)
			{
				minCost = cost;
				minPos = cv::Point(c, h - 1);
			}
		}

		while (minPos.y > 0)
		{
			const cv::Point p1 = minPos;
			const cv::Point p2 = indexTable(u, p1.y, p1.x);
			if (p1.x == O) // object
			{
				Stixel stixel;
				stixel.u = stixelWidth * u + stixelWidth / 2;
				stixel.vT = h - 1 - p1.y;
				stixel.vB = h - 1 - (p2.y + 1);
				stixel.width = stixelWidth;
				stixel.disp = dispTable(u, p1.y, p1.x);
				stixels.push_back(stixel);
			}
			minPos = p2;
		}
	}

}

void gpuStixelWorld::destroy()
{
	cudaFree(d_disparity_original);
	cudaFree(d_disparity_colReduced);
	cudaFree(d_disparity_columns);
}
