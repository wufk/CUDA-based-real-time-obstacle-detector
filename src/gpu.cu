#include "gpu.h"

#include <stdio.h>
#include <chrono>

inline static unsigned divup(unsigned n, unsigned div)
{
	return (n + div - 1) / div;
}

struct is_less_than_zero
{
	__host__ __device__
		bool operator()(float x)
	{
		return x < 0.f;
	}
};

struct is_larger_than_zero
{
	__host__ __device__
		bool operator()(float x)
	{
		return x >= 0.f;
	}
};


void gpuStixelWorld::compute(const cv::Mat & disparity, std::vector<Stixel>& stixels)
{
	CV_Assert(disparity.type() == CV_32F);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	const int stixelWidth = param_.stixelWidth;
	const int w = m_w;
	const int h = m_h;
	const int fnmax = static_cast<int>(param_.dmax);
	

	float *h_disparity = nullptr;
	if (disparity.isContinuous()) {
		h_disparity = (float*)disparity.data;
	}
	else {
		std::cout << "disparity not continuous\n";
		exit(1);
	}

	//cudaMemcpy(h_disparity_original, h_disparity, m_rows * m_cols * sizeof(float), cudaMemcpyHostToHost);
	cudaMemcpy(d_disparity_original, h_disparity, m_rows * m_cols * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
	int r = divup(h, BLOCKSIZE); int c = divup(w, BLOCKSIZE);
	dim3 dd(divup(h, dimBlock.x), divup(w, dimBlock.y), 1);
	dim3 dimGrid(c, r, 1);

	//columnReductionMean << <dd, dimBlock >> > (d_disparity_original, d_disparity_colReduced, stixelWidth, m_rows, m_cols, w);
	//transposeDisparity <<< dimGrid, dimBlock >>> (d_disparity_colReduced, d_disparity_columns, h, w);

	//columnReduction << <dimGrid, dimBlock >> > (d_disparity_original, d_disparity_colReduced, stixelWidth, m_rows, m_cols, w);
	//cudaMemcpy(data, d_disparity_colReduced, h * w * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(data, d_disparity_columns, h * w * sizeof(float), cudaMemcpyDeviceToHost);
	//cv::Mat columns = cv::Mat(w, h, CV_32F, data);

	columnReduction << <dimGrid, dimBlock >> > (d_disparity_original, d_disparity_colReduced, stixelWidth, m_rows, m_cols, m_w);
	/* if zero cpy for colReduced, comment the next line */
	//cudaMemcpy(h_disparity_colReduced, d_disparity_colReduced, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);

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
	//int k = 0;

	// get camera parameters
	const CameraParameters& camera = param_.camera;
	const float sinTilt = sinf(camera.tilt);
	const float cosTilt = cosf(camera.tilt);

	// compute expected ground disparity
	//d_groundDisp = nullptr;
	//cudaMalloc((void**)&d_groundDisp, h * sizeof(float));
	////dim3 blocknum(div(h, 32));

	//kernComputeGroundDisp << <divup(h, 32), 32 >> > (d_groundDisp, h, 
	//	camera.baseline, camera.height, camera.fu, camera.v0, sinTilt, cosTilt);
	
	// This is oridinary without zero copy
	//float *h_groundDisp = new float[h];
	//cudaMemcpy(h_groundDisp, d_groundDisp, h * sizeof(float), cudaMemcpyDeviceToHost);
	//std::vector<float> groundDisparity(h_groundDisp, h_groundDisp + h);

	/* This is cpu */
	//std::vector<float> groundDisparity(h);
	//for (int v = 0; v < h; v++) {
	//	groundDisparity[h - 1 - v] = std::max((camera.baseline / camera.height) * (camera.fu * sinTilt + (v - camera.v0) * cosTilt), 0.f);
	//}
	//const float m_vhor = h - 1 - (camera.v0 * cosTilt - camera.fu * sinTilt) / cosTilt;

	//gpuNegativeLogDataTermGrd dataTermG(param_.dmax, param_.dmin, param_.sigmaG, param_.pOutG, param_.pInvG, camera,
	//	d_groundDisp, m_vhor, param_.sigmaH, param_.sigmaA, h);
	//gpuNegativeLogDataTermObj dataTermO(param_.dmax, param_.dmin, param_.sigmaO, param_.pOutO, param_.pInvO, camera, param_.deltaz);
	//gpuNegativeLogDataTermSky dataTermS(param_.dmax, param_.dmin, param_.sigmaS, param_.pOutS, param_.pInvS);

	//const int G = gpuNegativeLogPriorTerm::G;
	//const int O = gpuNegativeLogPriorTerm::O;
	//const int S = gpuNegativeLogPriorTerm::S;
	/*gpuNegativeLogPriorTerm priorTerm(h, m_vhor, param_.dmax, param_.dmin, camera.baseline, camera.fu, param_.deltaz,
		param_.eps, param_.pOrd, param_.pGrav, param_.pBlg, h_groundDisp);*/

	m_dataTermG.computeCostsG(d_disparity_colReduced);
	m_dataTermS.computeCostsS(d_disparity_colReduced);
	m_dataTermO.computeCostsO(d_disparity_colReduced);

	cudaMemcpy(d_sum, d_disparity_colReduced, m_w * m_h * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_valid, d_disparity_colReduced, m_w * m_h * sizeof(float), cudaMemcpyDeviceToDevice);
	is_less_than_zero less;
	is_larger_than_zero large;
	thrust::device_ptr<float> thrust_d_sum(d_sum);
	thrust::device_ptr<float> thrust_d_valid(d_valid);
	thrust::replace_if(thrust::device, thrust_d_sum, thrust_d_sum + m_w * m_h, less, 0);
	thrust::replace_if(thrust::device, thrust_d_valid, thrust_d_valid + m_w * m_h, large, 1);
	thrust::replace_if(thrust::device, thrust_d_valid, thrust_d_valid + m_w * m_h, less, 0);

	dimGrid = dim3(divup(m_w, BLOCKSIZE));
	kernScanCosts << <dimGrid, BLOCKSIZE >> > (m_w, m_h, d_sum);
	kernScanCosts << <dimGrid, BLOCKSIZE >> > (m_w, m_h, d_valid);
	//cudaMemcpy(h_sum, d_sum, m_w * m_h * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_valid, d_valid, m_w * m_h * sizeof(float), cudaMemcpyDeviceToHost);

	/* This is test
	//for (int u = 0; u < w; u++) {
	//	float tmpSumG = 0.f;
	//	std::vector<float> tmpSumO(fnmax, 0.f);
	//	float tmpSum = 0.f;
	//	int tmpValid = 0;

	//	for (int v = 0; v < h; v++) {
	//		const float d = h_disparity_colReduced[u * h + v];
	//		if (d >= 0.f) {
	//			tmpSum += d;
	//			tmpValid++;
	//		}
	//		int n = u * m_h + v;
	//		//if (tmpValid - h_valid[n] > 0.01 || h_valid[n] - tmpValid > 0.01) {
	//		//	std::cout <<u << "," << v <<":"<< tmpValid << " " << h_valid[n] << std::endl;
	//		//}

	//		//if (tmpSum - h_sum[n] > 0.01 || h_sum[n] - tmpSum > 0.01)
	//		//	std::cout << tmpSum << "  " << h_sum[n] << std::endl;

	//		//for (int f = 0; f < fnmax; f++) {
	//		//	float incre = m_dataTermO(d, f);
	//		//	tmpSumO[f] += incre;
	//		//	// below is u * fnamx * v. which is correct 
	//		//	//int index = u * h * fnmax + f * h + v;
	//		//	//if (m_dataTermO.h_costsO[index] - incre > 0.1 || m_dataTermO.h_costsO[index] - incre < -0.1)
	//		//	//	std::cout << index << ": " << incre << " " << m_dataTermO.h_costsO[index] << std::endl;
	//		//	//if (m_dataTermO.h_costsO[index] - tmpSumO[f] > 0.1 || m_dataTermO.h_costsO[index] - tmpSumO[f] < -0.1)
	//		//	//if (m_dataTermO.h_costsO[index] > 0 && tmpSumO[f] > 0)
	//		//	//	std::cout << index << ": " << tmpSumO[f] << " " << m_dataTermO.h_costsO[index] << std::endl;

	//		//	// below is u * v * fnmax. which is not correct
	//		//	//int index = u * h * fnmax + v * fnmax + f;
	//		//	//if (m_dataTermO.h_costsO[index] - incre > 0.1)
	//		//	//	std::cout << index << ": " << incre << " " << m_dataTermO.h_costsO[index] << std::endl;
	//		//	//if (m_dataTermO.h_costsO[index] - tmpSumO[f] > 0.1)
	//		//	//	std::cout << u <<","<<v<<","<<fnmax << ": " << tmpSumO[f] << " " << m_dataTermO.h_costsO[index] << std::endl;
	//		//	//float incre = m_dataTermS(d);
	//		//	//tmpSumG += incre;
	//		//	//std::cout << tmpSumG << " " << m_dataTermS.h_costsS[u * h + v] << std::endl;
	//		//}
	//	}
	//}
	*/

	/*KernDP << <m_w, m_h >> > (m_w, m_h, fnmax, d_disparity_colReduced, d_sum, d_valid,
		m_dataTermG.d_costsG, m_dataTermO.d_costsO, m_dataTermS.d_costsS, 
		d_costTableG, d_costTableO, d_costTableS, d_dispTableG, d_dispTableO, d_dispTableS, d_indexTableG, d_indexTableO, d_indexTableS,
		m_priorTerm.d_costs0_, m_priorTerm.d_costs1_, m_priorTerm.d_costs2_O_G_, m_priorTerm.d_costs2_O_O_, m_priorTerm.d_costs2_O_S_, m_priorTerm.d_costs2_S_O_,
		N_LOG_0_0, m_vhor
	);*/

	int smem_size = 10 * m_h * sizeof(float);
	KernDP << <m_w, m_h, smem_size >> > (m_w, m_h, fnmax, d_disparity_colReduced, d_sum, d_valid,
		m_dataTermG.d_costsG, m_dataTermO.d_costsO, m_dataTermS.d_costsS,
		d_costTableG, d_costTableO, d_costTableS, d_dispTableG, d_dispTableO, d_dispTableS, d_indexTableG, d_indexTableO, d_indexTableS,
		m_priorTerm.d_costs0_, m_priorTerm.d_costs1_, m_priorTerm.d_costs2_O_G_, m_priorTerm.d_costs2_O_O_, m_priorTerm.d_costs2_O_S_, m_priorTerm.d_costs2_S_O_,
		N_LOG_0_0, m_vhor
	);

	cudaMemcpy(h_costTableG, d_costTableG, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_costTableS, d_costTableS, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_costTableO, d_costTableO, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_dispTableG, d_dispTableG, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dispTableO, d_dispTableO, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dispTableS, d_dispTableS, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_indexTableG, d_indexTableG, m_h * m_w * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_indexTableO, d_indexTableO, m_h * m_w * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_indexTableS, d_indexTableS, m_h * m_w * sizeof(glm::vec2), cudaMemcpyDeviceToHost);

	for (int u = 0; u < w; u++)
	{
		float minCost = std::numeric_limits<float>::max();
		cv::Point minPos;
		float cost = h_costTableG[u * m_h + h - 1];
		if (cost < minCost) {
			minCost = cost;
			minPos = cv::Point(0, h - 1);
		}
		cost = h_costTableO[u * m_h + h - 1];
		if (cost < minCost) {
			minCost = cost;
			minPos = cv::Point(1, h - 1);
		}
		cost = h_costTableS[u * m_h + h - 1];
		if (cost < minCost) {
			minCost = cost;
			minPos = cv::Point(2, h - 1);
		}
		while (minPos.y > 0)
		{
			const cv::Point p1 = minPos;
			cv::Point p2;
			switch (p1.x) {
			case 0:
				p2 = cv::Point(h_indexTableG[u * m_h + p1.y].x, h_indexTableG[u * m_h + p1.y].y);
				break;
			case 1:
				p2 = cv::Point(h_indexTableO[u * m_h + p1.y].x, h_indexTableO[u * m_h + p1.y].y);
				break;
			case 2:
				p2 = cv::Point(h_indexTableS[u * m_h + p1.y].x, h_indexTableS[u * m_h + p1.y].y);
				break;
			}
			if (p1.x == 1) // object
			{
				Stixel stixel;
				stixel.u = stixelWidth * u + stixelWidth / 2;
				stixel.vT = h - 1 - p1.y;
				stixel.vB = h - 1 - (p2.y + 1);
				stixel.width = stixelWidth;
				//stixel.disp = dispTable(u, p1.y, p1.x);
				stixel.disp = h_dispTableG[u * m_h + p1.y];
				stixels.push_back(stixel);
			}
			minPos = p2;
		}
	}
	
	// data cost LUT
	Matrixf costsG(w, h), costsO(w, h, fnmax), costsS(w, h), sum(w, h);
	Matrixi valid(w, h);

	// cost table
	Matrixf costTable(w, h, 3), dispTable(w, h, 3);
	Matrix<cv::Point> indexTable(w, h, 3);

	// process each column
//	int u;
//
//	for (u = 0; u < w; u++)
//	{
//		//////////////////////////////////////////////////////////////////////////////
//		// pre-computate LUT
//		//////////////////////////////////////////////////////////////////////////////
//		float tmpSumG = 0.f;
//		float tmpSumS = 0.f;
//		std::vector<float> tmpSumO(fnmax, 0.f);
//
//		float tmpSum = 0.f;
//		int tmpValid = 0;
//
//		for (int v = 0; v < h; v++)
//		{
//			// measured disparity
//			//const float d = columns.at<float>(u, v);
//			//const float d = columns(u, v);
//			//const float d = data[u * h + v];
//			const float d = h_disparity_colReduced[u * h + v];
//
//			// pre-computation for ground costs
//			tmpSumG += m_dataTermG(d, v);
//			costsG(u, v) = tmpSumG;
//
//			// pre-computation for sky costs
//			tmpSumS += m_dataTermS(d);
//			costsS(u, v) = tmpSumS;
//
//			// pre-computation for object costs
//			for (int fn = 0; fn < fnmax; fn++)
//			{
//				tmpSumO[fn] += m_dataTermO(d, fn);
//				costsO(u, v, fn) = tmpSumO[fn];
//			}
//
//			// pre-computation for mean disparity of stixel
//			if (d >= 0.f)
//			{
//				tmpSum += d;
//				tmpValid++;
//			}
//			sum(u, v) = tmpSum;
//			valid(u, v) = tmpValid;
//		}
//
//		//////////////////////////////////////////////////////////////////////////////
//		// compute cost tables
//		//////////////////////////////////////////////////////////////////////////////
//		for (int vT = 0; vT < h; vT++)
//		{
//			float minCostG, minCostO, minCostS;
//			float minDispG, minDispO, minDispS;
//			cv::Point minPosG(G, 0), minPosO(O, 0), minPosS(S, 0);
//
//			// process vB = 0
//			{
//				// compute mean disparity within the range of vB to vT
//				const float d1 = sum(u, vT) / std::max(valid(u, vT), 1);
//				const int fn = cvRound(d1);
//
//				// initialize minimum costs
//				minCostG = costsG(u, vT) + m_priorTerm.getG0(vT);
//				minCostO = costsO(u, vT, fn) + m_priorTerm.getO0(vT);
//				minCostS = costsS(u, vT) + m_priorTerm.getS0(vT);
//				minDispG = minDispO = minDispS = d1;
//			}
//
//			for (int vB = 1; vB <= vT; vB++)
//			{
//				// compute mean disparity within the range of vB to vT
//				const float d1 = (sum(u, vT) - sum(u, vB - 1)) / std::max(valid(u, vT) - valid(u, vB - 1), 1);
//				const int fn = cvRound(d1);
//
//				// compute data terms costs
//				const float dataCostG = vT < m_vhor ? costsG(u, vT) - costsG(u, vB - 1) : N_LOG_0_0;
//				const float dataCostO = costsO(u, vT, fn) - costsO(u, vB - 1, fn);
//				const float dataCostS = vT < m_vhor ? N_LOG_0_0 : costsS(u, vT) - costsS(u, vB - 1);
//
//				// compute priors costs and update costs
//				const float d2 = dispTable(u, vB - 1, O);
//				dataCostG + m_priorTerm.getGG(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, G);
//
//				float ct;
//				//GG
//				ct = dataCostG + m_priorTerm.getGG(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, G);
//				if (ct < minCostG) {
//					minCostG = ct;
//					minDispG = d1;
//					minPosG = cv::Point(G, vB - 1);
//				}
//				//GO
//				ct = dataCostG + m_priorTerm.getGO(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, O);
//				if (ct < minCostG) {
//					minCostG = ct;
//					minDispG = d1;
//					minPosG = cv::Point(O, vB - 1);
//				}
//				//GS
//				ct = dataCostG + m_priorTerm.getGS(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, S);
//				if (ct < minCostG) {
//					minCostG = ct;
//					minDispG = d1;
//					minPosG = cv::Point(S, vB - 1);
//				}
//				//OG
//				ct = dataCostO + m_priorTerm.getOG(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, G);
//				if (ct < minCostO) {
//					minCostO = ct;
//					minDispO = d1;
//					minPosO = cv::Point(G, vB - 1);
//				}
//				//OO
//				ct = dataCostO + m_priorTerm.getOO(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, O);
//				if (ct < minCostO) {
//					minCostO = ct;
//					minDispO = d1;
//					minPosO = cv::Point(O, vB - 1);
//				}
//				//OS
//				ct = dataCostO + m_priorTerm.getOS(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, S);
//				if (ct < minCostO) {
//					minCostO = ct;
//					minDispO = d1;
//					minPosO = cv::Point(S, vB - 1);
//				}
//				//SG
//				ct = dataCostS + m_priorTerm.getSG(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, G);
//				if (ct < minCostS) {
//					minCostS = ct;
//					minDispS = d1;
//					minPosS = cv::Point(G, vB - 1);
//				}
//				//SO
//				ct = dataCostS + m_priorTerm.getSO(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, O);
//				if (ct < minCostS) {
//					minCostS = ct;
//					minDispS = d1;
//					minPosS = cv::Point(O, vB - 1);
//				}
//				ct = dataCostS + m_priorTerm.getSS(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, S);
//				if (ct < minCostS) {
//					minCostS = ct;
//					minDispS = d1;
//					minPosS = cv::Point(S, vB - 1);
//				}
////#define UPDATE_COST(C1, C2) \
////				const float cost##C1##C2 = dataCost##C1 + m_priorTerm.get##C1##C2(vB, cvRound(d1), cvRound(d2)) + costTable(u, vB - 1, C2); \
////				if (cost##C1##C2 < minCost##C1) \
////				{ \
////					minCost##C1 = cost##C1##C2; \
////					minDisp##C1 = d1; \
////					minPos##C1 = cv::Point(C2, vB - 1); \
////				} \
//				
//				//UPDATE_COST(G, G);
//				//UPDATE_COST(G, O);
//				//UPDATE_COST(G, S);
//				//UPDATE_COST(O, G);
//				//UPDATE_COST(O, O);
//				//UPDATE_COST(O, S);
//				//UPDATE_COST(S, G);
//				//UPDATE_COST(S, O);
//				//UPDATE_COST(S, S);
//				
//			}
//
//			costTable(u, vT, G) = minCostG;
//			costTable(u, vT, O) = minCostO;
//			costTable(u, vT, S) = minCostS;
//
//			dispTable(u, vT, G) = minDispG;
//			dispTable(u, vT, O) = minDispO;
//			dispTable(u, vT, S) = minDispS;
//
//			indexTable(u, vT, G) = minPosG;
//			indexTable(u, vT, O) = minPosO;
//			indexTable(u, vT, S) = minPosS;
//		}
//	}
//
//	for (int u = 0; u < m_w; u++) {
//		for (int v = 0; v < m_h; v++) {
//			/*if (h_costTableG[u * m_h + v] != costTable(u, v, G))
//				std::cout << "**" << u << "," << v << ":" << h_costTableG[u * m_h + v] << " " << costTable(u, v, G);*/
//			if (h_costTableO[u * m_h + v] != costTable(u, v, O))
//				std::cout << "**" << u << "," << v << ":" << h_costTableO[u * m_h + v] << " " << costTable(u, v, O);
//		}
//	}
//
////
////	//////////////////////////////////////////////////////////////////////////////
////	// backtracking step
////	//////////////////////////////////////////////////////////////////////////////
//	for (int u = 0; u < w; u++)
//	{
//		float minCost = std::numeric_limits<float>::max();
//		cv::Point minPos;
//		for (int c = 0; c < 3; c++)
//		{
//			const float cost = costTable(u, h - 1, c);
//			if (cost < minCost)
//			{
//				minCost = cost;
//				minPos = cv::Point(c, h - 1);
//			}
//		}
//
//		while (minPos.y > 0)
//		{
//			const cv::Point p1 = minPos;
//			const cv::Point p2 = indexTable(u, p1.y, p1.x);
//			if (p1.x == O) // object
//			{
//				Stixel stixel;
//				stixel.u = stixelWidth * u + stixelWidth / 2;
//				stixel.vT = h - 1 - p1.y;
//				stixel.vB = h - 1 - (p2.y + 1);
//				stixel.width = stixelWidth;
//				stixel.disp = dispTable(u, p1.y, p1.x);
//				stixels.push_back(stixel);
//			}
//			minPos = p2;
//		}
//	}

}

void gpuStixelWorld::preprocess(const CameraParameters & camera, float sinTilt, float cosTilt)
{
	const auto t1 = std::chrono::system_clock::now();
	// THis is ordinary methods
	//d_groundDisp = nullptr;
	//cudaMalloc((void**)&d_groundDisp, m_h * sizeof(float));

	// This is zero copy
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc((void**)&h_groundDisp, m_h * sizeof(float), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&d_groundDisp, (void *)h_groundDisp, 0);

	kernComputeGroundDisp << <divup(m_h, BLOCKSIZE), BLOCKSIZE >> > (d_groundDisp, m_h,
		camera.baseline, camera.height, camera.fu, camera.v0, sinTilt, cosTilt);

	m_dataTermG = gpuNegativeLogDataTermGrd(param_.dmax, param_.dmin, param_.sigmaG, param_.pOutG, param_.pInvG, camera,
		d_groundDisp, m_vhor, param_.sigmaH, param_.sigmaA, m_h, m_w);
	m_dataTermO = gpuNegativeLogDataTermObj(param_.dmax, param_.dmin, param_.sigmaO, param_.pOutO, param_.pInvO, camera, param_.deltaz, m_w, m_h);
	m_dataTermS = gpuNegativeLogDataTermSky(param_.dmax, param_.dmin, param_.sigmaS, param_.pOutS, param_.pInvS, m_w, m_h);
	m_priorTerm = gpuNegativeLogPriorTerm(m_h, m_vhor, param_.dmax, param_.dmin, camera.baseline, camera.fu, param_.deltaz,
		param_.eps, param_.pOrd, param_.pGrav, param_.pBlg, h_groundDisp);

	const auto t2 = std::chrono::system_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	std::cout << "preprocess time: " << 1e-3 * duration << "[msec]" << std::endl;
}

void gpuStixelWorld::destroy()
{
	cudaFree(d_disparity_original);
	cudaFree(h_disparity_colReduced);
	cudaFree(h_groundDisp);
	//cudaFree(d_disparity_columns);
	cudaFree(d_groundDisp);
	cudaFree(d_sum);
	cudaFree(d_valid);

	cudaFree(d_costTableG);
	cudaFree(d_costTableO);
	cudaFree(d_costTableS);

	cudaFree(d_dispTableG);
	cudaFree(d_dispTableS);
	cudaFree(d_dispTableO);

	cudaFree(d_indexTableG);
	cudaFree(d_indexTableO);
	cudaFree(d_indexTableS);

	cudaFree(h_sum);
	cudaFree(h_valid);
	delete[] data;

	//cudaFree(d_disparity_colReduced);

}

void gpuNegativeLogDataTermGrd::init(float dmax, float dmin, float sigmaD, 
	float pOut, float pInv, const CameraParameters & camera, 
	float* d_groundDisparity, float m_vhor, float sigmaH, float sigmaA, int h)
{
	// uniform distribution term
	nLogPUniform_ = logf(dmax - dmin) - logf(pOut);
	const float cf = camera.fu * camera.baseline / camera.height;

	h_nLogPGaussian_ = new float[h];
	h_fn_ = new float[h];
	h_cquad_ = new float[h];
	h_costsG = new float[m_w * m_h];
	cudaMalloc((void**)&d_cquad_, h * sizeof(float));
	cudaMalloc((void**)&d_fn_, h * sizeof(float));
	cudaMalloc((void**)&d_nLogPGaussian_, h * sizeof(float));
	cudaMalloc((void**)&d_costsG, m_w * m_h * sizeof(float));

	/* this is for zero copy */
	//cudaSetDeviceFlags(cudaDeviceMapHost);
	//cudaHostAlloc((void**)&h_nLogPGaussian_, h * sizeof(float), cudaHostAllocMapped);
	//cudaHostAlloc((void**)&h_cquad_, h * sizeof(float), cudaHostAllocMapped);
	//cudaHostAlloc((void**)&h_fn_, h * sizeof(float), cudaHostAllocMapped);
	//cudaHostGetDevicePointer((void**)&d_nLogPGaussian_, (void *)h_nLogPGaussian_, 0);
	//cudaHostGetDevicePointer((void**)&d_cquad_, (void *)h_cquad_, 0);
	//cudaHostGetDevicePointer((void**)&d_fn_, (void *)h_fn_, 0);
	//cudaHostAlloc((void**)&h_costsG, m_h * m_w * sizeof(float), cudaHostAllocMapped);
	//cudaHostGetDevicePointer((void**)&d_costsG, (void *)h_costsG, 0);


	dim3 dimBlock(divup(h, BLOCKSIZE));
	kernComputeNegativeLogDataTermGrd << <dimBlock,BLOCKSIZE >> > (h, d_groundDisparity, d_nLogPGaussian_, d_fn_, d_cquad_,
		camera.fv, camera.tilt, camera.height, cf, sigmaA, sigmaH, sigmaD, dmax, dmin, SQRT2, PI, pOut, m_vhor);

	cudaMemcpy(h_nLogPGaussian_, d_nLogPGaussian_, h * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fn_, d_fn_, h * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cquad_, d_cquad_, h * sizeof(float), cudaMemcpyDeviceToHost);
}

void gpuNegativeLogDataTermGrd::computeCostsG(float* d_disp_colReduced)
{	
	int r = divup(m_w, BLOCKSIZE); int c = divup(m_h, BLOCKSIZE);
	dim3 dimGrid(r, c, 1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
	kernComputeCostsG << <dimGrid, dimBlock >> > (m_w, m_h, nLogPUniform_, d_costsG, d_nLogPGaussian_, d_cquad_, d_fn_, d_disp_colReduced);
	dimGrid = dim3(divup(m_w, BLOCKSIZE));
	kernScanCosts << <dimGrid, BLOCKSIZE >> > (m_w, m_h, d_costsG);

	cudaMemcpy(h_costsG, d_costsG, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
}

void gpuNegativeLogDataTermGrd::destroy()
{
	//cudaFree(h_costsG);
	//cudaFree(h_nLogPGaussian_);
	//cudaFree(h_cquad_);
	//cudaFree(h_fn_);


	cudaFree(d_nLogPGaussian_);
	cudaFree(d_cquad_);
	cudaFree(d_fn_);
	cudaFree(d_costsG);
	delete[] h_nLogPGaussian_;
	delete[] h_fn_;
	delete[] h_cquad_;
	delete[] h_costsG;
}


void gpuNegativeLogDataTermObj::computeCostsO(float * d_disp_colReduced)
{
	int r = divup(m_w, 8); int c = divup(m_h, 8); int t = divup(fnmax, 8);
	//dim3 dimGrid(r, c, t);
	//dim3 dimBlock(8, 8, 8);
	//kernComputeCostsO << <dimGrid, dimBlock >> > (m_w, m_h, fnmax, d_costsO, d_disp_colReduced, d_nLogPGaussian_, d_cquad_, nLogPUniform_);
	//dimGrid = dim3(divup(m_w, BLOCKSIZE), divup(m_h, BLOCKSIZE), 1);
	//dimBlock = dim3(BLOCKSIZE, BLOCKSIZE);
	//kernScanCostsObj << <dimGrid, dimBlock >> > (m_w, m_h, fnmax, d_costsO);

	dim3 dimGrid(r, t, c);
	dim3 dimBlock(8, 8, 8);
	kernComputeCostsOO << <dimGrid, dimBlock >> > (m_w, fnmax, m_h, d_costsO, d_disp_colReduced, d_nLogPGaussian_, d_cquad_, nLogPUniform_);
	dimGrid = dim3(divup(m_w, BLOCKSIZE), divup(fnmax, BLOCKSIZE), 1);
	dimBlock = dim3(BLOCKSIZE, BLOCKSIZE);
	kernScanCostsObj << <dimGrid, dimBlock >> > (m_w, fnmax, m_h, d_costsO);

	cudaMemcpy(h_costsO, d_costsO, m_h * m_w * fnmax * sizeof(float), cudaMemcpyDeviceToHost);}

void gpuNegativeLogDataTermObj::destroy()
{
	cudaFree(d_nLogPGaussian_);
	cudaFree(d_cquad_);
	cudaFree(d_costsO);
	delete[] h_nLogPGaussian_;
	delete[] h_cquad_;

	//cudaFree(h_nLogPGaussian_);
	//cudaFree(h_cquad_);
}

void gpuNegativeLogDataTermObj::init(float dmax, float dmin, float sigmaD, float pOut, float pInv, const CameraParameters & camera, float deltaz)
{
	// uniform distribution term
	nLogPUniform_ = logf(dmax - dmin) - logf(pOut);

	h_nLogPGaussian_ = new float[fnmax];
	h_cquad_ = new float[fnmax];
	h_costsO = new float[m_w * m_h * fnmax];
	cudaMalloc((void**)&d_cquad_, fnmax * sizeof(float));
	cudaMalloc((void**)&d_nLogPGaussian_, fnmax * sizeof(float));
	cudaMalloc((void**)&d_costsO, m_w * m_h * fnmax * sizeof(float));

	/* this is for zero copy */
	//cudaSetDeviceFlags(cudaDeviceMapHost);
	//cudaHostAlloc((void**)&h_nLogPGaussian_, fnmax * sizeof(float), cudaHostAllocMapped);
	//cudaHostAlloc((void**)&h_cquad_, fnmax * sizeof(float), cudaHostAllocMapped);
	//cudaHostGetDevicePointer((void**)&d_nLogPGaussian_, (void *)h_nLogPGaussian_, 0);
	//cudaHostGetDevicePointer((void**)&d_cquad_, (void *)h_cquad_, 0);

	dim3 dimBlock(divup(fnmax, BLOCKSIZE));
	kernComputeNegativeLogDataTermObj << <dimBlock, BLOCKSIZE >> > (fnmax, d_cquad_, d_nLogPGaussian_,
		camera.fu, camera.baseline, sigmaD, deltaz, SQRT2, PI, pOut, dmin, dmax);

	cudaMemcpy(h_nLogPGaussian_, d_nLogPGaussian_, fnmax * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cquad_, d_cquad_, fnmax * sizeof(float), cudaMemcpyDeviceToHost);
}

void gpuNegativeLogDataTermSky::init(float dmax, float dmin, float sigmaD, float pOut, float pInv, float fn)
{
	// uniform distribution term
	nLogPUniform_ = logf(dmax - dmin) - logf(pOut);

	// Gaussian distribution term
	const float ANorm = 0.5f * (erff((dmax - fn) / (SQRT2 * sigmaD)) - erff((dmin - fn) / (SQRT2 * sigmaD)));
	nLogPGaussian_ = logf(ANorm) + logf(sigmaD * sqrtf(2.f * PI)) - logf(1.f - pOut);

	// coefficient of quadratic part
	cquad_ = 1.f / (2.f * sigmaD * sigmaD);

	cudaMalloc((void**)&d_costsS, m_h * m_w * sizeof(float));
	h_costsS = new float[m_h * m_w];
}

void gpuNegativeLogDataTermSky::destroy()
{
	cudaFree(d_costsS);
	delete[] h_costsS;
}

void gpuNegativeLogDataTermSky::computeCostsS(float *d_disparity_colReduced)
{
	int r = divup(m_w, BLOCKSIZE); int c = divup(m_h, BLOCKSIZE);
	dim3 dimGrid(r, c, 1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
	kernComputeCostsS << <dimGrid, dimBlock >> > (m_w, m_h, d_costsS, d_disparity_colReduced, nLogPUniform_, cquad_, nLogPGaussian_, fn_);
	dimGrid = dim3(divup(m_w, BLOCKSIZE));
	kernScanCosts << <dimGrid, BLOCKSIZE >> > (m_w, m_h, d_costsS);
	cudaMemcpy(h_costsS, d_costsS, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
}

void gpuNegativeLogPriorTerm::init(int h, float m_vhor, float dmax, float dmin, float b, float fu, float deltaz, float eps, float pOrd, float pGrav, float pBlg, float* groundDisparity)
{
	//const int fnmax = static_cast<int>(dmax);

	cudaMalloc((void**)&d_costs0_, m_h * 2 * sizeof(float));
	cudaMalloc((void**)&d_costs1_, m_h * 9 * sizeof(float));
	cudaMalloc((void**)&d_costs2_O_O_, fnmax * fnmax * sizeof(float));
	cudaMalloc((void**)&d_costs2_O_S_, 1 * fnmax * sizeof(float));
	cudaMalloc((void**)&d_costs2_O_G_, m_h * fnmax * sizeof(float));
	cudaMalloc((void**)&d_costs2_S_O_, fnmax * fnmax * sizeof(float));

	costs0_.create(h, 2);
	costs1_.create(h, 3, 3);
	costs2_O_O_.create(fnmax, fnmax);
	costs2_O_S_.create(1, fnmax);
	costs2_O_G_.create(h, fnmax);
	costs2_S_O_.create(fnmax, fnmax);

	for (int vT = 0; vT < h; vT++)
	{
		const float P1 = N_LOG_1_0;
		const float P2 = -logf(1.f / h);
		const float P3_O = vT > m_vhor ? N_LOG_1_0 : N_LOG_0_5;
		const float P3_G = vT > m_vhor ? N_LOG_0_0 : N_LOG_0_5;
		const float P4_O = -logf(1.f / (dmax - dmin));
		const float P4_G = N_LOG_1_0;

		costs0_(vT, O) = P1 + P2 + P3_O + P4_O;
		costs0_(vT, G) = P1 + P2 + P3_G + P4_G;
	}

	for (int vB = 0; vB < h; vB++)
	{
		const float P1 = N_LOG_1_0;
		const float P2 = -logf(1.f / (h - vB));

		const float P3_O_O = vB - 1 < m_vhor ? N_LOG_0_7 : N_LOG_0_5;
		const float P3_G_O = vB - 1 < m_vhor ? N_LOG_0_3 : N_LOG_0_0;
		const float P3_S_O = vB - 1 < m_vhor ? N_LOG_0_0 : N_LOG_0_5;

		const float P3_O_G = vB - 1 < m_vhor ? N_LOG_0_7 : N_LOG_0_0;
		const float P3_G_G = vB - 1 < m_vhor ? N_LOG_0_3 : N_LOG_0_0;
		const float P3_S_G = vB - 1 < m_vhor ? N_LOG_0_0 : N_LOG_0_0;

		const float P3_O_S = vB - 1 < m_vhor ? N_LOG_0_0 : N_LOG_1_0;
		const float P3_G_S = vB - 1 < m_vhor ? N_LOG_0_0 : N_LOG_0_0;
		const float P3_S_S = vB - 1 < m_vhor ? N_LOG_0_0 : N_LOG_0_0;

		costs1_(vB, O, O) = P1 + P2 + P3_O_O;
		costs1_(vB, G, O) = P1 + P2 + P3_G_O;
		costs1_(vB, S, O) = P1 + P2 + P3_S_O;

		costs1_(vB, O, G) = P1 + P2 + P3_O_G;
		costs1_(vB, G, G) = P1 + P2 + P3_G_G;
		costs1_(vB, S, G) = P1 + P2 + P3_S_G;

		costs1_(vB, O, S) = P1 + P2 + P3_O_S;
		costs1_(vB, G, S) = P1 + P2 + P3_G_S;
		costs1_(vB, S, S) = P1 + P2 + P3_S_S;
	}

	for (int d1 = 0; d1 < fnmax; d1++)
		costs2_O_O_(0, d1) = N_LOG_0_0;

	for (int d2 = 1; d2 < fnmax; d2++)
	{
		const float z = b * fu / d2;
		const float deltad = d2 - b * fu / (z + deltaz);
		for (int d1 = 0; d1 < fnmax; d1++)
		{
			if (d1 > d2 + deltad)
				costs2_O_O_(d2, d1) = -logf(pOrd / (d2 - deltad));
			else if (d1 <= d2 - deltad)
				costs2_O_O_(d2, d1) = -logf((1.f - pOrd) / (dmax - d2 - deltad));
			else
				costs2_O_O_(d2, d1) = N_LOG_0_0;
		}
	}

	for (int v = 0; v < h; v++)
	{
		const float fn = groundDisparity[v];
		for (int d1 = 0; d1 < fnmax; d1++)
		{
			if (d1 > fn + eps)
				costs2_O_G_(v, d1) = -logf(pGrav / (dmax - fn - eps));
			else if (d1 < fn - eps)
				costs2_O_G_(v, d1) = -logf(pBlg / (fn - eps - dmin));
			else
				costs2_O_G_(v, d1) = -logf((1.f - pGrav - pBlg) / (2.f * eps));
		}
	}

	for (int d1 = 0; d1 < fnmax; d1++)
	{
		costs2_O_S_(d1) = d1 > eps ? -logf(1.f / (dmax - dmin - eps)) : N_LOG_0_0;
	}

	for (int d2 = 0; d2 < fnmax; d2++)
	{
		for (int d1 = 0; d1 < fnmax; d1++)
		{
			if (d2 < eps)
				costs2_S_O_(d2, d1) = N_LOG_0_0;
			else if (d1 <= 0)
				costs2_S_O_(d2, d1) = N_LOG_1_0;
			else
				costs2_S_O_(d2, d1) = N_LOG_0_0;
		}
	}


	/*costs0_.create(h, 2);
	costs1_.create(h, 3, 3);
	costs2_O_O_.create(fnmax, fnmax);
	costs2_O_S_.create(1, fnmax);
	costs2_O_G_.create(h, fnmax);
	costs2_S_O_.create(fnmax, fnmax);*/


	cudaMemcpy(d_costs0_, costs0_.data, m_h * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_costs1_, costs1_.data, m_h * 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_costs2_O_O_, costs2_O_O_.data, fnmax * fnmax * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_costs2_O_S_, costs2_O_S_.data, fnmax * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_costs2_O_G_, costs2_O_G_.data, m_h * fnmax * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_costs2_S_O_, costs2_S_O_.data, fnmax * fnmax * sizeof(float), cudaMemcpyHostToDevice);
}

void gpuNegativeLogPriorTerm::destroy()
{
	cudaFree(d_costs0_);
	cudaFree(d_costs1_);
	cudaFree(d_costs2_O_O_);
	cudaFree(d_costs2_O_G_);
	cudaFree(d_costs2_O_S_);
	cudaFree(d_costs2_S_O_);
}
