#include "gpu.h"

#include <stdio.h>

inline static unsigned divup(unsigned n, unsigned div)
{
	return (n + div - 1) / div;
}

void gpuStixelWorld::compute(const cv::Mat & disparity, std::vector<Stixel>& stixels)
{
	CV_Assert(disparity.type() == CV_32F);

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
	cudaMemcpy(d_disparity_original, h_disparity, m_rows * m_cols * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
	int r = divup(h, BLOCKSIZE); int c = divup(w, BLOCKSIZE);
	dim3 dd(divup(h, dimBlock.x), divup(w, dimBlock.y), 1);
	dim3 dimGrid(c, r, 1);

	columnReduction << <dimGrid, dimBlock >> > (d_disparity_original, d_disparity_colReduced, stixelWidth, m_rows, m_cols, m_w);// default stream
	cudaDeviceSynchronize();
	cudaMemcpyAsync(d_valid, d_disparity_colReduced, m_w * m_h * sizeof(float), cudaMemcpyDeviceToDevice, streams[1]); 
	cudaMemcpyAsync(d_sum, d_disparity_colReduced, m_w * m_h * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]); 
	kerndValid << <dimGrid, dimBlock, 0, streams[1] >> > (m_w, m_h, d_valid);
	kerndSum << <dimGrid, dimBlock, 0, streams[0] >> > (m_w, m_h, d_sum);

	m_dataTermO.computeCostsO1(d_disparity_colReduced, &streams[4]);
	m_dataTermS.computeCostsS1(d_disparity_colReduced, &streams[3]);
	m_dataTermG.computeCostsG1(d_disparity_colReduced, &streams[2]);

	dimGrid = dim3(divup(m_w, BLOCKSIZE));
	//kernScan1shf << <m_w, 512, 512 * sizeof(float), streams[1] >> > (m_w, m_h, d_valid);
	//kernScan1shf << <m_w, 512, 512 * sizeof(float), streams[0] >> > (m_w, m_h, d_sum);
	kernScan1 << <m_w, 512, 512 * sizeof(float), streams[1] >> > (m_w, m_h, d_valid);
	kernScan1 << <m_w, 512, 512 * sizeof(float), streams[0]>> > (m_w, m_h, d_sum);
	//kernScanCosts << <dimGrid, BLOCKSIZE, 0, streams[1] >> > (m_w, m_h, d_valid);
	//kernScanCosts << <dimGrid, BLOCKSIZE, 0, streams[0] >> > (m_w, m_h, d_sum);

	m_dataTermO.computeCostsO2(&streams[4]);
	m_dataTermS.computeCostsS2(&streams[3]);
	m_dataTermG.computeCostsG2(&streams[2]);

	/*test no streaming*/
	//cudaMemcpyAsync(d_valid, d_disparity_colReduced, m_w * m_h * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
	//cudaMemcpyAsync(d_sum, d_disparity_colReduced, m_w * m_h * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
	//kerndValid << <dimGrid, dimBlock, 0, streams[0] >> > (m_w, m_h, d_valid);
	//kerndSum << <dimGrid, dimBlock, 0, streams[0] >> > (m_w, m_h, d_sum);

	//m_dataTermO.computeCostsO1(d_disparity_colReduced, &streams[0]);
	//m_dataTermS.computeCostsS1(d_disparity_colReduced, &streams[0]);
	//m_dataTermG.computeCostsG1(d_disparity_colReduced, &streams[0]);

	//dimGrid = dim3(divup(m_w, BLOCKSIZE));
	//kernScan1 << <m_w, 512, 512 * sizeof(float), streams[0] >> > (m_w, m_h, d_valid);
	//kernScan1 << <m_w, 512, 512 * sizeof(float), streams[0]>> > (m_w, m_h, d_sum);
	////kernScanCosts << <dimGrid, BLOCKSIZE, 0, streams[0] >> > (m_w, m_h, d_valid);
	////kernScanCosts << <dimGrid, BLOCKSIZE, 0, streams[0] >> > (m_w, m_h, d_sum);

	//m_dataTermO.computeCostsO2(&streams[0]);
	//m_dataTermS.computeCostsS2(&streams[0]);
	//m_dataTermG.computeCostsG2(&streams[0]);

	cudaDeviceSynchronize();

	int smem_size = 13 * m_h * sizeof(float);
	//int smem_size = 9 * m_h * sizeof(float) + 512 * 4 * sizeof(float);
	KernDP << <m_w, 512, smem_size >> > (m_w, m_h, fnmax, d_disparity_colReduced, d_sum, d_valid,
		m_dataTermG.d_costsG, m_dataTermO.d_costsO, m_dataTermS.d_costsS,
		d_costTableG, d_costTableO, d_costTableS, d_dispTableG, d_dispTableO, d_dispTableS, d_indexTableG, d_indexTableO, d_indexTableS,
		m_priorTerm.d_costs0_, m_priorTerm.d_costs1_, m_priorTerm.d_costs2_O_G_, m_priorTerm.d_costs2_O_O_, m_priorTerm.d_costs2_O_S_, m_priorTerm.d_costs2_S_O_,
		N_LOG_0_0, m_vhor
	); // default stream

	cudaMemcpy(h_costTableG, d_costTableG, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_costTableS, d_costTableS, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_costTableO, d_costTableO, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_dispTableG, d_dispTableG, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dispTableO, d_dispTableO, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dispTableS, d_dispTableS, m_h * m_w * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_indexTableG, d_indexTableG, m_h * m_w * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_indexTableO, d_indexTableO, m_h * m_w * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_indexTableS, d_indexTableS, m_h * m_w * sizeof(glm::vec2), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

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
			if (p1.x == 1)
			{
				Stixel stixel;
				stixel.u = stixelWidth * u + stixelWidth / 2;
				stixel.vT = h - 1 - p1.y;
				stixel.vB = h - 1 - (p2.y + 1);
				stixel.width = stixelWidth;
				stixel.disp = h_dispTableG[u * m_h + p1.y];
				stixels.push_back(stixel);
			}
			minPos = p2;
		}
	}
}

void gpuStixelWorld::preprocess(const CameraParameters & camera, float sinTilt, float cosTilt)
{
	cudaMalloc((void**)&d_groundDisp, m_h * sizeof(float));
	h_groundDisp = new float[m_h];

	kernComputeGroundDisp << <divup(m_h, BLOCKSIZE), BLOCKSIZE >> > (d_groundDisp, m_h,
		camera.baseline, camera.height, camera.fu, camera.v0, sinTilt, cosTilt);

	cudaMemcpy(h_groundDisp, d_groundDisp, m_h * sizeof(float), cudaMemcpyDeviceToHost);

	m_dataTermG = gpuNegativeLogDataTermGrd(param_.dmax, param_.dmin, param_.sigmaG, param_.pOutG, param_.pInvG, camera,
		d_groundDisp, m_vhor, param_.sigmaH, param_.sigmaA, m_h, m_w);
	m_dataTermO = gpuNegativeLogDataTermObj(param_.dmax, param_.dmin, param_.sigmaO, param_.pOutO, param_.pInvO, camera, param_.deltaz, m_w, m_h);
	m_dataTermS = gpuNegativeLogDataTermSky(param_.dmax, param_.dmin, param_.sigmaS, param_.pOutS, param_.pInvS, m_w, m_h);
	m_priorTerm = gpuNegativeLogPriorTerm(m_h, m_vhor, param_.dmax, param_.dmin, camera.baseline, camera.fu, param_.deltaz,
		param_.eps, param_.pOrd, param_.pGrav, param_.pBlg, h_groundDisp);

}

void gpuStixelWorld::destroy()
{
	for (int i = 0; i < 5; i++) {
		cudaStreamDestroy(streams[i]);
	}

	cudaFree(d_disparity_original);
	cudaFree(h_disparity_colReduced);
	cudaFree(d_groundDisp);
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
	delete[] h_groundDisp;

}

void gpuNegativeLogDataTermGrd::init(float dmax, float dmin, float sigmaD, 
	float pOut, float pInv, const CameraParameters & camera, 
	float* d_groundDisparity, float m_vhor, float sigmaH, float sigmaA, int h)
{

	nLogPUniform_ = logf(dmax - dmin) - logf(pOut);
	const float cf = camera.fu * camera.baseline / camera.height;

	//h_nLogPGaussian_ = new float[h];
	//h_fn_ = new float[h];
	//h_cquad_ = new float[h];
	//h_costsG = new float[m_w * m_h];
	cudaMalloc((void**)&d_cquad_, h * sizeof(float));
	cudaMalloc((void**)&d_fn_, h * sizeof(float));
	cudaMalloc((void**)&d_nLogPGaussian_, h * sizeof(float));
	cudaMalloc((void**)&d_costsG, m_w * m_h * sizeof(float));

	dim3 dimBlock(divup(h, BLOCKSIZE));
	kernComputeNegativeLogDataTermGrd << <dimBlock,BLOCKSIZE >> > (h, d_groundDisparity, d_nLogPGaussian_, d_fn_, d_cquad_,
		camera.fv, camera.tilt, camera.height, cf, sigmaA, sigmaH, sigmaD, dmax, dmin, SQRT2, PI, pOut, m_vhor);

	//cudaMemcpy(h_nLogPGaussian_, d_nLogPGaussian_, h * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_fn_, d_fn_, h * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_cquad_, d_cquad_, h * sizeof(float), cudaMemcpyDeviceToHost);
}

inline void gpuNegativeLogDataTermGrd::computeCostsG1(float* d_disp_colReduced, cudaStream_t* stream)
{	
	int r = divup(m_w, BLOCKSIZE); int c = divup(m_h, BLOCKSIZE);
	dim3 dimGrid(r, c, 1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
	kernComputeCostsG << <dimGrid, dimBlock, 0, *stream >> > (m_w, m_h, nLogPUniform_, d_costsG, d_nLogPGaussian_, d_cquad_, d_fn_, d_disp_colReduced);

}

inline void gpuNegativeLogDataTermGrd::computeCostsG2(cudaStream_t* stream) {
	dim3 dimGrid = dim3(divup(m_w, BLOCKSIZE));

	//kernScanCosts << <dimGrid, BLOCKSIZE, 0, *stream >> > (m_w, m_h, d_costsG);
	kernScan1 << <m_w, 512, 512 * sizeof(float), *stream >> > (m_w, m_h, d_costsG);
	//kernScan1shf << <m_w, 512, 512 * sizeof(float), *stream >> > (m_w, m_h, d_costsG);

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
	//delete[] h_nLogPGaussian_;
	//delete[] h_fn_;
	//delete[] h_cquad_;
	//delete[] h_costsG;
}


inline void gpuNegativeLogDataTermObj::computeCostsO1(float * d_disp_colReduced, cudaStream_t* stream)
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
	kernComputeCostsOO << <dimGrid, dimBlock, 0, *stream >> > (m_w, fnmax, m_h, d_costsO, d_disp_colReduced, d_nLogPGaussian_, d_cquad_, nLogPUniform_);
	dimGrid = dim3(divup(m_w, BLOCKSIZE), divup(fnmax, BLOCKSIZE), 1);
}

inline void gpuNegativeLogDataTermObj::computeCostsO2(cudaStream_t* stream) {
	dim3 dimGrid = dim3(divup(m_w, BLOCKSIZE), divup(fnmax, BLOCKSIZE), 1);
	dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE);

	//kernScanCostsObj << <dimGrid, dimBlock, 0, *stream >> > (m_w, fnmax, m_h, d_costsO);
	//kernWarpSum << <m_w * fnmax, 512, 512 * sizeof(float), *stream >> > (m_w, m_h, fnmax, d_costsO);
	kernScan2 << <m_w * fnmax, 512, 512 * sizeof(float), *stream >> > (m_w, m_h, fnmax, d_costsO);

}

void gpuNegativeLogDataTermObj::destroy()
{
	cudaFree(d_nLogPGaussian_);
	cudaFree(d_cquad_);
	cudaFree(d_costsO);
}

void gpuNegativeLogDataTermObj::init(float dmax, float dmin, float sigmaD, float pOut, float pInv, const CameraParameters & camera, float deltaz)
{
	nLogPUniform_ = logf(dmax - dmin) - logf(pOut);

	cudaMalloc((void**)&d_cquad_, fnmax * sizeof(float));
	cudaMalloc((void**)&d_nLogPGaussian_, fnmax * sizeof(float));
	cudaMalloc((void**)&d_costsO, m_w * m_h * fnmax * sizeof(float));

	dim3 dimBlock(divup(fnmax, BLOCKSIZE));
	kernComputeNegativeLogDataTermObj << <dimBlock, BLOCKSIZE >> > (fnmax, d_cquad_, d_nLogPGaussian_,
		camera.fu, camera.baseline, sigmaD, deltaz, SQRT2, PI, pOut, dmin, dmax);

}

void gpuNegativeLogDataTermSky::init(float dmax, float dmin, float sigmaD, float pOut, float pInv, float fn)
{
	nLogPUniform_ = logf(dmax - dmin) - logf(pOut);

	const float ANorm = 0.5f * (erff((dmax - fn) / (SQRT2 * sigmaD)) - erff((dmin - fn) / (SQRT2 * sigmaD)));
	nLogPGaussian_ = logf(ANorm) + logf(sigmaD * sqrtf(2.f * PI)) - logf(1.f - pOut);

	cquad_ = 1.f / (2.f * sigmaD * sigmaD);

	cudaMalloc((void**)&d_costsS, m_h * m_w * sizeof(float));
}

void gpuNegativeLogDataTermSky::destroy()
{
	cudaFree(d_costsS);
}

inline void gpuNegativeLogDataTermSky::computeCostsS1(float *d_disparity_colReduced, cudaStream_t* stream)
{
	int r = divup(m_w, BLOCKSIZE); int c = divup(m_h, BLOCKSIZE);
	dim3 dimGrid(r, c, 1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
	kernComputeCostsS << <dimGrid, dimBlock, 0, *stream >> > (m_w, m_h, d_costsS, d_disparity_colReduced, nLogPUniform_, cquad_, nLogPGaussian_, fn_);
}

inline void gpuNegativeLogDataTermSky::computeCostsS2(cudaStream_t* stream) {
	dim3 dimGrid = dim3(divup(m_w, BLOCKSIZE));
	//kernScanCosts << <dimGrid, BLOCKSIZE, 0, *stream >> > (m_w, m_h, d_costsS);
	kernScan1 << <m_w, 512, 512 * sizeof(float), *stream >> > (m_w, m_h, d_costsS);
	//kernScan1shf << <m_w, 512, 512 * sizeof(float), *stream >> > (m_w, m_h, d_costsS);
	
}

void gpuNegativeLogPriorTerm::init(int h, float m_vhor, float dmax, float dmin, float b, float fu, float deltaz, float eps, float pOrd, float pGrav, float pBlg, float* groundDisparity)
{
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
