#pragma once
#include "common.h"

class cpuStixelWorld : StixelWorld {

public:
	cpuStixelWorld(const Parameters& param) {
		param_ = param;
	}

	virtual void compute(const cv::Mat& disp, std::vector<Stixel>& stixels) override;

private:
	float averageDisparity(const cv::Mat& disparity, const cv::Rect& rect, int minDisp, int maxDisp);

	class FreeSpace
	{
	public:

		using CameraParameters = StixelWorld::CameraParameters;

		struct Parameters
		{
			float alpha1;       //!< weight for object evidence
			float alpha2;       //!< weight for road evidence
			float objectHeight; //!< assumed object height
			float Cs;           //!< cost parameter penalizing jumps in depth
			float Ts;           //!< threshold saturating the cost function
			int maxPixelJump;   //!< maximum allowed jumps in pixel (higher value increases computation time)

								// default settings
			Parameters()
			{
				alpha1 = 2;
				alpha2 = 1;
				objectHeight = 0.5f;
				Cs = 50;
				Ts = 32;
				maxPixelJump = 100;
			}
		};

		FreeSpace(const Parameters& param = Parameters()) : param_(param)
		{
		}

		void compute(const cv::Mat1f& disparity, std::vector<int>& path, const CameraParameters& camera);

	private:
		Parameters param_;
	};

	class HeightSegmentation
	{
	public:

		using CameraParameters = StixelWorld::CameraParameters;

		struct Parameters
		{
			float deltaZ;     //!< allowed deviation in [m] to the base point
			float Cs;         //!< cost parameter penalizing jumps in depth and pixel
			float Nz;         //!< if the difference in depth between the columns is equal or larger than this value, cost of a jump becomes zero
			int maxPixelJump; //!< maximum allowed jumps in pixel (higher value increases computation time)

							  // default settings
			Parameters()
			{
				deltaZ = 5;
				Cs = 8;
				Nz = 5;
				maxPixelJump = 50;
			}
		};

		HeightSegmentation(const Parameters& param = Parameters()) : param_(param)
		{
		}

		void compute(const cv::Mat1f& disparity, const std::vector<int>& lowerPath, std::vector<int>& upperPath, const CameraParameters& camera);

	private:
		Parameters param_;
	};

	struct CoordinateTransform
	{
		using CameraParameters = StixelWorld::CameraParameters;

		CoordinateTransform(const CameraParameters& camera) : camera(camera)
		{
			sinTilt = (sinf(camera.tilt));
			cosTilt = (cosf(camera.tilt));
			B = camera.baseline * camera.fu / camera.fv;
		}

		inline float toY(float d, int v) const
		{
			return (B / d) * ((v - camera.v0) * cosTilt + camera.fv * sinTilt);
		}

		inline float toZ(float d, int v) const
		{
			return (B / d) * (camera.fv * cosTilt - (v - camera.v0) * sinTilt);
		}

		inline float toV(float Y, float Z) const
		{
			return camera.fv * (Y * cosTilt - Z * sinTilt) / (Y * sinTilt + Z * cosTilt) + camera.v0;
		}

		inline float toD(float Y, float Z) const
		{
			return camera.baseline * camera.fu / (Y * sinTilt + Z * cosTilt);
		}

		CameraParameters camera;
		float sinTilt, cosTilt, B;
	};
};