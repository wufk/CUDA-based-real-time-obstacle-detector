# CUDA-based real time obstacle detector
* Fengkai Wu
* Tested on: Windows 10, i7-4700HQ @ 2.40GHz 4GB, GEFORCE 745M 2048MB

## Overview
Driving assistance software on autonomous vehicles require real-time and robust awareness of the road situation. The project will focus on the implementation of a real-time algorithm that can be GPU accelerated for obstacle detection on a self-driving car.

The large amount of low-level pixel data can be represented as stixel world. given the fact that man-made environments are mostly presented to be horizontal and vertical. A stixel is a segment of image columns with a width of a few pixels, which can be used to represent the sky, the road and obstacles around. By classifying the stixels in the view, automobiles can be guided to the right direction.

The principle of calculating the stixels is based on a Maximum A Posteriori (MAP) probabilistic model, which can be solved using dynamic programming. The recurrence in the algorithm can be parallelized and proves to suit the scheme of GPU computation. See Ref-1 for details.

This project aims to develop a fast real-time obstacle detector using OpenCV and CUDA. The figure below shows the desired result. Figure credit to Ref-1.

[![](https://github.com/wufk/CUDA-based-real-time-obstacle-detector/blob/master/img/stixels.png)]()

## Build and Usage
* Make sure you have Visual Sutdio 2015 and CUDA 8 installed.
* Make sure you have opencv. 
* Navigate to the cloned repository. Generate/build project using CMake:
  ```
  mkdir build
  cd build
  cmake-gui ../
  ```
  Configure and generate.

* Usage: ` CUDA-based-real-time-obstacle-detector.exe left_image right_image camera_config.xml stixelWidth`. If you download the dataset from Ref-3, for example, run ` CUDA-based-real-time-obstacle-detector.exe images\img_c0_%09d.pgm images\img_c1_%09d.pgm camera.xml 7 `.

## Results
[![](https://github.com/wufk/CUDA-based-real-time-obstacle-detector/blob/master/img/stixels_goodWeather.gif)]()
Figure.2 Stixels computation under good weather. The above figure is the disaprity image. 

[![](https://github.com/wufk/CUDA-based-real-time-obstacle-detector/blob/master/img/stixels_badweather2.gif)]()
Figure.3 Stixels computation under bad weather. The above figure is the original image.

Results shows that under good weather, the detection is pretty well. While in rainy days, the reflection of the the vehicles are also recognized as stixels, which is not very satisfying.

## Performance Analysis


## Reference
1. Hernandez-Juarez, Daniel, et al. "GPU-accelerated real-time stixel computation." Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017.

2. Cordts, Marius, et al. "The Stixel world: A medium-level representation of traffic scenes." Image and Vision Computing (2017).

3. Dataset: [6d-vision](http://www.6d-vision.com/ground-truth-stixel-dataset)
