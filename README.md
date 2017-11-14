# CUDA-based real time obstacle detector
* Fengkai Wu

## Overview
Driving assistance software on autonomous vehicles require real-time and robust awareness of the road situation. The project will focus on the implementation of a GPU-accelerated computation from the data of a stereo camera on a self-driving car.

The large amount of low-level pixel data can be represented as stixel world. given the fact that man-made environments are mostly presented to be horizontal and vertical. A stixel is a segment of image columns with a width of a few pixels, which can be used to represent the sky, the road and obstacles around. By classifying the stixels in the view, automobiles can be guided to the right direction.

The principle of calculating the stixels is based on a Maximum A Posteriori (MAP) probabilistic model, which can be solved using dynamic programming. The recurrence in the algorithm can be parallelized and proves to suit the scheme of GPU computation. 

This project aims to develop a fast real-time obstacle detector using OpenCV and CUDA. The figure below shows the desired result.

## Milestones
* 11/20 Set up the environment and gives a naïve implementation by CPU
* 11/27 Parallelize the computation on GPU and some initial optimization
* 12/4  Further optimize the model, compare with other real-time methods like CNN
* 12/11 Final performance analysis, provide a video demo and give the final presentation

## Reference
1. Hernandez-Juarez, Daniel, et al. "GPU-accelerated real-time stixel computation." Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017.

2. Cordts, Marius, et al. "The Stixel world: A medium-level representation of traffic scenes." Image and Vision Computing (2017).
