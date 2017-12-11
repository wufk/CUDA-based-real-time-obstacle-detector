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

* Usage: ` CUDA-based-real-time-obstacle-detector.exe left_image right_image camera_config.xml stixelWidth`. 

* Example: If you download the dataset from Ref-3, run ` CUDA-based-real-time-obstacle-detector.exe images\img_c0_%09d.pgm images\img_c1_%09d.pgm camera.xml 7 `.

## Results

[![](https://github.com/wufk/CUDA-based-real-time-obstacle-detector/blob/master/img/stixels_goodWeather.gif)]()
Figure 2: Stixels computation under good weather. The above figure is the disaprity image. 

## GPU implementation pipeline
1. Suppose the input frame streams have same properties, e.g. image resolution and cameral configuration, read the first frame to allocate resources and pre-compute the variables needed.
2. On each image frames:
      * Compute disparities from the left and right image.
      * Reduce the columns by a factor of stixel width by replacing the disparities of consecutive pixels by their average. Transpose the image to have a better access pattern for subsequent tasks.
      * Compute the 'look-up tables' to have constant time computation.
      * Compute the stixels using dynamic programming and keep track of the stixels by backtracking.

## Performance Analysis
Suppose the height and width of the reduced-disparity image is `h x w`, and the disparity range is `dmax`. The dynamic programming part is the most time consuming part among the four stages discussed above while computing stixels for each frame. The total amount of work would be `O(h x h x w)`. The computation of the Object look-up table is also a main part. It basically compute a cost value based on the disparity using prefix sums. The complexity is `O(h x w x dmax)`. The performance focus is on the computation of these two operations. 

For the implementation below, the input data has `1024x333` pixels and `dmax = 64`. If not sepcified, the stixel width is `7`.

### Naive way and shared memory
In the dynamic programming stage, there are repeated read of the cost tables. Shared memory in GPU provides us a customized cache to enable fast data transfer. The computation is on each column is independent of other columns, so we can load many variables into the shared memory. The results are shown below.

|Method | Niave | Shared Memory |
|---|---:|---:|
|time per frame (ms)| 68.145| 61.817|

The implementation after this part all load cost table into shared memory

### CUDA streams
[![](https://github.com/wufk/CUDA-based-real-time-obstacle-detector/blob/master/img/sequential.PNG)]()
Figure 3: Sequential kernel launch

Streams are "Task-level" parallelism. By default, all kernel are launched on the same stream, which is shown on the figure above. In face, before the dynamic programming stage, the computation of the cost tables are indepedent. Running theses kernel functions in the same stream waste the resources. In this case, five additional streams are created. After some adjustment of the launching sequence, the results are show below.

[![](https://github.com/wufk/CUDA-based-real-time-obstacle-detector/blob/master/img/streams3ndTry.PNG)]()
Figure 4: Parallel kernel lanch

|Method | No streams | Use streams |
|---|---:|---:|
|time per frame (ms)| 61.817| 59.235|

### Improving scan performance
The scan tasks in this project are mainly doing prefix sum on each row of cost tables, which are 2D Matrices. So far my implementation is launching `w` threads and call `thrust::inclusive_scan` to compute the cost tables. The performance is not very good and it turns out that calling `thrust` in kernel functions will lead to sequential a scan. Replacing it with a parallel scan in shared memory improve the performance a lot, which is shown below.

|Method |  Scan in shared memory | Sequential |
|---|---:|---:|
|time per frame (ms)| 49.767 | 59.235|

### Scan on load
In the dynamic programming stage, the kernel function launch `w` blocks and 'h' threads in each block, which corresponds to the dimension of the input data. In addition, computing the prefix sums of the cost tables launch the same number of blocks and threads. So the scan can take place at the very beginning of the dyanmic programming stage. This saves the time of launching kernel functions and the computation also take place in the shared memory. Results are shown below

|Method |  Scan in separate kernel | Scan on load |
|---|---:|---:|
|time per frame (ms)| 49.767 | 45.391|

### Warp shuffle
Shuffle enables threads communicate within a warp in registers. We can use shuffle for scan. 
[![](https://github.com/wufk/CUDA-based-real-time-obstacle-detector/blob/master/img/scan.jpg)]()

The above figure shows the naive scan. A shuffle scan algorithm can be implemented using `__shfl_up`:
```
  scan each warp using __shufl_up.
  write warpsum whithin the warp in shared memory buffer.
  scan the buffer using __shufl_up.
  add the increment back to all elements in their respective warp
```
The results are follows

|Method |  Scan using shuffle | Scan using shared memory|
|---|---:|---:|
|time per frame (ms)| 36.409 | 45.391|

### Summary

[![](https://github.com/wufk/CUDA-based-real-time-obstacle-detector/blob/master/img/chart1.png)]()
Image size: 1024 * 333 pixels, Stixel width: 7 pixels


Image size: 1024 * 333 pixels

## Future work
### Under bad weather

[![](https://github.com/wufk/CUDA-based-real-time-obstacle-detector/blob/master/img/stixels_badweather2.gif)]()
Figure : Stixels computation under bad weather. The above figure is the original image.

Results shows that under good weather, the detection is pretty well. While in rainy days, the reflection of the the vehicles are also recognized as stixels, which is not very satisfying.

### Shortcoming

## Reference
1. Hernandez-Juarez, Daniel, et al. "GPU-accelerated real-time stixel computation." Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017.

2. Cordts, Marius, et al. "The Stixel world: A medium-level representation of traffic scenes." Image and Vision Computing (2017).

3. Dataset: [6d-vision](http://www.6d-vision.com/ground-truth-stixel-dataset)

4. [Similar previous work](https://github.com/gishi523/multilayer-stixel-world)
