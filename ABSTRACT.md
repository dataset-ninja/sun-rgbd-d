The authors have introduced the **SUN RGB-D Dataset** benchmark suite to advance the state-of-the-art in major scene understanding tasks. This dataset, captured by four different sensors, comprises 10,335 RGB-D images, making it comparable in scale to [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/). Each image is densely annotated, totaling 146,617 2D polygons. This extensive dataset allows for the training of data-hungry algorithms for scene understanding tasks, helps prevent overfitting to a small testing set, and facilitates the study of cross-sensor bias.

## Motivation

Scene understanding remains one of the most fundamental challenges in computer vision. Despite significant progress over the past decades, achieving general-purpose scene understanding continues to be difficult. The recent advent of affordable depth sensors in the consumer market has enabled the acquisition of reliable depth maps at low cost, leading to breakthroughs in various vision tasks such as body pose recognition, intrinsic image estimation, 3D modeling, and Structure from Motion (SfM) reconstruction. RGB-D sensors have notably accelerated advancements in scene understanding.

However, while color images can be easily obtained from the Internet, large-scale RGB-D data is not readily available online. As a result, existing RGB-D recognition benchmarks, like NYU Depth v2, are significantly smaller than modern recognition datasets for color images. Although these smaller datasets have successfully initiated progress in RGB-D scene understanding over recent years, their limited size now poses a critical bottleneck for advancing research further. These limitations lead to algorithm overfitting during evaluation and hinder the training of data-hungry algorithms that currently dominate color-based recognition tasks.

A large-scale RGB-D dataset would allow the field to leverage the same successes achieved in color image recognition. Moreover, while existing RGB-D datasets provide depth maps, their annotations and evaluation metrics are primarily within the 2D image domain. For scene understanding to be truly useful in real-world applications, a comprehensive approach that fully integrates the 3D information provided by RGB-D data is essential.

## Sensors

The goal of the authors dataset construction is to obtain an image dataset captured by various RGB-D sensors at a similar scale as the PASCAL VOC object detection benchmark. To improve the depth map quality, they take short videos and use multiple frames to obtain a refined depth map. For each
image, the authors annotated the objects with 2D polygons. Since there are several popular sensors available, with different size and power consumption, the authors construct the dataset using four kinds – Intel RealSense 3D Camera for tablets, Asus Xtion LIVE PRO for laptops, and Microsoft Kinect versions 1 and 2 for desktop.

<img src="https://github.com/dataset-ninja/sun-rgbd-d/assets/120389559/226a65a5-6cad-40ed-81cb-e26ae93535ce" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Comparison of the four RGB-D sensors.</span>

**Intel RealSense** is a lightweight, low-power depth sensor designed for tablets. The authors obtained two pre-release samples from Intel. This sensor projects an IR pattern onto the environment and uses stereo matching to generate the depth map. For outdoor environments, it can automatically switch to stereo matching without the IR pattern. However, upon visual inspection of the 3D point cloud, the authors found the depth map quality to be inadequate for accurate object recognition outdoors. Consequently, they used this sensor exclusively for capturing indoor scenes. Although the raw depth quality of Intel RealSense is inferior to that of other RGB-D sensors and its effective range for reliable depth measurements is shorter (depth becomes very noisy around 3.5 meters), its lightweight design allows it to be embedded in portable devices and widely deployed in consumer markets. Therefore, it is crucial to study algorithm performance with this type of sensor.

**Asus Xtion** and **Kinect v1** both utilize a near-IR light pattern for depth sensing. Asus Xtion is significantly lighter and is powered solely by USB, although it offers lower color image quality compared to Kinect v1. On the other hand, Kinect v1 requires an additional power source. Both sensors produce raw depth maps that exhibit noticeable quantization effects.

**Kinect v2** utilizes time-of-flight technology and consumes a significant amount of power. It captures more accurate raw depth maps with high fidelity, allowing for detailed depth measurement, but it struggles with black objects and slightly reflective surfaces. Although the hardware supports a long-distance depth range, the official Kinect for Windows SDK limits depth to 4.5 meters and applies filtering that often loses object details. To address this, the authors developed their driver and decoded the raw depth on the GPU, enabling real-time video capture without depth cutoffs or additional filtering.

## Sensor calibration

For RGB-D sensors, the authors needed to calibrate the intrinsic camera parameters and the transformation between the depth and color cameras. They used the default factory parameters for Intel RealSense and the default parameters from the OpenNI library for Asus Xtion, without accounting for radial distortion. However, due to the strong radial distortion in Kinect v2, they calibrated all cameras using a standard calibration toolbox. The depth cameras were calibrated by computing parameters with the IR image, identical to the depth camera. To prevent overexposure when viewing the checkerboard on IR, they covered the emitter with a piece of paper. The stereo calibration function was then used to calibrate the transformation between the depth (IR) and color cameras.

## Depth map improvement

The depth maps from these cameras are not perfect due to measurement noise, view angle issues with reflective surfaces, and occlusion boundaries. Since all the RGB-D sensors operate as video cameras, the authors leveraged nearby frames to improve the depth map by providing redundant data for denoising and filling in missing depths.

They propose a robust algorithm for depth map integration from multiple RGB-D frames. For each nearby frame within a time window, they project the points to 3D, generate a triangulated mesh from these points, and estimate the 3D rotation and translation between this frame and the target frame for depth improvement. Using this estimated transformation, they render the depth map of the mesh from the target frame camera. Once they have aligned and warped depth maps, they integrate them to achieve a robust estimation. For each pixel location, the authors compute the median depth and the 25% and 75% percentiles. If the raw target depth is missing or falls outside the 25%-75% range, and the median is computed from at least 10 warped depth maps, they use the median depth value. Otherwise, they retain the original value to avoid oversmoothing. Compared to a method that uses a 3D voxel-based TSDF representation, their depth map improvement algorithm requires much less memory and runs faster at equal resolution, allowing for higher-resolution integration.

## Data acquisition

To build a dataset on the scale of PASCAL VOC, the authors captured a substantial amount of new data and combined it with existing RGB-D datasets. They collected 3,784 images using Kinect v2 and 1,159 images using Intel RealSense. Additionally, they included 1,449 images from the [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) dataset and manually selected 554 realistic scene images from the [Berkeley B3DO Dataset](http://kinectdata.com/), both captured with Kinect v1. From the SUN3D videos captured by Asus Xtion, they manually chose 3,389 distinct frames without significant motion blur. Altogether, they amassed 10,335 RGB-D images.

To capture data, they attached an Intel RealSense to a laptop and carried it around. For Kinect v2, they used a mobile laptop harness and a camera stabilizer. Due to Kinect v2's high power consumption, they powered it using a 12V car battery and a 5V smartphone battery for the sensor and adapter circuit. Since RGB-D sensors perform optimally indoors, their data collection focused on universities, houses, and furniture stores in North America and Asia.

<img src="https://github.com/dataset-ninja/sun-rgbd-d/assets/120389559/e93a328a-13d1-414b-8090-3216260adb28" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Data Capturing Process. (a) RealSense attached to laptop, (b) Kinect v2 with battery, \(c\) Capturing setup for Kinect v2.</span>

<img src="https://github.com/dataset-ninja/sun-rgbd-d/assets/120389559/685b5efd-7698-42b4-89ba-9f706309b8f7" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Example images with annotation from the dataset.</span>

## Ground truth annotation

For each RGB-D image, the authors created [LabelMe](http://labelme.csail.mit.edu/Release3.0/) style 2D polygon annotations. To ensure quality and consistency, they generated their own ground truth labels for images from other datasets, with the exception of NYU, for which they used the existing 2D segmentation. They developed a LabelMe-style tool for Amazon Mechanical Turk to facilitate 2D polygon annotation. To maintain high label quality, they incorporated automatic evaluation within the tool. Each image had to meet specific criteria to complete the HIT: at least six objects needed to be labeled, the union of all object polygons had to cover at least 80% of the image, and small polygons (covering less than 30% of the image area) had to collectively cover at least 30% of the total image area. This approach prevented workers from cheating by using large polygons to cover everything. The authors then visually inspected the labeling results and manually corrected the layer ordering as necessary. Low-quality labelings were sent back for relabeling. They paid $0.10 per image, with some images requiring multiple iterations to meet their quality standards.

## Label statistics

For the 10,335 RGB-D images, the authors have 146,617 2D polygons annotated. Therefore, there are 14.2 objects in each image on average. In total, there are 47 ***scene*** categories and about 800 object categories.
