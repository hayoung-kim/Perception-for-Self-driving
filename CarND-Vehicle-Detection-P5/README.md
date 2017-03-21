# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, the goal is to write a software pipeline to detect vehicles in a video [`project_video.mp4`](./project_video.mp4). 

![pipeline](./imgs/pipeline_resized.png)


Dataset
---
### Vehicles and Non-vehicles data
Udacity CarND provides datasets for vehicles and non-vehicles data to train classifier. It contains `8792` vehicle images and `8968` non-vehicle images. These datasets are comprised of images taken from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.
* [Vehicles data](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) 
* [Non-vehicles data](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) 

Vehicle Detection Pipeline
---

You can find all the detail of vehicle detection pipeline on [`Vehicle_Detection_Pipeline.ipynb`](./Vehicle_Detection_Pipeline.ipynb).

The algorithm is structured as follows:
1. **Extracting Features** from images
    * `Spartially Binned Color`, `Color Histogram`, `HOG`
    * `skimage.feature.hog()` is used for extracting HOG features.
2. **Training** Classifier
    * `sklearn.svm.LinearSVC()` function is used.
3. **Searching** Using Sliding Window and **Classifying** Each Window
    * `find_car()` function is used for searching and classifying.
4. Removing **Multiple Detections** and **False Positives**
    * `apply_threshold()` function and `scipy.ndimage.measurements.label()` function is used.



Result
---
| Input | Result |
|:-----------------:|:----------------:|
| ![input_video](./imgs/project_video.gif)    | ![output_video](./imgs/output_video.gif)      |

Discussion
---
