## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, the goal is to write a software pipeline to identify the lane boundaries in a video.

![pipeline](./img/pipeline.png)

Pipeline with one example image
---
You can check the detail of advanced lane line detection pipeline on [`pipeline.ipynb`](./pipeline.ipynb). 

The algorithm consists of:
* Camera Calibration
* Thresholding
    * Color thresholding
    * Gradient thresholding
* Perspective Transform (make bird eye view)
* Lane Finding
* Inverse Perspective Transform
* Draw result!

Usage
---
Follow [`generate_video.ipynb`](./generate_video.ipynb)!


Result
---
From `project_video.mp4`, Here is result!

<div align="center">
  <img src="./img/output_video.gif" alt="output_video">  
</div>  
