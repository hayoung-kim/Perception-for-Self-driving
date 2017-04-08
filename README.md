# Perception Algorithms for Self-driving Car
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Perception related projects of Udacity's Self-driving Car Nanodegree Program. 
* [Lane Line Finding](https://github.com/Hayoung-Kim/Perception-for-Self-driving/tree/master/CarND-Advanced-Lane-Lines)
* [Vehicle Detection](https://github.com/Hayoung-Kim/Perception-for-Self-driving/tree/master/CarND-Vehicle-Detection-P5)
* [Traffic Sign Classification]()


Summary
---
### Lane Line Finding
[Lane Line Finding](https://github.com/Hayoung-Kim/Perception-for-Self-driving/tree/master/CarND-Advanced-Lane-Lines)에서는 Camera Calibration, Color Thresholding, Gradient Thresholding and Image Wraping등의 전통적인 Computer Vision 테크닉을 이용하여 Lane Line을 찾습니다. 영상에서 찾아진 Lane Line은 Bird eye view로 변환된 영상의 pixel 좌표에서 meter 좌표로 변환되어, 현재 차량의 CTE(Cross Track Error), 도로의 Curvature을 구합니다.

<p align="center">
  <img src="./CarND-Advanced-Lane-Lines/img/output_video_try2.gif">
</p>


### Vehicle Detection
[Vehicle Detection](https://github.com/Hayoung-Kim/Perception-for-Self-driving/tree/master/CarND-Vehicle-Detection-P5)에서는 Vehicle과 non-Vehicle을 구분할 수 있는 SVM Classifier를 학습시킨 뒤, Sliding window 방법을 이용하여 영상에서 차량의 위치를 검출합니다. Multi-detection과 False Positive 문제를 해결하기 위하여 이전 프레임 및 현재 프레임의 정보를 이용하여 Heat-map을 구축한 뒤 최종적으로 차량의 위치를 판단 할 수 있도록 알고리즘을 구성하였습니다.

<p align="center">
  <img src="./CarND-Vehicle-Detection-P5/imgs/output_video.gif">
</p>


### Traffic Sign Classification
[Traffic Sign Classification]()에서는 CNN(Convolutional Neural Network)를 이용하여 43종류의 표지판을 구분해 내는 Classifier를 설계했습니다. [LeNet](http://yann.lecun.com/exdb/lenet/)을 Retrain시킨 결과, Test accuracy 92.6%로 43종류의 표지판을 구분해 낼 수 있었습니다.


