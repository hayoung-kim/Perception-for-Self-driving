# Perception Algorithms for Self-driving Car
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Perception related projects of Udacity's Self-driving Car Nanodegree Program. 
* [Lane Line Finding](./Lane-Line-Finding/)
* [Vehicle Detection](./Vehicle-Detection/)
* [Traffic Sign Classification](./Traffic-Sign-Classifier/)


Summary
---
### Lane Line Finding
Traditional Computer Vision Techniques, such as Camera calibration, Color Thresholding, and Image Wrapping, have used for Lane Line Finding. Lane Line found in Bird eye view has converted from pixel unit to meter unit, which is calculated to obtain `CTE(Cross Track Error)` of vehicle and `Curvature` of the Lane.

<p align="center">
  <img src="./Lane-Line-Finding/img/output_video_try2.gif">
</p>


### Vehicle Detection
`SVM` Classifier has used to classify Vehicle and Non-Vehicle and `Sliding window` Method has used to detect vehicles from the image. The problem of Multi-detection and False Positive is prevented by `Heat-map` made up with information of current image frame and previous image frame.

<p align="center">
  <img src="./Vehicle-Detection/imgs/output_video.gif">
</p>


### Traffic Sign Classification
CNN(Convolution Neural Network) has used for Traffic Sign Classification, which recognizes and distinguish 43 different types of traffic sign. Test accuracy showed up to 93.5% in distinguishing traffic signs as a result of retraining [LeNet](http://yann.lecun.com/exdb/lenet/).

<p align="center">
  <img src="./Traffic-Sign-Classifier/images/convnet_fig.png">
  <img src="./Traffic-Sign-Classifier/images/softmax_probabilities_2.png">
  <img src="./Traffic-Sign-Classifier/images/softmax_probabilities_1.png">
</p>


