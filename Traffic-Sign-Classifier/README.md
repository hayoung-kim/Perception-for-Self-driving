## LeNet Implementation for Traffic Sign Classification - Tensorflow
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The object of the project is to distinguish 43 different types of traffic sign used is Germany. Number of Output layers in [LeNet](http://yann.lecun.com/exdb/lenet/) has changed to 43. Then, the network has retrained using Traffic sign data

Dataset
---
<p align="center">
  <img src="./images/data_example.png">
</p>

Images of [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) have used for training and test. Pickled datasets are available below.


### Vanilla Pickled dataset 
* [Training data](https://www.dropbox.com/s/8ldwwtgp8n4owuv/train.p?dl=0) (102MB)
* [Validation data](https://www.dropbox.com/s/cs96orc7i3sfvr3/test.p?dl=0) (12.9MB)
* [Test data](https://www.dropbox.com/s/cs96orc7i3sfvr3/test.p?dl=0) (37MB)

### Augmented Pickled Dataset
* [Training data](https://www.dropbox.com/s/v09biif3epk922v/train_aug.p?dl=0) (431MB)

or you can download all data [`Traffic_Signs_data.zip`](https://www.dropbox.com/s/9qaiamsvzknhrvb/Traffic_Signs_data.zip?dl=0) (395MB)


Training
---

<p align="center">
  <img src="./images/convnet_fig.png">
</p>

### Network
LeNet with `Batch Normalization` before each activation layer. No dropout! 

Convolutional weights were initialized by [`'He' method`](https://arxiv.org/abs/1502.01852).

### Environment
* Python 3.5.2
* Tensorflow 1.0.1

### Optimizer Settings
* Optimizer: `Adam`
* Learning rate: `10e-3`
* Loss: `Cross entropy`
* Batch Size: `1024`
* Epoch: `100`




Result
---
### Test Accuracy = 94.0%

<p align="center">
  <img src="./images/result_acc.png">
  <img src="./images/result_loss.png">
</p>



### Inference using arbitrary traffic sign data

<p align="center">
  <img src="./images/softmax_probabilities_1.png">
  <img src="./images/softmax_probabilities_2.png">
  <img src="./images/softmax_probabilities_3.png">
</p>

### Confusion Matrix
<p align="center">
  <img src="./images/confusion_matrix.png">
</p>