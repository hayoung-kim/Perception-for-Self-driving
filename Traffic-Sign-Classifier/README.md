## LeNet Implementation for Traffic Sign Classification - Tensorflow
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, 독일에서 사용되는 43 종류의 Traffic sign을 구분해 내는 것이 목표입니다. 이를 위해 Traffic Sign data를 이용하여 [LeNet](http://yann.lecun.com/exdb/lenet/)의 Output layer의 수를 43개로 바꾼 뒤, 네트워크를 Retrain 시켰습니다.

Dataset
---
<p align="center">
  <img src="./Traffic-Sign-Classifier/images/data_example.png">
</p>

[German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)의 이미지들을 이용합니다. Training/Validation/Test를 위한 Pickled dataset은 아래에서 받을 수 있습니다.

Pickled dataset: 
* [Training data](https://www.dropbox.com/s/8ldwwtgp8n4owuv/train.p?dl=0) (34799)
* [Validation data](https://www.dropbox.com/s/okqaizp6w1inx79/valid.p?dl=0) (4410)
* [Test data](https://www.dropbox.com/s/cs96orc7i3sfvr3/test.p?dl=0) (12630)

or you can download all data [`Traffic_Signs_data.zip`](https://www.dropbox.com/s/9qaiamsvzknhrvb/Traffic_Signs_data.zip?dl=0)

Training
---
### Environment
* Python 3.5.2
* Tensorflow 1.0.1


### Optimizer Settings
* Optimizer: AdaGrad
* Learning rate: 10^(-4)
* Loss: Cross Entropy Error
* Batch size: 1024
* Epoch: 10


Result
---


