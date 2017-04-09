**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/data_distribution.png "class num bar chart"
[image2]: ./images/distort.png "Grayscaling"
[image3]: ./images/data_distribution_augmented.png "class num augmented bar chart"
[image4]: ./images/confusion_matrix.png "confusion matrix"
[image5]: ./images/five_test_images.png "five test images"
[image6]: ./images/top_prob_1.png "misclassification"


---
You're reading it! and here is a link to my [project code](https://github.com/Hayoung-Kim/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is (34799, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.
A given training data set has a different number of data for each class. 
Especially when the number of data is large, there are close to 2,000, 
while when the number of data is small, about 200. 

That is, there is a difference in the number of training data given by class as below figure.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the eleventh code cell of the IPython notebook.

I decided to convert the images to be normalized. Since the gradient-based optimization algorithm is used for learning, it is recommended to input an input with mean value 0.
This is because this process can reduce the inability to control the gradient in a network that multiplies the weight and adds bias.

In this case, this process is done as follow:

(R-255)/255 + 0.5, (G-255)/255 + 0.5, (B-255)/255 + 0.5

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used an already splited dataset from Udacity github. So my code does not address this question.

Instead of this, I decided to generate additional training data. As can be seen from the distribution of training data, 
there is an imbalance in the training set between classes. 
I thought that this would have a negative impact when learning the network, 
so I created more data by adjusting the brightness after randomly cropping.

The code for this step is contained in the seventh to tenth code cells of IPython notebook.

Here is an example of an original image and an distorted image for augmenation:

![alt text][image2]

The difference between the original data set and the augmented data set is the following:
* Distribution of Training Data Set: I generated many augmented images for classes with large amount of data 
and less augmented images for large classes with large amount of data.
It is set to have about 2000 training data for all classes.
* Difficulty: From the viewpoint of training, I expected that it would be more hard to solve the problem 
using augmented data set rather than using original data set. Because augmented data set contains some part of original image.

After this process, the distribution of augmented training data is as follow (the number of augmented training data is 147174):

![alt text][image3]

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fourteenth cell of the ipython notebook.

This network was created by modifying LeNet-5, which I learned in class. 
The important modification is that the depth of the convolution layer is increased. 
This seems to reduce the tendency of the existing LeNet-5 to not distinguish the details of the sign when used for traffic sign classification.
Additionally, This final model architecture includes the dropout layer to prevent the over-fitting problem.
and it includes batch normalization layer to improve gradient-based optimization performance.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| Batch normalization   |                                               |
| RELU					|												|
| Dropout    	      	| keep probability 0.5 for training	    		|
| Max pooling	    	| outputs 14x14x32								|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 10x10x64 	|
| Batch normalization   |                                               |
| RELU					|												|
| Dropout    	      	| keep probability 0.5 for training	    		|
| Max pooling	    	| outputs 5x5x64								|
| Fully connected		| inputs 1200(flatten 5x5x64), outputs 120		|
| Fully connected		| inputs 120, outputs 84	        			|
| Fully connected		| inputs 84, outputs 43	            			|
| Softmax				|         								    	|
|						|												|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 16th cell of the ipython notebook. 

* Optimizer: AdamOptimizer
* Learning rate: 0.002
* Dropout probability: 0.5
* Batch size: 5000
* Epoch: ~ 300 (I set the number of epochs to 800, but stopped the training when the optimizer could not reduce the loss anymore.)


#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the nineth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 93.3%
* validation set accuracy of 94.2%
* test set accuracy of 92.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * LeNet-5[(Link)](http://yann.lecun.com/exdb/lenet/) was my first architecture. I choosed it because it's simple and it have pretty nice performance on image classification. 

* What were some problems with the initial architecture?
    * It was too simple to classify 43 labeled images.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * Firstly, I added dropout layer for preventing over fitting. This made classification algorithm getting more accuracy.
    * Secondly, I added batch normalization layer for letting optimization algorithm works well. By using this technique, learning rate could be large than before so that I could get results more faster.

* Which parameters were tuned? How were they adjusted and why?
    * I increased the depth of convolution layer compared to that of original LeNet-5. When I used LeNet-5, It classifies overall shape of traffic signs, but there was a tendency not to distinguish the specific shape inside.
    * So I thought the number of features, which is needed to classify them, was insufficient. As the number of features is highly related to depth of the convolution layer, I tried to increase it.
    * It works. Test accuracy is improved: 87.3% to 92.6%

#### 6. Performance of traffic sign Classifier

Here is confusion matrix:

![alt text][image4]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

The first image might be difficult to classify because the form of the traffic sign is different from that used for learning.
The second image also might be difficult to do because the pillar of the traffic sign occupies a large part of image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Left      		| Keep Left   									| 
| Children Crossing     | Bicycles Crossing	    						|
| 50 km/h				| 50 km/h										|
| Stop Sign	      		| Stop Sign	    				 				|
| Priority Road			| Priority Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The result is less than test set accuracy of 92.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the second image, the model is relatively sure that this is a bicycles crossing (probability of 0.67). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .67         			| Bicycles crossing	        					| 
| .19     				| Children crossing								|
| .04					| Bumpy road                                    |
| .03	      			| Road narrows on the right				 		|
| .02				    | Road work    						        	|

![alt text][image6]