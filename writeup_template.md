# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./output_images/visualization_training.png "Visualization Training"
[image2]: ./output_images/visualization_validation.png "Visualization Validation"
[image3]: ./output_images/visualization_test.png "Visualization Test"
[image4]: ./output_images/grayscale.png "Grayscaling"
[image5]: ./output_images/normalized.png "Normalization"
[image6]: ./data/web/resized/traffic_sign_1.png "Traffic Sign 1"
[image7]: ./data/web/resized/traffic_sign_2.png "Traffic Sign 2"
[image8]: ./data/web/resized/traffic_sign_3.png "Traffic Sign 3"
[image9]: ./data/web/resized/traffic_sign_4.png "Traffic Sign 4"
[image10]: ./data/web/resized/traffic_sign_5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! All of the submitted code is stated in the Ipython notebook "./Traffic_Sign_Classifier.ipynb".

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are distributed within the training set.

![alt text][image1]

The corresponding distribution of the validation set looks like follows:

![alt text][image2]

And the distribution of the test set:

![alt text][image3]

These distributions look very similar to each other.

Example images of each traffic sign are also plotted in the Ipython notebook "./Traffic_Sign_Classifier.ipynb".

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale in order to abstract from variations in brightness (e.g. some images were captured at day, some at night)

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

As a last step, I normalized the image data to have equal mean and equal variance images.

The output of a normalized example image is subsequently shown:

![alt text][image5] 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on LeNet-5 with bigger layer sizes and includes dropout layers. In detail, it consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x64 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 13x13x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x128 				    |
| Flatten feature maps	| outputs 4608 									|
| Fully connected		| outputs 512 									|
| RELU					|												|
| Dropout		        | keep probability = 0.5, outputs 512 			|
| Fully connected		| outputs 256 									|
| RELU					|												|
| Dropout		        | keep probability = 0.5, outputs 256 			|
| Fully connected		| outputs 43 									|
| Softmax				| outputs 43        							|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an SGD optimizer with momentum 0.9 minimizing the cross entropy loss. In total, the model was trained for 500 epochs with a batch size of 512. The initial learning rate was set to 0.001 and remains constant during training. Moreover, the weights and biases of the model were initialized according to a truncated normal distribution ($\mu = 0$, $\sigma = 0.1$).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.976
* validation set accuracy of 0.961 
* test set accuracy of 0.948

These values were calculated in code cell 10 and called from code cell 11 (training and validation) respectively code cell 12 (test) of the Ipython notebook "./Traffic_Sign_Classifier.ipynb".

An iterative approach was chosen to get to the final solution:

* What was the first architecture that was tried and why was it chosen?
The first architecture that was chosen was LeNet-5 because it is supposed to be a suitable baseline model for the problem of traffic sign recognition and should yield a validation accuracy of 0.89.

* What were some problems with the initial architecture?
Even though LeNet-5 should give a validation accuracy of 0.89 this accuracy couldn't be achieved at the beginning. Instead, the model didn't learn from the fed in training data at all which I thought was a reason of underfitting. So I modified the architecture which is explained afterwards.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Due to potential underfitting I adjusted the architecture by increasing the layer widths so that the model get the capacity to learn from the data. The model showed a relatively high training accuracy but still a too little validation accuracy (sign for overfitting). So I included dropout with a keep probability of 0.5 after each (ReLU activated) fully connected layer. Since the previous steps hadn't been successful I changed the architecture to a model similar to VGG net. But with this model I still faced overfitting issues (training acc. = 0.997; validation acc. = 0.85). After many experiments with different hyperparameters (learning rate, keep prob. of dropout mechanism) I figured out that the problem arises from a wrongly implemented preprocessing step. Thus, I switched back to the original LeNet-5 with bigger layer width (higher kernel numbers) and also dropout after each fully connected layer and finally achieved the aboved mentioned accuracies. The final model is stated in the table above.

* Which parameters were tuned? How were they adjusted and why?
The learning rate was tuned. It should be big enough to provide the model the capacity to learn from data but also not too big in order not to get stuck in a local minimum of the optimization problem. Moreover, the keep probability of the dropout mechanism has to be set reasonably. The adjustment is also a trade-off. Too little and too high values aren't suitable. The number of epochs used for training is also important because one has to make sure that the optimization problem converges.
During carried out experiments a range for the learning rate between 0.01 and 0.0001 was tested. For the keep prob. values between 0.4 and 0.6 were investigated. Furthermore, the batch size has to be chosen according to memory available on the graphics card. Processing higher batch sizes results in faster training.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
For the problem of traffic sign classification convolutional layers are very efficient because they extract low level features (e.g. edges of traffic signs) at the beginning of the network and combine them to higher level features (e.g. boundaries of traffic signs) when going deeper into the network. These features can then be used for classification by the fully connected layers. Dropout layers also help with creating a successful model as the architecture is prevented from overfitting the training data which results in better generalization on new unseen images.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
In practice, comparing the performance of our model (especially on the test set) with other state-of-the-art methods in the field of taffic sign classification is a good way to evaluate the quality of our architecture.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The first image (stop sign) should be easily identified although there is the "Opel emblem" in the background. However, the convolutional neural network should be capable to abstract from the background and concentrate on the interesting part of the image.
For the second image (turn left ahead sign) it should be even easier to make a correct prediction.
The third image (yield sign) comprises also a traffic light and another sign which will make the classification more difficult.
The fourth image (slippery road sign) is partly obscured by snow. Moreover, the image consists of two other signs making the prediction a real challenge.
Finally, the fifth one (60 km/h speed limit) should again be easy to identify even though it was captured on a dull day (dark light conditions).


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction (computation in code cell 14):

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Turn left ahead     	| Turn left ahead 								|
| Yield					| Road work										|
| Slippery Road			| Ahead only      							    |
| 60 km/h	      		| No entry					 				    |


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40% (computation in code cell 15). This doesn't compare favorably to the accuracy on the test set of 0.948. Reasons for this bad performance are the wrongly classified images (image 3 and 4) which are very hard to predict because there are more than one single traffic sign. Moreover, the wrongly predicted images are very pixelized what arise from downscaling the original images to the resolution of 32x32x3 which is accepted by LeNet. However, one has to consider also the top 5 softmax probabilities for evaluation of performance. Perhaps the predicted traffic signs are contained.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for computing the top 5 softmax probabilities and the corresponding sign type are calculated in code cell 16.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.446), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .446         			| Stop sign   									| 
| .298     				| Turn left ahead 								|
| .179					| Children crossing								|
| .044	      			| Keep right					 				|
| .009				    | Yield      							        |


For the second image the model is very certain that this is a turn left ahead sign (probability of 0.826). It is indeed a turn left ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|  
| .826     				| Turn left ahead 								|
| .156					| Keep right								    |
| .009	      			| Stop					 				        |
| .006				    | Go straight or right      				    |
| .002         			| Yield   									    |


For the third image the model is relatively sure that this is a Road work sign (probability of 0.538). However, it is a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|  
| .538     				| Road work 								    |
| .218					| Children crossing								|
| .123	      			| Slippery road					 				|
| .071				    | Beware of ice/snow      				        |
| .019         			| Bicycles crossing   							|


For the fourth image the model is relatively sure that this is a Ahead only sign (probability of 0.627). However, it is a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|  
| .627     				| Ahead only 								    |
| .218					| Yield								            |
| .123	      			| Road work					 				    |
| .071				    | Priority road      				            |
| .019         			| Go straight or right   						|


For the fifth image the model is very sure that this is a No entry sign (probability of 0.954). However, it is a 60 km/h speed limit sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|  
| .954     				| No entry 								        |
| .029					| End of all speed and passing limits			|
| .005	      			| Speed limit (120km/h)					 		|
| .005				    | Speed limit (30km/h)      				    |
| .002         			| Roundabout mandatory   						|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


