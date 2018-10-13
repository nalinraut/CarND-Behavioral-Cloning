
[//]: # (Image References)

[image1]: ./assets/output.gif
[image2]: ./assets/center.jpg
[image3]: ./assets/left.jpg
[image4]: ./assets/right.jpg
[image5]: ./assets/center_flip.jpg

# Behavioral Cloning Project

![alt text][image1]
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



---



#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `datamod.py` containing the folder that augments collected data
* `train.py` containing the script to train the modified data
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `video.py` containing the script that generates a video of the car driving autoomously



### Model Architecture and Training Strategy

The model architecture summary is shown below:

The following is the model architecture summary


```
___________________________________________________________________
Layer (type)                     Output Shape          Param #                    
===================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0                        
___________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 70, 320, 3)    0                              
___________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 158, 24)   1824                    
___________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636                   
___________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248                   
___________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712                   
___________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928                   
___________________________________________________________________
flatten_1 (Flatten)              (None, 4224)          0                       
___________________________________________________________________
dense_1 (Dense)                  (None, 1164)          4917900                       
___________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0                               
___________________________________________________________________
activation_1 (Activation)        (None, 1164)          0                             
___________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500                     
___________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0                               
___________________________________________________________________
activation_2 (Activation)        (None, 100)           0                             
___________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050                       
___________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0                               
___________________________________________________________________
activation_3 (Activation)        (None, 50)            0                             
___________________________________________________________________
dense_4 (Dense)                  (None, 10)            510                        
___________________________________________________________________
activation_4 (Activation)        (None, 10)            0                               
___________________________________________________________________
dense_5 (Dense)                  (None, 1)             11                         
===================================================================
Total params: 5,171,319
Trainable params: 5,171,319
Non-trainable params: 0
___________________________________________________________________
```
#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Also the number of epochs were limited to 8 or less since the model overfitted for epochs more than 8.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. A learning rate of 1e-4 is preferred for Adam optimizer.

#### 4. Training data set.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. To capture good driving behavior, I first recorded one lap on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center. The images below show the left, center and right camera images.

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data set, I also flipped images and angles thinking that this would generalize the model. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image4]

The training parameters used are shown below:

'''
	
'''


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8. I used an adam optimizer so that manually training the learning rate wasn't necessary. The ideal learning rate for Adam Optimizer is 0.0001
