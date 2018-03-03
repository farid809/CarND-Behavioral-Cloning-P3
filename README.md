# Behaviorial Cloning Project

[//]: # (Image References)

[image1]: ./images/model-architecture.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./images/image-original.png "Normal Image"
[image7]: ./images/image-flipped.png "Flipped Image"
[image8]: ./images/track1.gif "Track 1 Animation"
[image8a]: ./images/track2run1.gif "Track 2 Animation"
[image9]: ./images/recovery.gif "Recovery Animation"


[image10]: ./images/test_input.png "Test Input Image"
[image11]: ./images/lambda_out.png "Lambda Layer ouptut"
[image12]: ./images/Cropping2d_out.png "Cropping2D output"

[image13]: ./images/hist.PNG "Data Set Histogram"
[image14]: ./images/hist_augmented.PNG "Augmented Histogram"
[image15]: ./images/model_loss.PNG "Model Validation Loss"


Overview
---


In this project, I'm using deep neural networks and convolutional neural networks to clone driving behavior. I provided below step by step guide showing the approach i used to to train, validate and test a model using Keras. The model will output a steering angle to a simulated autonomous vehicle. A simulator are also used to gather the training data by capturing the steering angle and speed of a human drive (behavior cloning).

Project Files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (README.md markdown)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

* Keras



---
### Files Submitted 

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

#### 2. Model Training and driving code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. How to use submitted code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Data Collection

Provided below is the strategy I've followed to collect training data. 
* Five laps driving smoothly centered in track1 (CW)
* Five laps driving centered in track1 (CCW)
* Two laps of recovery driving from the sides
* Two laps focusing on driving smoothly around curves
* Five laps driving centered in track2



## Model Architecture and Training Strategy

### 1. Solution Design Approach


#### 1. Model Architecture

The initial model I used  based on nVidia model which consisted of lambda normalization layer followed by 5 convulutional layer followed by 4 fully connected layers. The model also includes RELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer. 


#### 2. Overfitting

I included 2 dropout layer before and after the fully connected layer as illustrated in the model visualization below to reduce overfitting. However, I was able to get better results by introducing SpatialDropout2D layer after each convultional layer

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.



#### 3. Model parameter tuning

I initially tried to use different learning rates, however, I got a better result by using Adam optimizer, so the learning rate was not tuned manually. I also, experimented with different batch sizes and EPOCH counts. 




### 4. Final Model Architecture


My Final model is based on nVidia model and which consisted of lambda normalization layer followed by 5 convulutional layer followed by 4 fully connected layers. The model also includes RELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer. Also, I choose to include dropout layer after convulutional layer and before 2 of the fully connected layers as illustrated in the architecture diagram below. 

Here is a visualization of the architecture using Keras plot_model from keras.utils library 

![alt text][image1]




### 5. Creation of the Training Set & Training Process


#### Image Preprocessing
1. Eliminated data with near-zero steering angle to balance the data set.
2. Used Lambda Layer to help with data normalization.
3. Used Cropping2D layer to remove area's that don't help the model.

#### Data Augmenation
My strategy for augmenting the data set included recording more laps by driving in both tracks and performing image manipulation by flipping the images. To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image8]

![alt text][image13]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to  recover from either side of the road. The below animated GIF shows what a recovery looks like starting from from the edge to the center of the road


![alt text][image9]

Then I repeated this process on track two in order to get more data points.

![alt text][image8a]

To augment the data sat, I also flipped images and angles thinking that this would help balancing the data set and produce a generalized model that is not biased toward driving either sid For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

![alt text][image14]

After the collection process, I had 55806 number of data points. I then preprocessed this data by eliminated data with near-zero steering angle to balance the data set. I randomly shuffled the data set and put 20% of the data into a validation set by passing the following parameters to model.fit (validation_split=0.20, shuffle=True). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. The ideal number of epochs was 40 as validation loss stopped improving and starting increasing.  However this model was only able to run in track1 and wasn't able to drive in track2. I kept experiementing with different number of EPOCHS until i was able to produce a model that is able to driving in both tracks after 120 EPOCHS. Also, I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image15]


Finally, I tested the model using sample input image and visualized the output of Frist and 2nd layers. Below are sample output from Lambda and Cropping2D layers respectively. 

Original Input Image

![alt text][image10]

Output From Lambda Layer (Layer1)

![alt text][image11]

Output From Cropping2D Layer (Layer2)

![alt text][image12]

At the end of the process, the vehicle is able to drive autonomously on both tracks without leaving the  road.

## [References]

1. [Keras Visualization] https://keras.io/visualization/
2. [Keras Fit Generator] https://keras.io/models/model/#fit_generator
3. [nVidia Model] https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
