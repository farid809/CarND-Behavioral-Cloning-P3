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

The initial model I used  based on nVidia model which consisted of lambda normalization layer followed by 5 convulutional layer followed by 4 fully connected layers. The model also includes RELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer. I attempted to train this model and tune different hyperparameters  However the model didn't generalize enough to drive centered on track 1.


#### 2. Overfitting

I included 2 dropout layer before and after the fully connected layer as illustrated in the model visualization below to reduce overfitting. However, I was able to get better results by introducing SpatialDropout2D layer after each convultional layer

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.



#### 3. Model parameter tuning

I initially tried to use different learning rates, however, I got a better result by using Adam optimizer, so the learning rate was not tuned manually. I also, experimented with different batch sizes and EPOCH counts.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 


My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.1
### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...


My Final model is based on nVidia model and consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 


Here is a visualization of the architecture using Keras plot_model from keras.utils library 

![alt text][image1]




### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image8]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... The below animated GIF shows what a recovery looks like starting from from the edge to the center of the road


![alt text][image9]

Then I repeated this process on track two in order to get more data points.

![alt text][image8a]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


Finally, I tested the model using sample input image and visualized the output of Frist and 2nd layers. Below are sample output from Lambda and Cropping2D layers respectively. 

Original Input Image

![alt text][image10]

Output From Lambda Layer (Layer1)

![alt text][image11]

Output From Cropping2D Layer (Layer2)

![alt text][image12]



## [References]

1. [Keras Visualization] https://keras.io/visualization/
2. [Keras Fit Generator] https://keras.io/models/model/#fit_generator
3. [nVidia Model] https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
