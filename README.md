# Gesture Recognition using 3DCNN/CNN+RNN
 We will build 2 models for gesture recognition in webcam, 3D convolution and 2d Convolution+RNN

## Problem Statement
Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

* Thumbs up:  Increase the volume
* Thumbs down: Decrease the volume
* Left swipe: 'Jump' backwards 10 seconds
* Right swipe: 'Jump' forward 10 seconds  
* Stop: Pause the movie

Each video is a sequence of 30 frames (or images).

## Dataset
The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use.
https://drive.google.com/uc?id=1ehyrYBQ5rbQQe6yL4XbLWe3FMvuVUGiL


## Preprocess
The data is in a zip file. The zip file contains a 'train' and a 'val' folder with two CSV files for the two folders. These folders are in turn divided into subfolders where each subfolder represents a video of a particular gesture. Each subfolder, i.e. a video, contains 30 frames (or images). Note that all images in a particular video subfolder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos). Hence, we will need to do some pre-processing to standardise the videos.

Each row of the CSV file represents one video and contains three main pieces of information - the name of the subfolder containing the 30 images of the video, the name of the gesture and the numeric label (between 0-4) of the video.

Our task is to train a model on the 'train' folder which performs well on the 'val' folder as well (as usually done in ML projects). The test folder for evaluation purposes has been withheld by the evaluator - our final model's performance will be tested on the 'test' set.

## Two Architectures: 3D Convs and CNN-RNN Stack
After understanding and acquiring the dataset, the next step is to try out different architectures to solve this problem. 

For analysing videos using neural networks, two types of architectures are used commonly. One is the standard CNN + RNN architecture in which we pass the images of a video through a CNN which extracts a feature vector for each image, and then pass the sequence of these feature vectors through an RNN.

The other popular architecture used to process videos is a natural extension of CNNs - a 3D convolutional network.

In this project, we will try **both** these architectures.

*Convolutions + RNN*
The conv2D network will extract a feature vector for each image, and a sequence of these feature vectors is then fed to an RNN-based network. The output of the RNN is a regular softmax (for a classification problem such as this one).

*3D Convolutional Network, or Conv3D*
3D convolutions are a natural extension to the 2D convolutions you are already familiar with. Just like in 2D conv, you move the filter in two directions (x and y), in 3D conv, you move the filter in three directions (x, y and z). In this case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is 100x100x3, for example, the video becomes a 4-D tensor of shape 100x100x3x30 which can be written as (100x100x30)x3 where 3 is the number of channels. Hence, deriving the analogy from 2-D convolutions where a 2-D kernel/filter (a square filter) is represented as (fxf)xc where f is filter size and c is the number of channels, a 3-D kernel/filter (a 'cubic' filter) is represented as (fxfxf)xc (here c = 3 since the input images have three channels). This cubic filter will now '3D-convolve' on each of the three channels of the (100x100x30) tensor.
As an example, let's calculate the output shape and the number of parameters in a Conv3D with an example of a video having 7 frames. Each image is an RGB image of dimension 100x100x3. Here, the number of channels is 3.

The input (video) is then 7 images stacked on top of each other, so the shape of the input is (100x100x7)x3, i.e (length x width x number of images) x number of channels. Now, let's use a 3-D filter of size 2x2x2. This is represented as (2x2x2)x3 since the filter has the same number of channels as the input (exactly like in 2D convs).

Now let's perform a 3D convolution using a (2x2x2) filter on a (100x100x7) matrix (without any padding and stride = 1). You know that the output size after convolution is given by: 

new_dimension = (old_dimension − filter_size + 2\*padding)/stride  + 1

In 3D convolutions, the filter convolves in three directions (exactly like it does in two dimensions in 2D convs), so you can extend this formula to the third dimension as well. You get the output shape as:

((100−2)/1+1,(100−2)/1+1, (7−2)/1+1)) = (99,99,6)

Thus, the output shape after performing a 3D conv with one filter is (99x99x6). Now, if we do (say) 24 such 3D convolutions, the output shape will become (99x99x6)x24. Hence, the new number of channels for the next Conv3D layer becomes 24. This is very similar to what happens in conv2D.

Now let's calculate the number of trainable parameters if the input shape is (100x100x7)x3 and it is convolved with 24 3D filters of size (2x2x2) each, expressed as (2x2x2)x3 to give an output of shape (99x99x6)x24. Each filter will have 2x2x2 = 8 trainable parameters for each of the 3 channels. Also, here we consider that there is one bias per filter. Hence, each filter has 8x3 + 1  = 25 trainable parameters. For 24 such filters, we get 25x24 = 600 trainable parameters.

There are a few key things to note about the conv-RNN architecture:

1. We can use transfer learning in the 2D CNN layer rather than training your own CNN 
2. GRU can be a better choice than an LSTM since it has lesser number of gates (and thus parameters)

## Understanding Generators
Now that we understand the two types of architectures to experiment with, let's discuss how to set up the data ingestion pipeline. As we already know, in most deep learning projects we need to feed data to the model in batches. This is done using the concept of generators. 

Creating data generators is probably the most important part of building a training pipeline. Although libraries such as Keras provide builtin generator functionalities, they are often restricted in scope and you have to write your own generators from scratch. For example, in this problem, we need to feed batches of videos, not images. Similarly, in an entirely different problem such as 'music generation,' we may need to write generators which can create batches of audio files. 

We will create a data generator from scratch.
The generator yields a batch of data and 'pauses' until the fit_generator calls next(). Note that in Python-3, this functionality is implemented by _next_(). 

We have to experiment your model with the following:
* number of images to be taken per video/sequence
* cropping the images
* resizing the images
* normalizing the images

## Goals of this Project
In this project, we will build a model to recognise 5 hand gestures.

We need to accomplish the following in the project:

1. Generator:  The generator should be able to take a batch of videos as input without any error. Steps like cropping, resizing and normalization should be performed successfully.
2. Model: Develop a model that is able to train without any errors which will be judged on the total number of parameters (as the inference(prediction) time should be less) and the accuracy achieved. As suggested by Snehansu, start training on a small amount of data and then proceed further.

Save the model parameters in the .h5 file for reproducing model without retraining.

## Results

The results are given in the Sample write up doc file.
h5 file for 3D CNN is attached.

2D CNN + RNN h5 files as they are quite big:
Link of h5 file with 89% valid_accuracy -> 
https://drive.google.com/open?id=1xhVJrXuFQ8xNO9G-f9hhD-eo3APnF7K7 
Link of h5 file with 91% valid_accuracy ->
https://drive.google.com/open?id=1iF9zf3sBbZ0ZnQnlCrizfJWWIoV_6UXq 
