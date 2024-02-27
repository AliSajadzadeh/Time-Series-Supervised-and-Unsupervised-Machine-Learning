# Time Series Supervised Machine Learning Algorithms

This project implements two univariate time series classification algorithms using Python.

## Overview

The implemented algorithms aim to classify time series data into predefined classes using supervised learning techniques. The project consists of two main approaches:

- Deep Learning Model with 1-D Convolution
        In this approach, the time series data is windowed into fixed-size segments.
        Each window is then fed into a deep learning model with 1-D convolution layers for training and testing.
        The model learns to classify each window into predefined classes based on the features extracted from the time series data.

- Feature Extraction with tsfresh and SelectKbest
        This approach involves splitting the time series data into fixed-size windows.
        Features are extracted from each window using the tsfresh feature extraction package.
        The SelectKbest feature selection algorithm is applied to identify the optimal subset of features that minimize validation loss.
        During the testing phase, the selected features are extracted from the test data and used to predict the class labels using a trained model.

## Objective
The objective of the project is to gain a comprehensive understanding of deep learning layers and their functionality by implementing them from scratch. 
This hands-on approach helps us reinforce our understanding of the concepts learned in the deep learning course.


## Implemented Deep Learning Layers
This project includes the following deep learning layers:
- **Dense Layer:** Fully connected layer with customizable activation function.
- **Convolutional Layer:** Convolutional layer with customizable kernel size, padding, and stride.
- **Pooling Layer:** Pooling operations such as max pooling or average pooling.
- **Activation Layer:** Various activation functions such as ReLU, sigmoid, or tanh.
- **Optimization Algorithms:** SGD and ADAM algorithms
- **Loss Functions:** CrossEntropyLoss function.

## Testing
For all implemented layers and functions, unit tests have been written to verify the performance of each component. These tests cover various states and situations to ensure the robustness and correctness of the deep learning functionalities. Prior to using the implemented layers in your projects, it is essential to run these unit tests to validate their behavior.

## Conclusion
Deep learning has often been referred to as a "black box" due to the lack of transparency or interpretability in the internal workings of it. However, through the process of implementing deep learning layers from scratch in this project, we've gained invaluable insights that have illuminated many aspects of how deep learning works.This hands-on approach has allowed us to see beyond the abstraction provided by high-level deep learning frameworks and comprehend the mechanics at play.

