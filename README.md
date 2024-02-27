# Time Series Supervised Classification Algorithms

This project implements two univariate time series supervised classification algorithms using Python.
Overview

The implemented algorithms aim to classify time series data into predefined classes using supervised learning techniques. The project consists of two main approaches

## Deep Learning Model with 1-D Convolution
        In this approach, the time series data is windowed into fixed-size segments.
        Each window is then fed into a deep learning model with 1-D convolution layers for training and testing.
        The model learns to classify each window into predefined classes based on the features extracted from the time series data.

## Feature Extraction with tsfresh and SelectKbest
        This approach involves splitting the time series data into fixed-size windows.
        Features are extracted from each window using the tsfresh feature extraction package.
        The SelectKbest feature selection algorithm is applied to identify the optimal subset of features that minimize validation loss.
        During the testing phase, the selected features are extracted from the test data and used to predict the class labels using a trained model.

Implementation Details

    Deep Learning Model
        Implemented using TensorFlow and Keras.
        Architecture includes 1-D convolutional layers followed by dense layers for classification.
        Data preprocessing involves windowing the time series data into segments of fixed size.
        Model training is performed using labeled data, and testing evaluates the model's performance on unseen data.

    Feature Extraction with tsfresh
        Utilizes the tsfresh package for automated feature extraction from time series data.
        Features are extracted from each windowed segment of the time series.
        SelectKbest feature selection algorithm is applied to select the most relevant features based on validation performance.

Usage

    Installation
        Clone the repository to your local machine.
        Install the required dependencies using pip install -r requirements.txt.

    Data Preparation
        Prepare your time series data in a suitable format.
        Ensure that the data is labeled with the corresponding class labels.

    Training and Testing
        Choose the desired classification approach (Deep Learning Model or Feature Extraction with tsfresh).
        Follow the provided scripts or notebooks to preprocess the data, train the model, and evaluate its performance.
        Adjust hyperparameters and model architectures as needed to optimize performance.

    Evaluation:
        Evaluate the trained models on test data to assess their classification accuracy.
        Analyze the results and refine the models as necessary to improve performance.

Dependencies

    Python 3.x
    TensorFlow
    Keras
    tsfresh
    NumPy
    Pandas
    scikit-learn

