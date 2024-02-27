# Time Series Supervised Machine Learning Algorithms

This project implements two univariate time series classification algorithms using Python.

## Overview

The implemented algorithms aim to classify time series data into predefined classes using supervised learning techniques. The project consists of two main approaches:

- **Deep Learning model with 1-D convolution**

  In this approach, the time series data is windowed into fixed-size segments.
  Each window is then fed into a deep learning model with 1-D convolution layers for training and testing.
  The model learns to classify each window into predefined classes.
  
- **Model based on feature extraction and featre selection**
  
  This approach involves splitting the time series data into fixed-size windows.
  Features are extracted from each window using the tsfresh feature extraction package.
  The SelectKbest feature selection algorithm is applied to identify the optimal subset of features that minimize validation loss.
  During the testing phase, the selected features from training phase are extracted from the test data and used to predict the class labels using a trained model.

## Implementation Details

  - **Deep learning model**
    
    Implemented using TensorFlow and Keras.
    Architecture includes 1-D convolutional layers followed by dense layers for classification.
    Data preprocessing involves windowing the time series data into segments of fixed size.
    Model training is performed using labeled data, and testing evaluates the model's performance on unseen data.

  - **Feature extraction model**
    
    Utilizes the tsfresh package for automated feature extraction from time series data. It extracts hundreds of feature including statistical, frequency and distributional features.
    Features are extracted from each windowed segment of the time series.
    SelectKbest feature selection algorithm is applied to select the most relevant feature. The optimal window size for windowing is based on achieving the lowest validation loss.

## Usage

  - **Installation:**
    Clone the repository to your local machine.
    Install the required dependencies using pip install -r requirements.txt.

- **Data Preparation:**
  Prepare your time series data in a suitable format.
  Ensure that the data is labeled with the corresponding class labels.

- **Training and Testing:**
    Choose the desired classification approach (Deep learning model or feature extraction with tsfresh).
    Follow the provided scripts or notebooks to preprocess the data, train the model, and evaluate its performance.
    Adjust hyperparameters and model architectures as needed to optimize performance.

- **Evaluation:**
    Evaluate the trained models on test data to assess their classification accuracy.
    Analyze the results and refine the models as necessary to improve performance.

  ## Dependencies

    Python 3.x
  
    TensorFlow
  
    Keras
  
    tsfresh
  
    NumPy
  
    Pandas
  
    scikit-learn


