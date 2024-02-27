"""!
@brief  Univariate Time Series Supervised Machine Learning Algorithms
@see    https://developer.ibm.com/learningpaths/get-started-time-series-classification-api/what-is-time-series-classification
@see    https://towardsdatascience.com/time-series-clustering-deriving-trends-and-archetypes-from-sequential-data-bb87783312b4
"""

import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsfresh import extract_features
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense, BatchNormalization, Dropout
from preprocessing import Windowing

# Assumptions
X_LABEL = 'Time [s]'
Y_LABEL = 'Measured Variable'
classes = ['class1', 'class2', 'class3', 'class4']
# Map each class to a number
class_mapping = {'class1': 0, 'class2': 1, 'class3': 2, 'class4': 3}
# Assign a color for each class
label_colors = {'class1': 'blue', 'class2': 'black', 'class3': 'green', 'class4': 'red'}


def accuracy(real_label, predicted_label):
    """!
    @brief     Calculation of ACC and F1-Score having real and predicted labels
    @param     real_label True labels of each sample
    @param     predicted_label Label predicted by a trained model for each sample
    @return    ACC, F1_Score and confusion matrix for all classes
    @ see      https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
    """
    # Confusion matrix
    confusion_mat = confusion_matrix(real_label, predicted_label, labels=classes)
    precision = np.diag(confusion_mat) / np.sum(confusion_mat, axis=0)
    recall = np.diag(confusion_mat) / np.sum(confusion_mat, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    true_pos = np.diag(confusion_mat)
    # Calculation of the sum of negative predictions and true positive
    all_samples = np.sum(confusion_mat, axis=1)
    class_accuracy = true_pos / all_samples
    return f1_score, class_accuracy, confusion_mat


class DeepLearningModel:
    """!
    @brief          Time series classification based on deep learning model and 1-d convolution
    @date           20.10.2023
    @dependencies   Keras, TensorFlow
    @see            https://www.kaggle.com/code/mersico/understanding-1d-2d-and-3d-convolution-network
    @see            https://tigerprints.clemson.edu/cgi/viewcontent.cgi?article=3918&context=all_theses
    """

    def __init__(self, path_data, path_result):
        """!
        @brief  Initial setup of class instance
        @param  path_data which contains data repository
        @param path_result where the result of algorithm is going to be saved
        """
        self._path_data = path_data
        self._path_result = path_result

    def train_prediction(self):
        """!
        @brief  Train a deep leaning model considering different window size for train data
        """
        # Different window length for time series windowing
        window_length = [500, 700, 900, 1100, 1900, 2200, 28000, 4000, 5500]
        self.model_train(window_length)

    def model_train(self, window_lengths):
        """!
        @brief    Train DL model based on 1-d conclusion and time series windows
        @param    window_lengths A list containing different window sizes
        @return   A saved trained model using train data and a saved train information procedure
        @see      https://www.tensorflow.org/js/guide/train_models
        @see      https://www.tensorflow.org/api_docs/python/tf/keras/metrics/categorical_crossentropy
        """
        train_repository = pathlib.Path(self._path_data) / 'train'
        annotated_train_data = list(train_repository.glob("*csv"))
        validation_repository = pathlib.Path(self._path_data) / 'validation'
        annotated_val_data = list(validation_repository.glob("*csv"))
        best_val_acc = 0
        best_window_length = 0
        best_val_loss = float('inf')
        for window in window_lengths:
            # Get train data and train label after windowing
            x_train, y_train = self.window_train_format(annotated_train_data, window)
            # Get validation data and validation label after windowing
            x_val, y_val = self.window_train_format(annotated_train_data, window)
            # Create a deep learning model based on 1-d convolution
            dl_model = self.model(window, len(classes))
            # Compile the model and setting model parameters
            dl_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            epochs = 100
            for epoch in range(epochs):
                history = dl_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=16)
                current_val_loss = history.history['val_loss'][0]
                current_val_acc = history.history['val_accuracy'][0]
                if current_val_loss <= best_val_loss:
                    train_acc = history.history['accuracy'][0]
                    train_loss = history.history['loss'][0]
                    best_val_loss = current_val_loss
                    best_val_acc = current_val_acc
                    best_window = window
                    # Save the model with lower validation loss to be used in inference phase
                    save_model(dl_model, 'model_trained.h5')
        test_repository = pathlib.Path(self._path_data) / 'test'
        test_repository_result = test_repository / 'Time Series Deep Learning'
        # Create a path to save the prediction result for test data
        if not test_repository_result.exists():
            test_repository_result.mkdir(parents=True)
        # Save the train information
        train_information = {
            "train_acc": {
                "value": train_acc,
                "comment": "Classification accuracy of train data while lowest validation loss is achieved"
            },

            "train_loss": {
                "value": train_loss,
                "comment": "Classification loss of train data while lowest validation loss is achieved"
            },

            "best_val_acc": {
                "value": best_val_acc,
                "comment": "Validation accuracy corresponding to lowes validation loss"
            },

            "val_loss": {
                "value": best_val_loss,
                "comment": "lowest validation loss through all training epochs"
            },
        }
        # Save test result as a json file
        output_train_result = test_repository_result / "train_information.json"
        # Save dictionary as JSON file
        with open(output_train_result, 'w', encoding="utf-8") as file:
            json.dump(train_information, file, indent=4)
        # Test the trained model using the saved trained model and the window size giving lowest validation loss
        self.model_test(best_window)

    def model_test(self, window_size) -> plt:
        """!
        @brief  Inference of trained model using test data
        @param window_size The length of the window corresponding with the lowest validation loss
        @return  Figures and result of time series classification model for test data
        @see    https://www.tensorflow.org/tutorials/images/cnn
        """
        test_repository = pathlib.Path(self._path_data) / 'test'
        test_data = list(test_repository.glob("*csv"))
        test_result = {}
        # Create a path where test result is saved
        result_path = self._path_result / 'DL supervised model'
        if not result_path.exists():
            result_path.mkdir(parents=True)
        for data_name in test_data:
            x_test, true_test_label = self.window_test_format(data_name, window_size)
            # Load the trained model saved in training phase
            loaded_model = load_model('model_trained.h5')
            # Trained model predicts the label of each test time window
            prediction = loaded_model.predict(x_test)
            predicted_window_labels = np.argmax(prediction, axis=1)
            # Map the label of each window to samples within that window
            predicted_sample_labels = []
            window_num = len(true_test_label) // window_size + 1
            for id_window in range(window_num):
                label = predicted_window_labels[id_window]
                predicted_sample_labels.extend([label] * window_size)
            predicted_sample_labels = predicted_sample_labels[:len(true_test_label)]
            predict_test_label = [key for value in predicted_sample_labels for key, val in class_mapping.items() if val == value]
            f1_score, acc, ms = accuracy(np.array(true_test_label), np.array(predict_test_label))
            test_result = {'f1_score': f1_score.tolist(), 'ACC': acc.tolist(), 'Confusion Matrix': ms.tolist()}
            figure = self.test_figure (data_name, predict_test_label)
            figure.savefig(f'{result_path / str(data_name.stem)}.png')
        test_information = {
            "Window": {
                "value": window_size,
                "comment": "The size of window used to partition test data"
            },

            "Test Result": {
                "value": test_result,
                "comment": "Quantitative evaluation of test data"
            },
            }
        # Save dictionary as json file
        with open(result_path, 'w', encoding="utf-8") as file:
            json.dump(test_information, file, indent=4)

    def window_train_format(self, data_list, window):
        """!
        @brief    Bring all the data in the list in windowing format
        @param    data_list a list containing all data
        @param    window the length of window
        @return   All window data and their corresponding label
        @see      https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        """
        # Bring time series data in an array which the row is the number of windows and column is size of window
        window_array = np.empty((0, window))
        # Push the Label of each window to a list
        label_window = []
        # Put all window data under each other in an array
        for _, data in enumerate(data_list):
            annotated_data = pd.read_csv(data)
            # Annotated train data windowing
            window_object = Windowing(annotated_data, window)
            window_samples, label = window_object.train_windowing()
            window_array = np.vstack((window_array, window_samples))
            label_window.extend(label)
        x = window_array.reshape(window_array.shape[0], window_array.shape[1], 1)
        # Convert labels to numbers based on label_mapping
        window_labeled = [class_mapping[cls] for cls in label_window]
        # Bring the label of each class to one-hot encoding format
        y = tf.keras.utils.to_categorical(window_labeled, len(classes))
        return x, y

    def window_test_format(self, data, window):
        """!
        @brief    Bring all the data in the list in windowing format
        @param    data The name of test data
        @param    window the length of window
        @return   The windows of the each test data and the sample true labels
        @see      https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        """
        # Bring time series data in an array which the row is the number of windows and column is size of window
        window_array = np.empty((0, window))
        # Put all window data under each other in an array
        annotated_test_data = pd.read_csv(data)
        # Annotated train data windowing
        window_object = Windowing(annotated_test_data, window)
        window_samples = window_object.test_windowing()
        window_array = np.vstack((window_array, window_samples))
        x = window_array.reshape(window_array.shape[0], window_array.shape[1], 1)
        # Get true labels of the samples test
        y = annotated_test_data['label']
        return x, y

    def model(self, window_length, num_classes):
        """!
        @brief    Create a deep learning convolution model
        @param    window_length the length of window
        @param    num_classes the numbers of data classes
        @return   A deep leaning model
        @see      https://www.tensorflow.org/guide/keras/sequential_model
        """
        dl_model = Sequential()
        dl_model.add(Conv1D(56, 3, activation='relu', input_shape=(window_length, 1)))
        dl_model.add(BatchNormalization())
        dl_model.add(AveragePooling1D(6))
        dl_model.add(Conv1D(128, 3, activation='relu'))
        dl_model.add(BatchNormalization())
        dl_model.add(Dropout(0.6))
        dl_model.add(AveragePooling1D(6))
        dl_model.add(Flatten())
        dl_model.add(Dense(180, activation='relu'))
        dl_model.add(Dense(40, activation='relu'))
        dl_model.add(Dense(num_classes, activation='softmax'))
        return dl_model

    def test_figure (self, data_name, labels) -> plt:
        figure = plt.figure(figsize=(12, 8))
        colors = [label_colors[label] for label in labels]
        data = pd.read_csv(data_name)
        plt.scatter(data['time'], data['Measured Variable'], c=colors)
        legend_handles = [plt.Line2D([0], [0], marker='o', color='r', markerfacecolor=color)
                          for color in label_colors.values()]
        plt.legend(legend_handles, labels, fontsize='8', bbox_to_anchor=(1.2, 1.4), fancybox=True, framealpha=1)
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        return figure


class StatisticsFeatures:
    """!
    @brief          Time series supervised machine learning algorithm based on statistical extracted feature
    @date           20.10.2023
    @dependencies   tsfresh, SelectKBest
    @see            https://www.sciencedirect.com/science/article/pii/S2352711020300017
    @see            https://towardsdatascience.com/time-series-feature-extraction-on-really-large-data-samples-b732f805ba0e
    """

    def __init__(self, path_data, path_result):
        """!
        @brief  Initial setup of class instance
        @param  path_data which contains data repository
        @param  path_result where the result of algorithm is going to be saved
        """
        self._path_data = path_data
        self._path_result = path_result

    def train_prediction(self, phase='Train'):
        """!
        @brief  Train a model based on extracted features considering different window size for train data
        @param phase Determines at which phase the model is
        """
        if phase == 'Train':
            # Different window length for time series windowing
            window_length = [500, 700, 900, 1100, 1900, 2200, 28000, 4000, 5500]
            # The list of number of selected features for training
            feature_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 20]
            self.model_train(window_length, feature_num)

    def feature_extraction(self, window, num_features):
        """!
        @brief    Data windowing and then transform data to another space with different features
        @param    window The size of window used in data preparation windowing
        @param    num_features The number of selected features for train a model
        @return   All data and information of the data required for the model training
        @see      https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
        """
        # Get the train data used in model training
        train_repository = pathlib.Path(self._path_data) / 'Train'
        train_data = list(train_repository.glob("*csv"))
        # Get the validation data used in the process of optimal hyper parameters selection like optimal window size
        val_repository = pathlib.Path(self._path_data) / 'Validation'
        val_data = list(val_repository.glob("*csv"))
        # Annotated validation data windowing
        val_data_window, val_label_window = self.train_data_windowing(val_data, window)
        # Annotated train data windowing
        train_data_window, train_label_window = self.train_data_windowing(train_data, window)
        # For feature extraction by tsfresh the label of windows are not needed
        train_data_window = train_data_window.drop('label', axis=1)
        flat_train_dataframe = train_data_window.reset_index(drop=True)
        train_features = extract_features(flat_train_dataframe, column_id='id', column_sort='time')
        # Remove feature with value inf and undefined
        train_features = train_features.dropna(axis=1, how='any', inplace=False)
        train_features = train_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any', inplace=False)
        # Get the nam of features
        features_names = train_features.columns.tolist()
        train_features = train_features.values
        # Select the the best features among hundreds of features extracted by tsfresh
        feature_indices = self.feature_selection(train_features, np.array(train_label_window), num_features)
        # Get the name of top selected features based on time windows
        features_names = [features_names[i] for i in feature_indices]
        # Form train data using selected features
        x_train = train_features[features_names].values
        # Form validation data using selected features
        val_data_window = val_data_window.drop('label', axis=1)
        val_data_window = val_data_window.reset_index(drop=True)
        val_features = extract_features(val_data_window, column_id='id', column_sort='time')
        val_features = val_features.dropna(axis=1, how='any', inplace=False)
        val_features = val_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any', inplace=False)
        val_features = val_features.values
        # Form validation data using selected features
        x_val = val_features[features_names].values
        return features_names, x_train, train_label_window, x_val, val_label_window

    def train_data_windowing(self, data, window):
        """!
        @brief    Bring the data in some windows and each window has an id and a label
        @param    data Time series data list
        @param    window The size of window used in data preparation windowing
        @return   Time series data in the windowing format and the label of each window
        @see      https://speechprocessingbook.aalto.fi/Representations/Windowing.html
        """
        # Create a list containing the label of time series windows
        window_label = []
        flat_dataframe = pd.DataFrame()
        # Assign an id to each window starting from 1, required for tsfresh feature extraction
        window_id = 1
        for index in data:
            annotated_data = pd.read_csv(index)
            window_object = Windowing(annotated_data, window)
            data_window, labels = window_object.flat_windowing_train(window_id)
            flat_dataframe = pd.concat([flat_dataframe, data_window])
            window_label.extend(labels)
            window_id = + len(window_label) + 1
        return flat_dataframe, window_label


    def test_data_windowing(self, data, window):
        """!
        @brief    Bring the data in some windows and each window has a label
        @param    data Time series data
        @param    window The size of window used in data preparation windowing
        @return   Time series data in the windowing format and the label of each window
        @see      https://speechprocessingbook.aalto.fi/Representations/Windowing.html
        """
        test_data = pd.read_csv(data)
        window_object = Windowing(test_data, window)
        data_window = window_object.flat_windowing_test()
        return data_window

    def model_train(self, window_list, feature_list):
        """!
        @brief    Train DL model based on 1-d conclusion and time series windows
        @param    window_list A list containing different window sizes
        @param    feature_list Different number of selected features for training
        @return   A saved trained model, optimal window size and number of features
        @see      https://www.tensorflow.org/js/guide/train_models
        @see      https://www.tensorflow.org/api_docs/python/tf/keras/metrics/categorical_crossentropy
        """
        max_val_acc = 0
        best_window_size = 0
        optimal_num_features = 0
        min_val_loss = float('inf')
        best_features = None
        for window in window_list:
            for k in feature_list:
                features_names, x_train, label_window_train, x_val, label_window_val = self.feature_extraction(window,k)
                # Map labels to numbers for model training
                val_label = [class_mapping[category] for category in label_window_val]
                y_val = tf.keras.utils.to_categorical(val_label, len(classes))
                train_label = [class_mapping[category] for category in label_window_train]
                y_train = tf.keras.utils.to_categorical(train_label, len(classes))
                # Create a fully connected deep learning model for classification
                model = self.classifier(len(x_train[0]), len(classes))
                # Set model parameters
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                # Model training
                epochs = 100
                for epoch in range(epochs):
                    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=16)
                    val_loss = history.history['val_loss'][0]
                    val_acc = history.history['val_accuracy'][0]
                    if val_loss <= min_val_loss:
                        max_val_acc = val_acc
                        best_window_size = window
                        optimal_num_features = k
                        min_val_loss = val_loss
                        train_acc = history.history['accuracy'][0]
                        train_loss = history.history['loss'][0]
                        save_model(model, 'model_trained.h5')
                        best_features = features_names
        test_path = pathlib.Path(self._path_data) / 'Test'
        test_repository = test_path / 'Test_Result'
        # Create the path where result of test data is saved
        if not test_repository.exists():
            test_repository.mkdir(parents=True)
        train_information = {
            "Minimum validation classification loss": {
                "value": min_val_loss,
                "comment": "The minimum validation loss achieved during training"
            },
            "Validation ACC": {
                "value": max_val_acc,
                "comment": "The validation accuracy corresponding to lowest validation loss"
            },
            "Train Loss": {
                "value": train_loss,
                "comment": "The train classification loss corresponding to the lowest validation loss"
            },
            "Train ACC": {
                "value": train_acc,
                "comment": "The train classification accuracy corresponding to the lowest validation loss"
            },
            "Optimal window length": {
                "value": best_window_size,
                "comment": "The window length to achieve the lowest validation loss"
            },
            "Optimal number of selected features": {
                "value": optimal_num_features,
                "comment": "Optimal number of selected features "
            },
            "Optimal feature names": {
                "value": best_features,
                "comment": "The name of optimal selected features"
            },
        }
        # Save test result
        train_result = test_repository / "train_ef.json"
        with open(train_result, 'w', encoding="utf-8") as file:
            json.dump(train_information, file, indent=4)
        # Evaluate the trained model using optimal features and window size
        self.model_test(best_features, best_window_size)

    def model_test(self, feature_names, window_length):
        """!
        @brief    Inference trained model using test data
        @param    feature_names The name of best selected features in training phase
        @param    window_length The length of the window (must be the same size in training)
        @return   A figure showing the result of trained model for all classified samples (for all data)
        @see      https://www.tensorflow.org/tutorials/images/cnn
        """
        # Get the test data
        test_repository = pathlib.Path(self._path_data) / 'Test'
        test_data_list = list(test_repository.glob("*csv"))
        test_result = {}
        # Create a path to save the result of test data
        test_result_path = self._path_result / 'Test Result'
        if not test_result_path.exists():
            # Create the result path for test data
            test_result_path.mkdir(parents=True)
        for data in test_data_list:
            # Load test data
            test_data = pd.pd.read_csv(data)
            # Calculate the number of windows considering the size of data
            windows_num = len(test_data) // window_length + 1
            x_test = self.test_data_windowing(data, window_length)
            # Extract all features of test data
            features = extract_features(x_test, column_id='id', column_sort='time')
            # Exclude the features which are not selected in training
            x_test_selected = features[feature_names].values
            # Load the trained model
            model_loaded = load_model('trained_model.h5')
            # Use the loaded model for predictions
            predictions = model_loaded.predict(x_test_selected)
            predicted_classes = np.argmax(predictions, axis=1)
            # Assign the label of each window to sample within that window
            predicted_labels = []
            for window_id in range(windows_num):
                label = predicted_classes[window_id]
                predicted_labels.extend([label] * window_length)
            predicted_labels = predicted_labels[:len(test_data)]
            predicted_labels = [key for value in predicted_labels for key, val in class_mapping.items() if val == value]
            colors = [label_colors[label] for label in predicted_labels]
            plt.scatter(test_data['time'], test_data['Measured Variable'], c=colors)
            plt.xlabel(X_LABEL)
            plt.ylabel(Y_LABEL)
            true_label = test_data['label']
            f1_score, class_acc, ms = accuracy(np.array(true_label), np.array(predicted_labels))
            # Save the test result for each data
            test_result = {'f1_score': f1_score.tolist(), 'class_acc': class_acc.tolist(),
                           'confusion matrix': ms.tolist()}
            test_result[data.stem] = test_result
        test_information = {
            "window_length": {
                "value": window_length,
                "comment": "The list of windows with different sizes"
            },
            "selected features from training phase": {
                "value": features,
                "comment": "the best features selected in training phase for testing"
            },
            "Result": {
                "value": test_result,
                "comment": "result of each test data individually"
            },
        }
        output_result = test_result_path / "Test Result"
        with open(output_result, 'w', encoding="utf-8") as file:
            json.dump(test_information, file, indent=4)

    def feature_selection(self, features, target, num_best_features):
        """!
        @brief     Select the top features
        @param     features All extracted features
        @param     target The label of each time series window
        @param     num_best_features The number of best features that are going to be selected
        @return    The feature selected indices
        @see       https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
        """
        feature_selector = SelectKBest(score_func=mutual_info_classif, k=num_best_features)
        feature_selector.fit_transform(features, target)
        # Get the features indices
        feature_indices = feature_selector.get_support(indices=True)
        return feature_indices

    def classifier(self, length, num_classes):
        """!
        @brief    Create a fully-connected layer for classification
        @param    length the number of features
        @param    num_classes the numbers of data classes
        @return   A deep leaning model
        @see      https://www.tensorflow.org/guide/keras/sequential_model
        """
        model = Sequential()
        model.add(Dense(500, activation='relu', input_shape=length))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())
        return model