"""!
@brief   Time series preprocessing based on windowing approach
@see     https://albertum.medium.com/preprocessing-time-series-to-windowed-datasets-a464799b1df7
"""

import math
import numpy as np
import pandas as pd


# Assumptions
X_LABEL = 'time'
Y_LABEL = 'current'
classes = ['class1', 'class2', 'class3', 'class4']
# Assign a color for each class
label_colors = {'class1': 'blue', 'class2': 'black', 'class3': 'green', 'class4': 'red'}
# offsets considered for different classes to used overlapping windows
offset_dict = {'class1': 0.1, 'class2': 0.2, 'class3': 0.3, 'class4': 0.4}


class Windowing:
    """!
    @brief          Preprocessing based of windowing process used for time series data
    @date           20.10.2023
    @dependencies   pandas, numpy
    @see            https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html
    """

    def __init__(self, data, window_length):
        """!
        @brief  Initial setup
        @param  data Time series data used for windowing
        @param  window_length time series window size
        """
        self._data = data
        self.window_length = window_length

    def train_windowing(self):
        """!
        @brief    Partition time series data into windows
        @return   Time series windows and the label of windows
        @see      https://speechprocessingbook.aalto.fi/Representations/Windowing.html
        """
        # Separate each class
        class1 = self._data[self._data.iloc[:, 2] == 'class1']
        class2 = self._data[self._data.iloc[:, 2] == 'class2']
        class3 = self._data[self._data.iloc[:, 2] == 'class3']
        class4 = self._data[self._data.iloc[:, 2] == 'class4']
        class_dataframes = [class1, class2, class3, class4]
        window_labels = []
        data_window = np.empty((0, self.window_length))
        for item in class_dataframes[:]:
            label = item.iloc[0, item.columns.get_loc('label')]
            margin = int(offset_dict[label]*self.window_length)
            window_num = math.ceil((len(item) - margin) / (self.window_length - margin))
            for i in range(window_num):
                start_index = (self.window_length - margin) * i
                end_index = min(start_index + self.window_length, len(item))
                if i == window_num - 1:
                    start_index = end_index - self.window_length
                if len(item) >= self.window_length:
                    samples = item[start_index:end_index]
                    current_sample = samples['Measured Variable']
                    window_labels.append(label)
                    data_window = np.vstack((data_window, current_sample))
        return data_window, window_labels

    def test_windowing(self):
        """!
        @brief    Partition time series data into some windows
        @return   time series windows test data
        @see      https://speechprocessingbook.aalto.fi/Representations/Windowing.html
        """
        # Calculate the number of windows considering the size of data
        windows_num = len(self._data) // self.window_length + 1
        # Create an array for data windows
        windows = np.empty((0, self.window_length))
        for j in range(windows_num):
            start_index = j * self.window_length
            end_index = start_index + self.window_length
            if end_index > len(self._data):
                end_index = len(self._data)-1
                start_index = end_index - self.window_length
            samples = self._data[start_index:end_index]
            sample_value = samples['Measured Variable']
            windows = np.vstack((windows, sample_value))
        return windows

    def flat_windowing_test(self):
        """!
        @brief    Partition time series data into some windows in flat dataframe format
        @return   time series test windows data and their labels
        @see      https://speechprocessingbook.aalto.fi/Representations/Windowing.html
        @see      https://tsfresh.readthedocs.io/en/latest/text/data_formats.html
        """
        flat_dataframe = pd.DataFrame()
        # Calculate the number of windows considering the size of data
        windows_num = len(self._data) // self.window_length + 1
        for k in range(windows_num):
            start_index = k * self.window_length
            end_index = start_index + self.window_length
            if end_index > len(self._data):
                end_index = len(self._data) - 1
                start_index = end_index - self.window_length
            id_window = k + 1
            samples = self._data[start_index:end_index]
            samples['id'] = id_window
            flat_dataframe = pd.concat([flat_dataframe, samples])
        return flat_dataframe

    def flat_windowing_train(self, window_id):
        """!
        @brief    Partition time series data into some windows in flat dataframe format
        @param    window_id each windows and its samples should have an ID
        @return   time series windows data and their labels
        @see      https://speechprocessingbook.aalto.fi/Representations/Windowing.html
        @see      https://tsfresh.readthedocs.io/en/latest/text/data_formats.html
        """
        flat_dataframe = pd.DataFrame()
        # Calculate the number of windows considering the size of data
        windows_num = len(self._data) // self.window_length + 1
        label_windows = []
        for i in range(windows_num):
            # determine start and end indices for each window
            start_index = i * self.window_length
            end_index = start_index + self.window_length
            if end_index > len(self._data):
                end_index = len(self._data) - 1
                start_index = end_index - self.window_length
            samples = self._data[start_index:end_index]
            samples['id'] = window_id
            sample_label = samples['label']
            label = sample_label.value_counts().idxmax()
            label_windows.append(label)
            flat_dataframe = pd.concat([flat_dataframe, samples])
            id += 1
        return flat_dataframe, label_windows

