"""!
@brief   Two time series supervised machine learning methods: Based on deep learning and statistical extracted features
@see     https://developer.ibm.com/learningpaths/get-started-time-series-classification-api/what-is-time-series-classification
@see     https://towardsdatascience.com/time-series-classification-with-deep-learning-d238f0147d6f
@see     https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
@see     https://arxiv.org/pdf/1809.04356.pdf

"""

import pathlib
from supervised_ml import DeepLearningModel, StatisticsFeatures


class Context:
    """!
    @brief           Time series machine learning algorithms
    @date            20.10.2023
    @dependencies    Pandas, NumPy, Keras, TensorFlow, tsfresh
    @see             https://docs.python.org/3/library/functions.html#super
    """
    def __init__(self, algorithm):
        """!
        @brief  Initialize common attributes
        @param  algorithm The algorithm that we want to get result from
        """
        self.algorithm = algorithm

    def running_algorithm(self):
        """!
        @brief  Execution of selected algorithm
        @return the selected algorithm for execution
        """
        if self.algorithm == 'DeepLearningModel':
            algorithm_instance = DeepLearningModel(data_repository, result_repository)
            algorithm_instance.train_prediction()
        elif self.algorithm == 'StatisticsFeatures':
            algorithm_instance = StatisticsFeatures(data_repository, result_repository)
            algorithm_instance.train_prediction()
        else:
            raise ValueError("Invalid algorithm selected.")


if __name__ == '__main__':
    # Current project repository
    project_repository = pathlib.Path.cwd().parent
    data_repository = project_repository / 'Data'
    # If data is not in the format of csv, change the format
    data_list = list(data_repository.glob("*.csv"))
    result_repository = project_repository / 'Result'
    # Create a Context instance for each algorithm
    context_supervised = Context('DeepLearningModel')
    context_combined = Context('StatisticsFeatures')
    # Execute the selected algorithms
    context_supervised.running_algorithm()
    context_combined.running_algorithm()
    
