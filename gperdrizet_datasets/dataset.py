'''Generates variations of a dataset using a pool of feature engineering
techniques. Used for training ensemble models.'''

import h5py
import pandas as pd

class DataSet:
    '''Dataset generator class.'''

    def __init__(
            self,
            training_data_file:str,
            submission_data_file:str,
            dataset_file:str,
            string_features:list
        ):

        self.training_data_file=training_data_file
        self.submission_data_file=submission_data_file
        self.dataset_file=dataset_file
        self.string_features=string_features
