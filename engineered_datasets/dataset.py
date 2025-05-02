'''Generates variations of a dataset using a pool of feature engineering
techniques. Used for training ensemble models.'''

import h5py
import pandas as pd

class DataSet:
    '''Dataset generator class.'''

    def __init__(
            self,
            dataset_file: str,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame=None,
            string_features: list=None
        ):

        self.dataset_file=dataset_file
        self.train_data=train_data
        self.test_data=test_data
        self.string_features=string_features
