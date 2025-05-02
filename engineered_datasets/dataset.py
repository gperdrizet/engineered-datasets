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

        if isinstance(train_data, pd.DataFrame):
            self.dataset_file=dataset_file
        else:
            raise TypeError('Dataset file name is not a string.')

        if isinstance(train_data, pd.DataFrame):
            self.train_data=train_data

        else:
            raise TypeError('Train data is not a Pandas DataFrame.')

        if isinstance(train_data, pd.DataFrame) or test_data is None:
            self.test_data=test_data

        else:
            raise TypeError('Test data is not a Pandas DataFrame.')

        if isinstance(train_data, pd.DataFrame) or test_data is None:
            self.string_features=string_features

        else:
            raise TypeError('String features is not a list.')
