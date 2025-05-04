'''Generates variations of a dataset using a pool of feature engineering
techniques. Used for training ensemble models.'''

from pathlib import Path
from random import choice, shuffle

import h5py
import pandas as pd

class DataSet:
    '''Dataset generator class.'''

    def __init__(
            self,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame = None,
            string_features: list = None
        ):

        # Type check the user arguments and assign them to attributes
        if isinstance(train_data, pd.DataFrame):
            self.train_data = train_data

        else:
            raise TypeError('Train data is not a Pandas DataFrame.')

        if isinstance(test_data, pd.DataFrame) or test_data is None:
            self.test_data = test_data

        else:
            raise TypeError('Test data is not a Pandas DataFrame.')

        if isinstance(string_features, list) or string_features is None:
            self.string_features = string_features

        else:
            raise TypeError('String features is not a list.')

        # Create the HDF5 output
        Path('data').mkdir(parents=True, exist_ok=True)

        with h5py.File('data/dataset.hdf5', 'a') as hdf:
            _ = hdf.require_group('train')
            _ = hdf.require_group('test')

        # Define the feature engineering pipeline operations
        self.string_encodings={
            'onehot_encoding': {},
            'ordinal_encoding': {}
        }

        self.engineerings={
            'poly_features': {
                'degree': [2, 3],
                'interaction_only': [True, False],
            },
            'spline_features': {
                'n_knots': [3, 4, 5],
                'degree': [2, 3, 4],
                'knots': ['uniform', 'quantile'],
                'extrapolation': ['error', 'constant', 'linear', 'continue', 'periodic']
            }
        }


    def _generate_data_pipeline(self):
        '''Generates one random sequence of feature engineering operations. Starts with
        a string encoding method if we have string features'''

        pipeline={}

        if self.string_features is not None:
            options=list(self.string_encodings.keys())
            selection=choice(options)
            pipeline[selection]=self.string_encodings[selection]

        operations=list(self.engineerings.keys())
        shuffle(operations)

        for operation in operations:

            pipeline[operation]={}
            parameters=self.engineerings[operation]

            for parameter, values in parameters.items():

                value=choice(values)
                pipeline[operation][parameter]=value

        return pipeline

