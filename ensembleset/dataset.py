'''Generates variations of a dataset using a pool of feature engineering
techniques. Used for training ensemble models.'''

from pathlib import Path
from random import choice, shuffle

import h5py
import numpy as np
import pandas as pd

import ensembleset.feature_methods as fm


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
            train_data.columns = train_data.columns.astype(str)
            self.train_data = train_data

        else:
            raise TypeError('Train data is not a Pandas DataFrame.')

        if isinstance(test_data, pd.DataFrame) or test_data is None:
            test_data.columns = test_data.columns.astype(str)
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
            'onehot_encoding': {'sparse_output': False},
            'ordinal_encoding': {
                'handle_unknown': 'use_encoded_value',
                'unknown_value': np.nan  
            }
        }

        self.engineerings={
            'poly_features': {
                'degree': [2, 3],
                'interaction_only': [True, False],
            },
            'spline_features': {
                'n_knots': [5],
                'degree': [2, 3, 4],
                'knots': ['uniform', 'quantile'],
                'extrapolation': ['error', 'constant', 'linear', 'continue', 'periodic']
            },
            'log_features':{
                'base': ['2', 'e', '10']
            },
            'ratio_features':{
                'div_zero_value': [np.nan]
            },
            'exponential_features':{
                'base': ['2', 'e']
            }
        }


    def make_datasets(self, n_datasets:int, n_features:int, n_steps:int):
        '''Makes n datasets with different feature subsets and pipelines.'''

        hdf = h5py.File('data/dataset.hdf5', 'w')

        for n in range(n_datasets):

            train_df=self.train_data.copy()
            test_df=self.test_data.copy()
            pipeline=self._generate_data_pipeline(n_steps)

            for operation, arguments in pipeline.items():
                func = getattr(fm, operation)

                if operation in self.string_encodings:
                    train_df, test_df=func(
                        train_df,
                        test_df,
                        self.string_features,
                        arguments
                    )

                else:
                    features=self._select_features(n_features, train_df)

                    train_df, test_df=func(
                        train_df,
                        test_df,
                        features,
                        arguments
                    )

            _ = hdf.create_dataset(f'train/{n}', data=np.array(train_df))
            _ = hdf.create_dataset(f'test/{n}', data=np.array(test_df))

        hdf.close()


    def _select_features(self, n_features:int, data_df:pd.DataFrame):
        '''Selects a random subset of features.'''

        features=data_df.columns.to_list()#.astype(str).to_list()
        shuffle(features)
        features=features[:n_features]

        return features


    def _generate_data_pipeline(self, n_steps:int):
        '''Generates one random sequence of feature engineering operations. Starts with
        a string encoding method if we have string features.'''

        pipeline={}

        # Choose a string encoding method, if needed
        if self.string_features is not None:
            options=list(self.string_encodings.keys())
            selection=choice(options)
            pipeline[selection]=self.string_encodings[selection]

        # Construct a random sequence of feature engineering operations
        operations=list(self.engineerings.keys())
        shuffle(operations)
        operations=operations[:n_steps]

        for operation in operations:

            pipeline[operation]={}
            parameters=self.engineerings[operation]

            for parameter, values in parameters.items():

                value=choice(values)
                pipeline[operation][parameter]=value

        return pipeline

