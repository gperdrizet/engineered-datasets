'''Generates variations of a dataset using a pool of feature engineering
techniques. Used for training ensemble models.'''

from pathlib import Path
from random import choice, shuffle

import h5py
import numpy as np
import pandas as pd

import ensembleset.feature_engineerings as engineerings
import ensembleset.feature_methods as fm


class DataSet:
    '''Dataset generator class.'''

    def __init__(
            self,
            label: str,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame = None,
            string_features: list = None
        ):

        # Check user argument types
        type_check = self.check_argument_types(
            label,
            train_data,
            test_data,
            string_features
        )

        # If the type check passed, assign arguments to attributes
        if type_check is True:
            self.label = label
            self.train_data = train_data.copy()
            self.test_data = test_data.copy()
            self.string_features = string_features

        # Enforce string type on DataFrame columns
        self.train_data.columns = self.train_data.columns.astype(str)

        if self.test_data is not None:
            self.test_data.columns = self.test_data.columns.astype(str)

        # Retrieve and assign the training labels, set NAN if they don't exist
        # then remove them from the training data
        if self.label in self.train_data.columns:
            self.train_labels=np.array(self.train_data[label])
            self.train_data.drop(self.label, axis=1, inplace=True)

        else:
            self.train_labels=[np.nan] * len(self.train_data)

        # Retrieve and assign the testing labels, set NAN if they don't exist
        # then remove them from the training data
        if self.label in self.test_data.columns:
            self.test_labels=np.array(self.test_data[label])
            self.test_data.drop(self.label, axis=1, inplace=True)

        else:
            self.test_labels=[np.nan] * len(self.test_data)

        # Create the HDF5 output
        Path('data').mkdir(parents=True, exist_ok=True)

        # Create groups for training and testing datasets
        with h5py.File('data/dataset.h5', 'a') as hdf:

            _ = hdf.require_group('train')
            _ = hdf.require_group('test')

        # Add the training and testing labels
        with h5py.File('data/dataset.h5', 'w') as hdf:
            _ = hdf.create_dataset('train/labels', data=self.train_labels)
            _ = hdf.create_dataset('test/labels', data=self.test_labels)

        # Define the feature engineering pipeline methods
        self.string_encodings=engineerings.STRING_ENCODINGS
        self.numerical_methods=engineerings.NUMERICAL_METHODS


    def make_datasets(self, n_datasets:int, n_features:int, n_steps:int):
        '''Makes n datasets with different feature subsets and pipelines.'''

        with h5py.File('data/dataset.h5', 'a') as hdf:

            # Generate n datasets
            for n in range(n_datasets):

                print(f'\nGenerating dataset {n+1} of {n_datasets}')

                # Take a copy of the training and test data
                train_df = self.train_data.copy()
                test_df = self.test_data.copy()

                # Generate a data pipeline
                pipeline = self._generate_data_pipeline(n_steps)

                # Loop on and apply each method in the pipeline
                for method, arguments in pipeline.items():

                    print(f' Applying {method}')
                    func = getattr(fm, method)

                    if method in self.string_encodings:
                        train_df, test_df = func(
                            train_df,
                            test_df,
                            self.string_features,
                            arguments
                        )

                    else:
                        features = self._select_features(n_features, train_df)

                        train_df, test_df = func(
                            train_df,
                            test_df,
                            features,
                            arguments
                        )

                # Save the results to HDF5 output
                _ = hdf.create_dataset(f'train/{n}', data=np.array(train_df))
                _ = hdf.create_dataset(f'test/{n}', data=np.array(test_df))


    def _select_features(self, n_features:int, data_df:pd.DataFrame):
        '''Selects a random subset of features.'''

        features = data_df.columns.to_list()
        shuffle(features)
        features = features[:n_features]

        return features


    def _generate_data_pipeline(self, n_steps:int):
        '''Generates one random sequence of feature engineering operations. Starts with
        a string encoding method if we have string features.'''

        pipeline={}

        # Choose a string encoding method, if needed
        if self.string_features is not None:
            options = list(self.string_encodings.keys())
            selection = choice(options)
            pipeline[selection] = self.string_encodings[selection]

        # Construct a random sequence of numerical feature engineering methods
        methods = list(self.numerical_methods.keys())
        shuffle(methods)
        methods = methods[:n_steps]

        for method in methods:

            pipeline[method] = {}
            parameters = self.numerical_methods[method]

            for parameter, values in parameters.items():

                value = choice(values)
                pipeline[method][parameter] = value

        return pipeline


    def check_argument_types(self,
            label: str,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame = None,
            string_features: list = None
    ) -> bool:

        '''Checks user argument types, returns true or false for all passing.'''

        check_pass = False

        if isinstance(label, str):
            check_pass = True

        else:
            raise TypeError('Label is not a string.')

        if isinstance(train_data, pd.DataFrame):
            check_pass = True

        else:
            raise TypeError('Train data is not a Pandas DataFrame.')

        if isinstance(test_data, pd.DataFrame) or test_data is None:
            check_pass = True

        else:
            raise TypeError('Test data is not a Pandas DataFrame.')

        if isinstance(string_features, list) or string_features is None:
            check_pass = True

        else:
            raise TypeError('String features is not a list.')

        return check_pass
