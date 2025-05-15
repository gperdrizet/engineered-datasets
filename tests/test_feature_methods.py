'''Unittests for feature methods.'''

import logging
from pathlib import Path

import unittest
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import ensembleset.feature_methods as fm
import tests.dummy_dataframe as test_data

Path('tests/logs').mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()

logging.basicConfig(
    filename='tests/logs/test_feature_methods.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

class TestFeatureMethods(unittest.TestCase):
    '''Tests feature engineering method functions.'''

    def setUp(self):
        '''Dummy DataFrames for tests.'''

        self.dummy_df = test_data.DUMMY_DF


    def test_onehot_encoding(self):
        '''Tests string features onehot encoder.'''

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.onehot_encoding(
                train_df=self.dummy_df.copy(),
                test_df=testing_data,
                features=['strings'],
                kwargs={'sparse_output': False}
            )

            self.assertTrue(isinstance(train_df, pd.DataFrame))

            for feature in list(train_df.columns):
                self.assertTrue(is_numeric_dtype(train_df[feature]))

            if testing_data is not None:

                self.assertTrue(isinstance(test_df, pd.DataFrame))

                for feature in list(test_df.columns):
                    self.assertTrue(is_numeric_dtype(test_df[feature]))

            elif testing_data is None:
                self.assertEqual(test_df, None)


    def test_ordinal_encoding(self):
        '''Tests string feature ordinal encoder.'''

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.ordinal_encoding(
                train_df=self.dummy_df.copy(),
                test_df=testing_data,
                features=['strings'],
                kwargs={
                    'handle_unknown': 'use_encoded_value',
                    'unknown_value': np.nan  
                }
            )

            self.assertTrue(isinstance(train_df, pd.DataFrame))
            self.assertFalse(is_string_dtype(train_df['strings']))

            if testing_data is not None:
                self.assertTrue(isinstance(test_df, pd.DataFrame))
                self.assertFalse(is_string_dtype(test_df['strings']))


    def test_poly_features(self):
        '''Tests polynomial feature transformer.'''

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.poly_features(
                train_df=self.dummy_df.copy(),
                test_df=testing_data,
                features=list(self.dummy_df.columns),
                kwargs={'degree': 2}
            )

            self.assertTrue(isinstance(train_df, pd.DataFrame))

            if testing_data is not None:
                self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_spline_features(self):
        '''Tests spline features transformer.'''

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.spline_features(
                train_df=self.dummy_df.copy(),
                test_df=testing_data,
                features=list(self.dummy_df.columns),
                kwargs={'n_knots': 2}
            )

            self.assertTrue(isinstance(train_df, pd.DataFrame))

            if testing_data is not None:
                self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_log_features(self):
        '''Tests log features transformer.'''

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.log_features(
                train_df=self.dummy_df.copy(),
                test_df=testing_data,
                features=list(self.dummy_df.columns),
                kwargs={'base': '2'}
            )

            self.assertTrue(isinstance(train_df, pd.DataFrame))

            if testing_data is not None:
                self.assertTrue(isinstance(test_df, pd.DataFrame))

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.log_features(
                train_df=self.dummy_df.copy(),
                test_df=testing_data,
                features=list(self.dummy_df.columns),
                kwargs={'base': 'e'}
            )

            self.assertTrue(isinstance(train_df, pd.DataFrame))

            if testing_data is not None:
                self.assertTrue(isinstance(test_df, pd.DataFrame))

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.log_features(
                train_df=self.dummy_df.copy(),
                test_df=testing_data,
                features=list(self.dummy_df.columns),
                kwargs={'base': '10'}
            )

            self.assertTrue(isinstance(train_df, pd.DataFrame))

            if testing_data is not None:
                self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_ratio_features(self):
        '''Tests ratio feature transformer.'''

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.ratio_features(
                self.dummy_df.copy(),
                testing_data,
                list(self.dummy_df.columns),
                {'div_zero_value': np.nan}
            )

        self.assertTrue(isinstance(train_df, pd.DataFrame))

        if testing_data is not None:
            self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_exponential_features(self):
        '''Tests exponential features transformer.'''

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.exponential_features(
                self.dummy_df.copy(),
                testing_data,
                list(self.dummy_df.columns),
                {'base': 'e'}
            )

        self.assertTrue(isinstance(train_df, pd.DataFrame))

        if testing_data is not None:
            self.assertTrue(isinstance(test_df, pd.DataFrame))


        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.exponential_features(
                self.dummy_df.copy(),
                testing_data,
                list(self.dummy_df.columns),
                {'base': '2'}
            )

        self.assertTrue(isinstance(train_df, pd.DataFrame))

        if testing_data is not None:
            self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_sum_features(self):
        '''Tests sum features transformer.'''

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.sum_features(
                self.dummy_df.copy(),
                testing_data,
                list(self.dummy_df.columns),
                {'n_addends': 2}
            )

            self.assertTrue(isinstance(train_df, pd.DataFrame))

            if testing_data is not None:
                self.assertTrue(isinstance(test_df, pd.DataFrame))

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.sum_features(
                self.dummy_df.copy(),
                testing_data,
                list(self.dummy_df.columns),
                {'n_addends': 3}
            )

        self.assertTrue(isinstance(train_df, pd.DataFrame))

        if testing_data is not None:
            self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_difference_features(self):
        '''Tests difference features transformer.'''

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.difference_features(
                self.dummy_df.copy(),
                testing_data,
                list(self.dummy_df.columns),
                {'n_subtrahends': 2}
            )

            self.assertTrue(isinstance(train_df, pd.DataFrame))

            if testing_data is not None:
                self.assertTrue(isinstance(test_df, pd.DataFrame))

        for testing_data in [None, self.dummy_df.copy()]:

            train_df, test_df=fm.difference_features(
                self.dummy_df.copy(),
                testing_data,
                list(self.dummy_df.columns),
                {'n_subtrahends': 100}
            )

        self.assertTrue(isinstance(train_df, pd.DataFrame))

        if testing_data is not None:
            self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_kde_smoothing(self):
        '''Tests kde smoother.'''

        train_df, test_df=fm.kde_smoothing( # pylint: disable=E1101
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'bandwidth': 'silverman', 'sample_size': 1000}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_kbins_quantization(self):
        '''Tests kbins discretizer.'''

        train_df, test_df=fm.kbins_quantization( # pylint: disable=E1101
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'n_bins': 64, 'encode': 'ordinal', 'strategy': 'quantile'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))
