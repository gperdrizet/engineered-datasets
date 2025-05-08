'''Unittests for dataset class.'''

import unittest
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import ensembleset.feature_methods as fm
import tests.dummy_dataframe as test_data


class TestFeatureMethods(unittest.TestCase):
    '''Tests feature engineering method functions.'''

    def setUp(self):
        '''Dummy DataFrames for tests.'''

        self.dummy_df = test_data.DUMMY_DF


    def test_onehot_encoding(self):
        '''Tests string features onehot encoder.'''

        train_df, test_df=fm.onehot_encoding(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['strings'],
            {'sparse_output': False}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        for feature in list(train_df.columns):
            self.assertTrue(is_numeric_dtype(train_df[feature]))

        for feature in list(test_df.columns):
            self.assertTrue(is_numeric_dtype(test_df[feature]))


    def test_ordinal_encoding(self):
        '''Tests string feature ordinal encoder.'''

        train_df, test_df=fm.ordinal_encoding(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['strings'],
            {
                'handle_unknown': 'use_encoded_value',
                'unknown_value': np.nan  
            }
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))
        self.assertFalse(is_string_dtype(train_df['strings']))
        self.assertFalse(is_string_dtype(test_df['strings']))


    def test_poly_features(self):
        '''Tests polynomial feature transformer.'''

        train_df, test_df=fm.poly_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'degree': 2}

        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_spline_features(self):
        '''Tests spline features transformer.'''

        train_df, test_df=fm.spline_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'n_knots': 2}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_log_features(self):
        '''Tests log features transformer.'''

        train_df, test_df=fm.log_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'base': '2'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.log_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'base': 'e'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.log_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'base': '10'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_ratio_features(self):
        '''Tests ratio feature transformer.'''

        train_df, test_df=fm.ratio_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'div_zero_value': np.nan}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_exponential_features(self):
        '''Tests exponential features transformer.'''

        train_df, test_df=fm.exponential_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'base': 'e'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.exponential_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'base': '2'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_sum_features(self):
        '''Tests sum features transformer.'''

        train_df, test_df=fm.sum_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'n_addends': 2}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.sum_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'n_addends': 4}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_difference_features(self):
        '''Tests difference features transformer.'''

        train_df, test_df=fm.difference_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'n_subtrahends': 2}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.difference_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            list(self.dummy_df.columns),
            {'n_subtrahends': 4}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))
