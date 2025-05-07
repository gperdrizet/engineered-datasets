'''Unittests for dataset class.'''

import unittest
import h5py
import numpy as np
import pandas as pd
import ensembleset.dataset as ds
import ensembleset.feature_methods as fm

# pylint: disable=protected-access

class TestDataSetInit(unittest.TestCase):
    '''Tests for main data set generator class initialization.'''

    def setUp(self):
        '''Dummy DataFrames and datasets for tests.'''

        self.dummy_df_without_strings = pd.DataFrame({
            'feature1': [0,1],
            'feature2': [3,4],
            'feature3': [5,6]
        })

        self.dataset_without_string_feature = ds.DataSet(
            label='feature2',
            train_data=self.dummy_df_without_strings,
            test_data=self.dummy_df_without_strings
        )

        self.dummy_df_with_strings = pd.DataFrame({
            'feature1': [0,1],
            'feature2': [3,4],
            'feature3': ['a', 'b']
        })

        self.dataset_with_string_feature = ds.DataSet(
            label='feature2',
            train_data=self.dummy_df_with_strings,
            test_data=self.dummy_df_with_strings,
            string_features=['feature3']
        )


    def test_class_arguments(self):
        '''Tests assignments of class attributes from user arguments.'''

        self.assertTrue(isinstance(self.dataset_with_string_feature.label, str))
        self.assertTrue(isinstance(self.dataset_with_string_feature.train_data, pd.DataFrame))
        self.assertTrue(isinstance(self.dataset_with_string_feature.test_data, pd.DataFrame))
        self.assertTrue(isinstance(self.dataset_with_string_feature.string_features, list))
        self.assertEqual(self.dataset_with_string_feature.string_features[0], 'feature3')
        self.assertEqual(self.dataset_without_string_feature.train_labels[-1], 4)
        self.assertEqual(self.dataset_without_string_feature.test_labels[-1], 4)

        with self.assertRaises(TypeError):
            ds.DataSet(
                2,
                self.dummy_df_with_strings,
                test_data=self.dummy_df_with_strings,
                string_features=['feature3']
            )

        with self.assertRaises(TypeError):
            ds.DataSet(
                'feature2',
                'Not a Pandas Dataframe',
                test_data=self.dummy_df_with_strings,
                string_features=['feature3']
            )

        with self.assertRaises(TypeError):
            ds.DataSet(
                'feature2',
                self.dummy_df_with_strings,
                test_data='Not a Pandas Dataframe',
                string_features=['feature3']
            )

        with self.assertRaises(TypeError):
            ds.DataSet(
                'feature2',
                self.dummy_df_with_strings,
                test_data=self.dummy_df_with_strings,
                string_features='Not a list of features'
            )


    def test_label_assignment(self):
        '''Tests assigning and saving labels.'''

        dataset=ds.DataSet(
            'feature2',
            self.dummy_df_with_strings,
            test_data=self.dummy_df_with_strings,
            string_features=['feature3']
        )

        self.assertEqual(dataset.train_labels[-1], 4)
        self.assertEqual(dataset.test_labels[-1], 4)

        dataset=ds.DataSet(
            'feature7',
            self.dummy_df_with_strings,
            test_data=self.dummy_df_with_strings,
            string_features=['feature3']
        )

        self.assertTrue(np.isnan(dataset.train_labels[-1]))
        self.assertTrue(np.isnan(dataset.test_labels[-1]))


    def test_output_creation(self):
        '''Tests the creation of the HDF5 output sink.'''

        _=ds.DataSet(
            'feature2',
            self.dummy_df_with_strings,
            test_data=self.dummy_df_with_strings,
            string_features=['feature3']
        )

        hdf = h5py.File('data/dataset.hdf5', 'a')

        self.assertTrue('train' in hdf)
        self.assertTrue('test' in hdf)
        self.assertEqual(hdf['test/labels'][-1], 4)

        hdf.close()

        _=ds.DataSet(
            'feature7',
            self.dummy_df_with_strings,
            test_data=self.dummy_df_with_strings,
            string_features=['feature3']
        )

        hdf = h5py.File('data/dataset.hdf5', 'a')

        self.assertTrue('train' in hdf)
        self.assertTrue('test' in hdf)
        self.assertTrue(np.isnan(hdf['test/labels'][-1]))

        hdf.close()


    def test_pipeline_options(self):
        '''Tests the creation of feature engineering pipeline options'''

        self.assertTrue(isinstance(self.dataset_with_string_feature.string_encodings, dict))
        self.assertTrue(isinstance(self.dataset_with_string_feature.engineerings, dict))


class TestDataPipelineGen(unittest.TestCase):
    '''Tests for data pipeline generator function.'''

    def setUp(self):
        '''Dummy DataFrames and datasets for tests.'''

        self.dummy_df = pd.DataFrame({
            'feature1': [0,1],
            'feature2': [3,4],
            'feature3': ['a', 'b']
        })

        self.dataset = ds.DataSet(
            'feature2',
            self.dummy_df,
            test_data=self.dummy_df,
            string_features=['feature3']
        )


    def test_generate_data_pipeline(self):
        '''Tests the data pipeline generation function.'''

        pipeline=self.dataset._generate_data_pipeline(2)

        self.assertEqual(len(pipeline), 3)

        for operation, parameters in pipeline.items():
            self.assertTrue(isinstance(operation, str))
            self.assertTrue(isinstance(parameters, dict))


class TestFeatureSelection(unittest.TestCase):
    '''Tests for data pipeline generator function.'''

    def setUp(self):
        '''Dummy DataFrames and datasets for tests.'''

        self.dummy_df = pd.DataFrame({
            1: [0,1],
            'feature2': [3,4],
            'feature3': ['a', 'b']
        })

        self.dataset = ds.DataSet(
            'feature2',
            self.dummy_df,
            test_data=self.dummy_df,
            string_features=['feature3']
        )


    def test_select_features(self):
        '''Tests feature selection function.'''

        features=self.dataset._select_features(3, self.dummy_df)

        self.assertEqual(len(features), 2)

        for feature in features:
            self.assertTrue(isinstance(feature, str))

        self.assertFalse('feature2' in features)


class TestDatasetGeneration(unittest.TestCase):
    '''Tests dataset generation.'''

    def setUp(self):
        '''Dummy DataFrames and datasets for tests.'''

        self.dummy_df = pd.DataFrame({
            1: [0,1],
            'feature2': [3,4],
            'feature3': ['a', 'b']
        })

        self.dataset = ds.DataSet(
            'feature2',
            self.dummy_df,
            test_data=self.dummy_df,
            string_features=['feature3']
        )

        self.dataset.make_datasets(2, 2, 1)


    def test_make_datasets(self):
        '''Tests generation of datasets.'''

        hdf = h5py.File('data/dataset.hdf5', 'a')

        training_datasets=hdf['train']
        self.assertEqual(len(training_datasets), 2)

        testing_datasets=hdf['test']
        self.assertEqual(len(testing_datasets), 2)


class TestFeatureMethods(unittest.TestCase):
    '''Tests feature engineering method functions.'''

    def setUp(self):
        '''Dummy DataFrames for tests.'''

        self.dummy_df = pd.DataFrame({
            'feature1': [-1,0,1,2],
            'feature2': [3,4,5,np.nan],
            'feature3': ['a', 'b', 'c', 'd']
        })


    def test_onehot_encoding(self):
        '''Tests string features onehot encoder.'''

        train_df, test_df=fm.onehot_encoding(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature3'],
            {'sparse_output': False}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_ordinal_encoding(self):
        '''Tests string feature ordinal encoder.'''

        train_df, test_df=fm.ordinal_encoding(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature3'],
            {
                'handle_unknown': 'use_encoded_value',
                'unknown_value': np.nan  
            }
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_poly_features(self):
        '''Tests polynomial feature transformer.'''

        train_df, test_df=fm.poly_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature2'],
            {'degree': 2}

        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_spline_features(self):
        '''Tests spline features transformer.'''

        train_df, test_df=fm.spline_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1', 'feature2'],
            {'n_knots': 2}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_log_features(self):
        '''Tests log features transformer.'''

        train_df, test_df=fm.log_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1'],
            {'base': '2'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.log_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1'],
            {'base': 'e'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.log_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1'],
            {'base': '10'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_ratio_features(self):
        '''Tests ratio feature transformer.'''

        train_df, test_df=fm.ratio_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1', 'feature2'],
            {'div_zero_value': np.nan}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_exponential_features(self):
        '''Tests exponential features transformer.'''

        train_df, test_df=fm.exponential_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1', 'feature2'],
            {'base': 'e'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.exponential_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1', 'feature2'],
            {'base': '2'}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_sum_features(self):
        '''Tests sum features transformer.'''

        train_df, test_df=fm.sum_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1', 'feature2'],
            {'n_addends': 2}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.sum_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1', 'feature2'],
            {'n_addends': 4}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))


    def test_difference_features(self):
        '''Tests difference features transformer.'''

        train_df, test_df=fm.difference_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1', 'feature2'],
            {'n_subtrahends': 2}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))

        train_df, test_df=fm.difference_features(
            self.dummy_df.copy(),
            self.dummy_df.copy(),
            ['feature1', 'feature2'],
            {'n_subtrahends': 4}
        )

        self.assertTrue(isinstance(train_df, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))