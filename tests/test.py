'''Unittests for dataset class.'''

import unittest
import h5py
import pandas as pd
import engineered_datasets.dataset as ds

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
            self.dummy_df_without_strings,
            test_data=self.dummy_df_without_strings
        )

        self.dummy_df_with_strings = pd.DataFrame({
            'feature1': [0,1],
            'feature2': [3,4],
            'feature3': ['a', 'b']
        })

        self.dataset_with_string_feature = ds.DataSet(
            self.dummy_df_with_strings,
            test_data=self.dummy_df_with_strings,
            string_features=['feature3']
        )


    def test_class_arguments(self):
        '''Tests assignments of class attributes from user arguments.'''

        self.assertTrue(isinstance(self.dataset_with_string_feature.train_data, pd.DataFrame))
        self.assertTrue(isinstance(self.dataset_with_string_feature.test_data, pd.DataFrame))
        self.assertTrue(isinstance(self.dataset_with_string_feature.string_features, list))
        self.assertEqual(self.dataset_with_string_feature.string_features[0], 'feature3')

        with self.assertRaises(TypeError):
            ds.DataSet(
                'Not a Pandas Dataframe',
                test_data=self.dummy_df_with_strings,
                string_features=['feature3']
            )

        with self.assertRaises(TypeError):
            ds.DataSet(
                self.dummy_df_with_strings,
                test_data='Not a Pandas Dataframe',
                string_features=['feature3']
            )

        with self.assertRaises(TypeError):
            ds.DataSet(
                self.dummy_df_with_strings,
                test_data=self.dummy_df_with_strings,
                string_features='Not a list of features'
            )


    def test_output_creation(self):
        '''Tests the creation of the HDF5 output sink.'''

        hdf = h5py.File('data/dataset.hdf5', 'a')

        self.assertTrue('train' in hdf)
        self.assertTrue('test' in hdf)


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
            self.dummy_df,
            test_data=self.dummy_df,
            string_features=['feature3']
        )


    def test_generate_data_pipeline(self):
        '''Tests the data pipeline generation function.'''

        pipeline=self.dataset._generate_data_pipeline()

        self.assertGreater(len(pipeline), 1)

        for operation, parameters in pipeline.items():
            self.assertTrue(isinstance(operation, str))
            self.assertTrue(isinstance(parameters, dict))

