'''Unittests for dataset class.'''

import unittest
import h5py
import numpy as np
import pandas as pd
import ensembleset.dataset as ds
import tests.dummy_dataframe as test_data

# pylint: disable=protected-access

class TestDataSetInit(unittest.TestCase):
    '''Tests for main data set generator class initialization.'''

    def setUp(self):
        '''Dummy DataFrames and datasets for tests.'''

        self.dummy_df = test_data.DUMMY_DF

        self.dataset = ds.DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings']
        )


    def test_class_arguments(self):
        '''Tests assignments of class attributes from user arguments.'''

        self.assertTrue(isinstance(self.dataset.label, str))
        self.assertTrue(isinstance(self.dataset.train_data, pd.DataFrame))
        self.assertTrue(isinstance(self.dataset.test_data, pd.DataFrame))
        self.assertTrue(isinstance(self.dataset.string_features, list))
        self.assertEqual(self.dataset.string_features[0], 'strings')

        with self.assertRaises(TypeError):
            ds.DataSet(
                label=2, # Bad label
                train_data=self.dummy_df.copy(),
                test_data=self.dummy_df.copy(),
                string_features=['strings']
            )

        with self.assertRaises(TypeError):
            ds.DataSet(
                label='float_pos',
                train_data='Not a Pandas Dataframe', # Bad train data
                test_data=self.dummy_df.copy(),
                string_features=['strings']
            )

        with self.assertRaises(TypeError):
            ds.DataSet(
                label='float_pos',
                train_data=self.dummy_df.copy(),
                test_data='Not a Pandas Dataframe', # Bad test data
                string_features=['strings']
            )

        with self.assertRaises(TypeError):
            ds.DataSet(
                label='float_pos',
                train_data=self.dummy_df.copy(),
                test_data=self.dummy_df.copy(),
                string_features='Not a list of features' # Bad string features
            )


    def test_label_assignment(self):
        '''Tests assigning and saving labels.'''

        self.assertEqual(self.dataset.train_labels[-1], 7.0)
        self.assertEqual(self.dataset.test_labels[-1], 7.0)

        dataset=ds.DataSet(
            label='bad_label_feature',
            train_data=self.dummy_df,
            test_data=self.dummy_df,
            string_features=['strings']
        )

        self.assertTrue(np.isnan(dataset.train_labels[-1]))
        self.assertTrue(np.isnan(dataset.test_labels[-1]))


    def test_output_creation(self):
        '''Tests the creation of the HDF5 output sink.'''

        hdf = h5py.File('data/dataset.h5', 'r')

        self.assertTrue('train' in hdf)
        self.assertTrue('test' in hdf)
        self.assertEqual(hdf['test/labels'][-1], 7.0)
        self.assertEqual(hdf['test/labels'][-1], 7.0)

        hdf.close()

        _=ds.DataSet(
            label='bad_label_feature',
            train_data=self.dummy_df,
            test_data=self.dummy_df,
            string_features=['strings']
        )

        hdf = h5py.File('data/dataset.h5', 'r')

        self.assertTrue('train' in hdf)
        self.assertTrue('test' in hdf)
        self.assertTrue(np.isnan(hdf['test/labels'][-1]))
        self.assertTrue(np.isnan(hdf['test/labels'][-1]))

        hdf.close()


    def test_pipeline_options(self):
        '''Tests the creation of feature engineering pipeline options'''

        self.assertTrue(isinstance(self.dataset.string_encodings, dict))
        self.assertTrue(isinstance(self.dataset.numerical_methods, dict))


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
        self.assertEqual(len(features), 3)


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

        hdf = h5py.File('data/dataset.h5', 'a')

        training_datasets=hdf['train']
        self.assertEqual(len(training_datasets), 3)

        testing_datasets=hdf['test']
        self.assertEqual(len(testing_datasets), 3)
