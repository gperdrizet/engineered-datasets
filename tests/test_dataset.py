'''Unittests for dataset class.'''

import logging
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import ensembleset.dataset as ds
import tests.dummy_dataframe as test_data

# pylint: disable=protected-access

Path('tests/logs').mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()

logging.basicConfig(
    filename='tests/logs/test_dataset.log',
    filemode='w',
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

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

        hdf = h5py.File('ensembleset_data/dataset.h5', 'r')

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

        hdf = h5py.File('ensembleset_data/dataset.h5', 'r')

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

        self.dummy_df = test_data.DUMMY_DF

        self.dataset = ds.DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings']
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

        self.dummy_df = test_data.DUMMY_DF

        self.dataset = ds.DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings']
        )


    def test_select_features(self):
        '''Tests feature selection function.'''

        features=self.dataset._select_features(3, self.dummy_df)
        self.assertEqual(len(features), 3)


class TestDatasetGeneration(unittest.TestCase):
    '''Tests dataset generation.'''

    def setUp(self):
        '''Dummy DataFrames and datasets for tests.'''

        self.n_datasets = 3
        self.frac_features = 0.1
        self.n_steps = 3

        self.dummy_df = test_data.DUMMY_DF

        self.dataset = ds.DataSet(
            label='floats_pos',
            train_data=self.dummy_df.copy(),
            test_data=self.dummy_df.copy(),
            string_features=['strings']
        )

        self.dataset.make_datasets(
            n_datasets=self.n_datasets,
            frac_features=self.frac_features,
            n_steps=self.n_steps
        )


    def test_make_datasets(self):
        '''Tests generation of datasets.'''

        hdf = h5py.File('ensembleset_data/dataset.h5', 'a')

        training_datasets=hdf['train']
        self.assertEqual(len(training_datasets), self.n_datasets + 1)

        testing_datasets=hdf['test']
        self.assertEqual(len(testing_datasets), self.n_datasets + 1)
