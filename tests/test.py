'''Unittest for dataset class.'''

import unittest
import pandas as pd
import engineered_datasets.dataset as ds

class TestDataSet(unittest.TestCase):
    '''Test class for main data set generator class'''

    def test_class_arguments(self):
        '''Tests assignments of class attributes from user arguments.'''

        dummy_df=pd.DataFrame({'col1': [0,1], 'col2': [3,4]})

        dataset=ds.DataSet(
            'test.h5',
            dummy_df,
            test_data=dummy_df,
            string_features=['feature1', 'feature2']
        )

        self.assertEqual(dataset.dataset_file, 'test.h5')
        self.assertTrue(isinstance(dataset.train_data, pd.DataFrame))
        self.assertTrue(isinstance(dataset.test_data, pd.DataFrame))
        self.assertTrue(isinstance(dataset.string_features, list))
        self.assertEqual(dataset.string_features[0], 'feature1')


if __name__ == '__main__':
    unittest.main()
