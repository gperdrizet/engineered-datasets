'''Collection of functions to run feature engineering operations.'''

import logging
import multiprocessing as mp
from math import e
from itertools import permutations, combinations
from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import gaussian_kde
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    SplineTransformer,
    KBinsDiscretizer
)

pd.set_option('display.width', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


def onehot_encoding(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's one hot encoder.'''

    logger = logging.getLogger(__name__ + '.onehot_encoding')
    logger.addHandler(logging.NullHandler())
    logger.info('One-hot encoding string features')

    if features is not None:

        encoder=OneHotEncoder(**kwargs)

        encoded_data=encoder.fit_transform(train_df[features])
        encoded_df=pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        train_df=pd.concat(
            [train_df.reset_index(drop=True), encoded_df.reset_index(drop=True)],
            axis=1
        )
        train_df.drop(features, axis=1, inplace=True)

        if test_df is not None:
            encoded_data=encoder.transform(test_df[features])
            encoded_df=pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
            test_df=pd.concat(
                [test_df.reset_index(drop=True), encoded_df.reset_index(drop=True)],
                axis=1
            )
            test_df.drop(features, axis=1, inplace=True)

    return train_df, test_df


def ordinal_encoding(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's ordinal encoder.'''

    logger = logging.getLogger(__name__ + '.ordinal_encoding')
    logger.addHandler(logging.NullHandler())
    logger.info('Ordinal encoding string features')

    if features is not None:

        encoder=OrdinalEncoder(**kwargs)

        train_df[features]=encoder.fit_transform(train_df[features])

        if test_df is not None:
            test_df[features]=encoder.transform(test_df[features])

    return train_df, test_df


def poly_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's polynomial feature transformer.'''

    logger = logging.getLogger(__name__ + '.poly_features')
    logger.addHandler(logging.NullHandler())
    logger.info('Adding polynomial features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None:

        transformer=PolynomialFeatures(**kwargs)

        try:
            transformed_data=transformer.fit_transform(train_working_df[features])
            new_columns=transformer.get_feature_names_out()
            transformed_train_df=pd.DataFrame(transformed_data, columns=new_columns)

        except ValueError:
            print('Value error in poly feature transformer.')

        transformed_test_df = None

        if test_df is not None:

            try:
                transformed_data=transformer.transform(test_working_df[features])
                new_columns=transformer.get_feature_names_out()
                transformed_test_df=pd.DataFrame(transformed_data, columns=new_columns)

            except ValueError:
                print('Value error in poly feature transformer.')

        train_df, test_df = add_new_features(
            new_train_features = transformed_train_df,
            new_test_features = transformed_test_df,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df


def spline_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's polynomial feature transformer.'''

    logger = logging.getLogger(__name__ + '.spline_features')
    logger.addHandler(logging.NullHandler())
    logger.info('Adding spline features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf',
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None:

        transformer=SplineTransformer(**kwargs)

        try:
            transformed_data=transformer.fit_transform(train_working_df[features])
            new_columns=transformer.get_feature_names_out()
            transformed_train_df=pd.DataFrame(transformed_data, columns=new_columns)

        except ValueError:
            print('Caught value error error during spline feature concatenation')

        transformed_test_df = None

        if test_df is not None:

            try:
                transformed_data=transformer.transform(test_working_df[features])
                new_columns=transformer.get_feature_names_out()
                transformed_test_df=pd.DataFrame(transformed_data, columns=new_columns)

            except ValueError:
                print('Caught value error error during spline feature concatenation')

        train_df, test_df = add_new_features(
            new_train_features = transformed_train_df,
            new_test_features = transformed_test_df,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df


def log_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Takes log of feature, uses sklearn min-max scaler if needed
    to avoid undefined log errors.'''

    logger = logging.getLogger(__name__ + '.log_features')
    logger.addHandler(logging.NullHandler())
    logger.info('Adding log features')

    features, train_working_df, test_working_df = preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants'
        ]
    )

    features, train_working_df, test_working_df = scale_to_range(
        features=features,
        train_df=train_working_df,
        test_df=test_working_df,
        min_val=1,
        max_val=10
    )

    if features is not None:

        for feature in features:

            if kwargs['base'] == '2':
                train_df[f'{feature}_log2']=np.log2(train_working_df[feature])

                if test_df is not None:
                    test_df[f'{feature}_log2']=np.log2(test_working_df[feature])

            if kwargs['base'] == 'e':
                train_df[f'{feature}_ln']=np.log(train_working_df[feature])

                if test_df is not None:
                    test_df[f'{feature}_ln']=np.log(test_working_df[feature])

            if kwargs['base'] == '10':
                train_df[f'{feature}_log10']=np.log10(train_working_df[feature])

                if test_df is not None:
                    test_df[f'{feature}_log10']=np.log10(test_working_df[feature])

        train_df.dropna(axis=1, how='all', inplace=True)

        if test_df is not None:
            test_df.dropna(axis=1, how='all', inplace=True)

    return train_df, test_df


def ratio_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds every possible ratio feature, replaces divide by zero errors
    with np.nan.'''

    logger = logging.getLogger(__name__ + '.ratio_features')
    logger.addHandler(logging.NullHandler())
    logger.info('Adding ratio features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants'
        ]
    )

    if features is not None:

        features, train_working_df, test_working_df = scale_to_range(
            features=features,
            train_df=train_working_df,
            test_df=test_working_df,
            min_val=1,
            max_val=10
        )

        feature_pairs=permutations(features, 2)

        new_train_features={}
        new_test_features={}

        for feature_a, feature_b in feature_pairs:

            quotient = np.divide(
                np.array(train_working_df[feature_a]),
                np.array(train_working_df[feature_b]),
                out=np.array([kwargs['div_zero_value']]*len(train_working_df[feature_a])),
                where=np.array(train_working_df[feature_b]) != 0
            )

            new_train_features[f'{feature_a}_over_{feature_b}'] = quotient

            if test_df is not None:

                quotient = np.divide(
                    np.array(test_working_df[feature_a]),
                    np.array(test_working_df[feature_b]),
                    out=np.array([kwargs['div_zero_value']]*len(test_working_df[feature_a])),
                    where=np.array(test_working_df[feature_b]) != 0
                )

                new_test_features[f'{feature_a}_over_{feature_b}'] = quotient

        train_df, test_df=add_new_features(
            new_train_features = new_train_features,
            new_test_features = new_test_features,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df


def exponential_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds exponential features with base 2 or base e.'''

    logger = logging.getLogger(__name__ + '.exponential_features')
    logger.addHandler(logging.NullHandler())
    logger.info('Adding exponential features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None:

        new_train_features={}
        new_test_features={}

        for feature in features:

            if kwargs['base'] == 'e':
                new_train_features[f'{feature}_exp_base_e'] = (
                    e**train_working_df[feature].astype(float)
                )

                if test_df is not None:
                    new_test_features[f'{feature}_exp_base_e'] = (
                        e**test_working_df[feature].astype(float)
                    )

            elif kwargs['base'] == '2':
                new_train_features[f'{feature}_exp_base_2'] = (
                    2**train_working_df[feature].astype(float))

                if test_df is not None:
                    new_test_features[f'{feature}_exp_base_2'] = (
                        2**test_working_df[feature].astype(float)
                    )

        train_df, test_df=add_new_features(
            new_train_features = new_train_features,
            new_test_features = new_test_features,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df


def sum_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds sum features for variable number of addends.'''

    logger = logging.getLogger(__name__ + '.sum_features')
    logger.addHandler(logging.NullHandler())
    logger.info('Adding sum features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None:

        if kwargs['n_addends'] > len(features):
            n_addends=len(features)

        else:
            n_addends=kwargs['n_addends']

        new_test_features={}
        new_train_features={}
        addend_sets=list(combinations(features, n_addends))

        logger.info(
            'Will compute sums for %s sets of %s features', len(addend_sets), n_addends
        )

        for i, addend_set in enumerate(addend_sets):

            train_sum = [0]*len(train_working_df)

            for addend in addend_set:

                train_sum += train_working_df[addend]

            new_train_features[f'sum_feature_{i}'] = train_sum

            if test_df is not None:

                test_sum = [0]*len(test_working_df)

                for addend in addend_set:

                    test_sum += test_working_df[addend]

                new_test_features[f'sum_feature_{i}'] = test_sum

        train_df, test_df=add_new_features(
            new_train_features = new_train_features,
            new_test_features = new_test_features,
            train_df = train_df,
            test_df = test_df
        )

    else:
        logger.info('No features to sum')

    return train_df, test_df


def difference_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds difference features for variable number of subtrahends.'''

    logger = logging.getLogger(__name__ + '.difference_features')
    logger.addHandler(logging.NullHandler())
    logger.info('Adding difference features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None:

        if kwargs['n_subtrahends'] > len(features):
            n_subtrahends=len(features)

        else:
            n_subtrahends=kwargs['n_subtrahends']

        new_test_features={}
        new_train_features={}
        subtrahend_sets=combinations(features, n_subtrahends)

        for subtrahend_set in subtrahend_sets:

            train_difference = train_working_df[subtrahend_set[0]]

            for subtrahend in subtrahend_set[1:]:

                train_difference -= train_working_df[subtrahend]

            new_train_features['-'.join(subtrahend_set)] = train_difference

            if test_df is not None:

                test_difference = test_working_df[subtrahend_set[0]]

                for subtrahend in subtrahend_set[1:]:

                    test_difference -= test_working_df[subtrahend]

                new_test_features['-'.join(subtrahend_set)] = test_difference

        train_df, test_df=add_new_features(
            new_train_features = new_train_features,
            new_test_features = new_test_features,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df


def kde_smoothing(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Uses kernel density estimation to smooth features.'''

    logger = logging.getLogger(__name__ + '.kde_smoothing')
    logger.addHandler(logging.NullHandler())
    logger.info('Adding kernel density estimate smoothed features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None:

        new_test_features={}
        new_train_features={}

        if len(train_working_df) > kwargs['sample_size']:
            sample_df=train_working_df.sample(n=kwargs['sample_size'])

        else:
            sample_df=train_working_df

        workers = mp.cpu_count() - 2

        for feature in features:

            try:

                scipy_kde = gaussian_kde(
                    sample_df[feature].to_numpy().flatten(),
                    bw_method = kwargs['bandwidth']
                )

                with mp.Pool(workers) as p:
                    new_train_features[f'{feature}_kde'] = np.concatenate(p.map(
                        scipy_kde,
                        np.array_split(train_working_df[feature].to_numpy().flatten(), workers)
                    ))

                if test_df is not None:
                    with mp.Pool(workers) as p:
                        new_test_features[f'{feature}_kde'] = np.concatenate(p.map(
                            scipy_kde,
                            np.array_split(test_working_df[feature].to_numpy().flatten(), workers)
                        ))

            except np.linalg.LinAlgError:
                print('Numpy linear algebra error in gaussian KDE.')

        train_df, test_df=add_new_features(
            new_train_features = new_train_features,
            new_test_features = new_test_features,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df


def kbins_quantization(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Discretizes feature with Kbins quantization.'''

    logger = logging.getLogger(__name__ + '.kbins_quantization')
    logger.addHandler(logging.NullHandler())
    logger.info('Adding k-bins quantized features')

    features, train_working_df, test_working_df=preprocess_features(
        features=features,
        train_df=train_df,
        test_df=test_df,
        preprocessing_steps=[
            'exclude_string_features',
            'enforce_floats',
            'remove_inf', 
            'remove_large_nums',
            'remove_small_nums',
            'knn_impute',
            'remove_constants',
            'scale_to_range'
        ]
    )

    if features is not None:

        if len(train_df) <= kwargs['n_bins']:
            kwargs['n_bins'] = len(train_df) - 1

        kbins = KBinsDiscretizer(**kwargs)

        try:
            binned_features = kbins.fit_transform(train_working_df[features])
            binned_feature_names = kbins.get_feature_names_out()
            binned_feature_names = [f'{feature_name}_bins' for feature_name in binned_feature_names]
            binned_train_features_df = pd.DataFrame(binned_features, columns=binned_feature_names)

        except ConvergenceWarning:
            print('Caught ConvergenceWarning in KbinsDescretizer.')

        except UserWarning:
            print('Caught UserWarning in KbinsDiscretizer.')

        except ValueError:
            print('Caught ValueError in KbinsDiscretizer.')

        binned_test_features_df = None

        if test_df is not None:

            try:
                binned_features = kbins.transform(test_working_df[features])
                binned_feature_names = kbins.get_feature_names_out()
                binned_feature_names = (
                    [f'{feature_name}_bins' for feature_name in binned_feature_names]
                )
                binned_test_features_df = pd.DataFrame(
                    binned_features,
                    columns=binned_feature_names
                )

            except ConvergenceWarning:
                print('Caught ConvergenceWarning in KbinsDescretizer.')

            except UserWarning:
                print('Caught UserWarning in KbinsDiscretizer.')

            except ValueError:
                print('Caught ValueError in KbinsDiscretizer.')

        train_df, test_df = add_new_features(
            new_train_features = binned_train_features_df,
            new_test_features = binned_test_features_df,
            train_df = train_df,
            test_df = test_df
        )

    return train_df, test_df


def preprocess_features(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        preprocessing_steps:list
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Runs feature preprocessing steps.'''

    logger = logging.getLogger(__name__ + '.preprocess_features')
    logger.addHandler(logging.NullHandler())

    train_working_df=train_df.copy()

    if test_df is not None:
        test_working_df = test_df.copy()

    else:
        test_working_df = None

    for preprocessing_step in preprocessing_steps:

        preprocessing_func = globals().get(preprocessing_step)

        if features is not None:

            logger.info('Preprocessor running %s', preprocessing_step)

            features, train_working_df, test_working_df = preprocessing_func(
                features,
                train_working_df,
                test_working_df
            )

        else:

            logger.info('Preprocessor step %s received no features', preprocessing_step)


    return features, train_working_df, test_working_df


def exclude_string_features(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Removes string features from features list.'''

    if features is not None:

        for feature in features:
            if test_df is not None:
                if (is_numeric_dtype(train_df[feature]) is False or
                    is_numeric_dtype(test_df[feature]) is False):

                    train_df.drop(feature, axis=1, inplace=True, errors='ignore')
                    test_df.drop(feature, axis=1, inplace=True, errors='ignore')
                    features.remove(feature)

            elif test_df is None:
                if is_numeric_dtype(train_df[feature]) is False:
                    train_df.drop(feature, axis=1, inplace=True, errors='ignore')
                    features.remove(feature)

    return features, train_df, test_df


def enforce_floats(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Changes features to float dtype.'''

    if features is not None:

        train_df[features]=train_df[features].astype(float).copy()

        if test_df is not None:
            test_df[features]=test_df[features].astype(float).copy()

    return features, train_df, test_df


def remove_inf(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Replaces any np.inf values with np.NAN.'''

    if features is not None:

        # Get rid of np.inf
        train_df[features]=train_df[features].replace(
            [np.inf, -np.inf],
            np.nan
        )

        if test_df is not None:
            test_df[features]=test_df[features].replace(
                [np.inf, -np.inf],
                np.nan
            )

    return features, train_df, test_df


def remove_large_nums(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Replaces numbers larger than the cube root of the float64 limit with np.nan.'''

    if features is not None:

        # Get rid of large values
        train_df[features] = train_df[features].mask(
            abs(train_df[features]) > 1.0*10**102
        )

        if test_df is not None:
            test_df[features] = test_df[features].mask(
                test_df[features] > 1.0*10**102
            )

    return features, train_df, test_df


def remove_small_nums(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Replaces values smaller than the float64 limit with zero.'''

    if features is not None:

        # Get rid of small values
        train_df[features] = train_df[features].mask(
            abs(train_df[features]) < 1.0-102
        ).fillna(0.0)

        if test_df is not None:
            test_df[features] = test_df[features].mask(
                abs(test_df[features]) < 1.0-102
            ).fillna(0.0)

    return features, train_df, test_df


def knn_impute(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Uses SciKit-lean's KNN imputer to fill np.nan.'''

    if features is not None:

        imputer=KNNImputer()
        train_df[features] = imputer.fit_transform(train_df[features])

        if test_df is not None:
            test_df[features] = imputer.transform(test_df[features])

    return features, train_df, test_df


def remove_constants(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Removes constant valued features.'''

    constant_features=train_df.loc[:,train_df.nunique(dropna=False) == 1]
    train_df.drop(constant_features, axis=1, inplace=True)

    if test_df is not None:
        test_df.drop(constant_features, axis=1, inplace=True)

    new_features = list(set(features) & set(train_df.columns.to_list()))

    if len(new_features) == 0:
        new_features = None
        train_df = None
        test_df = None

    return new_features, train_df, test_df


def scale_to_range(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        min_val:float = 0.0,
        max_val:float = 1.0
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Scales features into range'''

    if features is not None:

        scaler=MinMaxScaler(feature_range=(min_val, max_val))
        train_df[features]=scaler.fit_transform(train_df[features])

        if test_df is not None:
            test_df[features]=scaler.transform(test_df[features])

    return features, train_df, test_df


def add_new_features(
    new_train_features,
    new_test_features,
    train_df:pd.DataFrame,
    test_df:pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds new features to dataframes'''

    if isinstance(new_test_features, dict):
        new_train_features=pd.DataFrame.from_dict(new_train_features)

        if new_test_features is not None:
            new_test_features=pd.DataFrame.from_dict(new_test_features)

    train_df=pd.concat(
        [train_df.reset_index(drop=True), new_train_features.reset_index(drop=True)],
        axis=1
    )

    train_df = train_df.loc[:,~train_df.columns.duplicated()].copy()
    train_df.sort_index(axis=1, inplace=True)
    train_df.reset_index(inplace=True, drop=True)

    if test_df is not None:
        test_df=pd.concat(
            [test_df.reset_index(drop=True), new_test_features.reset_index(drop=True)],
            axis=1
        )

        test_df = test_df.loc[:,~test_df.columns.duplicated()].copy()
        test_df.sort_index(axis=1, inplace=True)
        test_df.reset_index(inplace=True, drop=True)

    return train_df, test_df
