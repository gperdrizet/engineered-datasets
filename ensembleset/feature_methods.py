'''Collection of functions to run feature engineering operations.'''

from math import e
from itertools import permutations, combinations
from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import KNNImputer
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, SplineTransformer, KBinsDiscretizer

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

    encoder=OneHotEncoder(**kwargs)

    encoded_data=encoder.fit_transform(train_df[features])
    encoded_df=pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
    train_df.drop(features, axis=1, inplace=True)
    train_df=pd.concat([train_df, encoded_df], axis=1)

    if test_df is not None:
        encoded_data=encoder.transform(test_df[features])
        encoded_df=pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        test_df.drop(features, axis=1, inplace=True)
        test_df=pd.concat([test_df, encoded_df], axis=1)

    return train_df, test_df


def ordinal_encoding(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's label encoder.'''

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

    '''Runs sklearn's polynomial feature transformer..'''

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

    transformer=PolynomialFeatures(**kwargs)

    transformed_data=transformer.fit_transform(train_working_df[features])
    new_columns=transformer.get_feature_names_out()
    transformed_train_df=pd.DataFrame(transformed_data, columns=new_columns)

    transformed_test_df = None

    if test_df is not None:

        transformed_data=transformer.transform(test_working_df[features])
        new_columns=transformer.get_feature_names_out()
        transformed_test_df=pd.DataFrame(transformed_data, columns=new_columns)

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

    '''Runs sklearn's polynomial feature transformer..'''

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

    transformer=SplineTransformer(**kwargs)

    transformed_data=transformer.fit_transform(train_working_df[features])
    new_columns=transformer.get_feature_names_out()
    transformed_train_df=pd.DataFrame(transformed_data, columns=new_columns)

    transformed_test_df = None

    if test_df is not None:

        transformed_data=transformer.transform(test_working_df[features])
        new_columns=transformer.get_feature_names_out()
        transformed_test_df=pd.DataFrame(transformed_data, columns=new_columns)

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

    features, train_working_df, test_working_df= scale_to_range(
        features=features,
        train_df=train_working_df,
        test_df=test_working_df,
        min_val=1,
        max_val=10
    )

    for feature in features:

        if kwargs['base'] == '2':
            train_df[f'{feature}_log2']=np.log2(train_working_df[feature])
            test_df[f'{feature}_log2']=np.log2(test_working_df[feature])

        if kwargs['base'] == 'e':
            train_df[f'{feature}_ln']=np.log(train_working_df[feature])
            test_df[f'{feature}_ln']=np.log(test_working_df[feature])

        if kwargs['base'] == '10':
            train_df[f'{feature}_log10']=np.log10(train_working_df[feature])
            test_df[f'{feature}_log10']=np.log10(test_working_df[feature])

    train_df.dropna(axis=1, how='all', inplace=True)
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

    features, train_working_df, test_working_df= scale_to_range(
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

    new_train_features={}
    new_test_features={}

    for feature in features:
        if min(train_working_df[feature]) <= 0 or min(test_working_df[feature]) <= 0:

            scaler=MinMaxScaler(feature_range=(1, 10))

            train_working_df[feature]=scaler.fit_transform(train_working_df[feature].to_frame())
            test_working_df[feature]=scaler.transform(test_working_df[feature].to_frame())

        if kwargs['base'] == 'e':
            new_train_features[f'{feature}_exp_base_e'] = e**train_working_df[feature].astype(float)
            new_test_features[f'{feature}_exp_base_e'] = e**test_working_df[feature].astype(float)

        elif kwargs['base'] == '2':
            new_train_features[f'{feature}_exp_base_2'] = 2**train_working_df[feature].astype(float)
            new_test_features[f'{feature}_exp_base_2'] = 2**test_working_df[feature].astype(float)

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

    if kwargs['n_addends'] > len(features):
        n_addends=len(features)

    else:
        n_addends=kwargs['n_addends']

    new_test_features={}
    new_train_features={}
    addend_sets=combinations(features, n_addends)

    for i, addend_set in enumerate(addend_sets):

        train_sum = [0]*len(train_working_df)
        test_sum = [0]*len(test_working_df)

        for addend in addend_set:

            train_sum += train_working_df[addend]
            test_sum += test_working_df[addend]

        new_train_features[f'sum_feature_{i}'] = train_sum
        new_test_features[f'sum_feature_{i}'] = test_sum

    train_df, test_df=add_new_features(
        new_train_features = new_train_features,
        new_test_features = new_test_features,
        train_df = train_df,
        test_df = test_df
    )

    return train_df, test_df


def difference_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds difference features for variable number of subtrahends.'''

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

    if kwargs['n_subtrahends'] > len(features):
        n_subtrahends=len(features)

    else:
        n_subtrahends=kwargs['n_subtrahends']

    new_test_features={}
    new_train_features={}
    subtrahend_sets=combinations(features, n_subtrahends)

    for subtrahend_set in subtrahend_sets:

        train_difference = train_working_df[subtrahend_set[0]]
        test_difference = test_working_df[subtrahend_set[0]]

        for subtrahend in subtrahend_set[1:]:

            train_difference -= train_working_df[subtrahend]
            test_difference -= test_working_df[subtrahend]

        new_train_features['-'.join(subtrahend_set)] = train_difference
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

    new_test_features={}
    new_train_features={}

    for feature in features:
        kde = KernelDensity(**kwargs).fit(train_working_df[feature].to_numpy().reshape(-1, 1))
        new_train_features[f'{feature}_kde']=kde.score_samples(
            train_working_df[feature].to_numpy().reshape(-1, 1)
        )
        new_test_features[f'{feature}_kde']=kde.score_samples(
            test_working_df[feature].to_numpy().reshape(-1, 1)
        )

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

    if len(train_df) <= kwargs['n_bins']:
        kwargs['n_bins'] = len(train_df) - 1

    kbins = KBinsDiscretizer(**kwargs)
    binned_features = kbins.fit_transform(train_working_df[features])
    binned_feature_names = kbins.get_feature_names_out()
    binned_feature_names = [f'{feature_name}_bins' for feature_name in binned_feature_names]
    binned_train_features_df = pd.DataFrame(binned_features, columns=binned_feature_names)

    binned_test_features_df = None

    if test_df is not None:
        binned_features = kbins.transform(test_working_df[features])
        binned_feature_names = kbins.get_feature_names_out()
        binned_feature_names = [f'{feature_name}_bins' for feature_name in binned_feature_names]
        binned_test_features_df = pd.DataFrame(binned_features, columns=binned_feature_names)

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

    train_working_df=train_df.copy()
    test_working_df=test_df.copy()

    for preprocessing_step in preprocessing_steps:

        preprocessing_func = globals().get(preprocessing_step)

        features, train_working_df, test_working_df = preprocessing_func(
            features,
            train_working_df,
            test_working_df
        )

    return features, train_working_df, test_working_df


def exclude_string_features(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Removes string features from features list.'''

    for feature in features:
        if test_df is not None:
            if is_numeric_dtype(train_df[feature]) is False or is_numeric_dtype(test_df[feature]) is False:
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

    # Get rid of large values
    train_df[features] = train_df[features].mask(
        abs(train_df[features]) > 1.79*10**102
    )

    if test_df is not None:
        test_df[features] = test_df[features].mask(
            test_df[features] > 1.79*10**102
        )

    return features, train_df, test_df


def remove_small_nums(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Replaces values smaller than the float64 limit with zero.'''

    # Get rid of small values
    train_df[features] = train_df[features].mask(
        abs(train_df[features]) < 2.23-308
    ).fillna(0.0)

    if test_df is not None:
        test_df[features] = test_df[features].mask(
            abs(test_df[features]) < 2.23-308
        ).fillna(0.0)

    return features, train_df, test_df


def knn_impute(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Uses SciKit-lean's KNN imputer to fill np.nan.'''

    imputer=KNNImputer()
    train_df[features] = imputer.fit_transform(train_df[features])

    if test_df is not None:
        test_df[features] = imputer.transform(test_df[features])

    return features, train_df, test_df


def scale_to_range(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        min_val:float = 0.0,
        max_val:float = 1.0
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Scales features into range'''

    scaler=MinMaxScaler(feature_range=(min_val, max_val))
    train_df[features]=scaler.fit_transform(train_df[features])

    if test_df is not None:
        test_df[features]=scaler.transform(test_df[features])

    return features, train_df, test_df


def remove_constants(
        features: list,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
) -> Tuple[list, pd.DataFrame, pd.DataFrame]:

    '''Removes constant valued features.'''

    train_df = train_df.loc[:,train_df.nunique(dropna=False) != 1]

    if test_df is None:
        test_df = test_df.loc[:,test_df.nunique(dropna=False) != 1]

    return train_df, test_df


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

    train_df=pd.concat([train_df, new_train_features], axis=1)
    test_df=pd.concat([test_df, new_test_features], axis=1)

    train_df.dropna(axis=1, how='all', inplace=True)
    test_df.dropna(axis=1, how='all', inplace=True)

    return train_df, test_df
