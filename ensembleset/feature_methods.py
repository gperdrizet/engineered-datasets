'''Collection of functions to run feature engineering operations.'''

from math import e
from itertools import permutations, combinations
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, SplineTransformer


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

    transformer=PolynomialFeatures(**kwargs)
    imputer=SimpleImputer(strategy='mean')

    imputed_data=imputer.fit_transform(train_df[features])
    encoded_data=transformer.fit_transform(imputed_data)
    encoded_df=pd.DataFrame(encoded_data, columns=transformer.get_feature_names_out())
    train_df.drop(features, axis=1, inplace=True)
    train_df=pd.concat([train_df, encoded_df], axis=1)

    if test_df is not None:

        imputed_data=imputer.transform(test_df[features])
        encoded_data=transformer.transform(imputed_data)
        encoded_df=pd.DataFrame(encoded_data, columns=transformer.get_feature_names_out())
        test_df.drop(features, axis=1, inplace=True)
        test_df=pd.concat([test_df, encoded_df], axis=1)

    return train_df, test_df


def spline_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's polynomial feature transformer..'''

    transformer=SplineTransformer(**kwargs)
    imputer=SimpleImputer(strategy='mean')

    imputed_data=imputer.fit_transform(train_df[features])
    encoded_data=transformer.fit_transform(imputed_data)
    encoded_df=pd.DataFrame(encoded_data, columns=transformer.get_feature_names_out())
    train_df.drop(features, axis=1, inplace=True)
    train_df=pd.concat([train_df, encoded_df], axis=1)

    if test_df is not None:

        imputed_data=imputer.transform(test_df[features])
        encoded_data=transformer.transform(imputed_data)
        encoded_df=pd.DataFrame(encoded_data, columns=transformer.get_feature_names_out())
        test_df.drop(features, axis=1, inplace=True)
        test_df=pd.concat([test_df, encoded_df], axis=1)

    return train_df, test_df


def log_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Takes log of feature, uses sklearn min-max scaler if needed
    to avoid undefined log errors.'''

    for feature in features:
        if min(train_df[feature]) <= 0:

            scaler=MinMaxScaler(feature_range=(1, 100))

            train_df[feature]=scaler.fit_transform(train_df[feature].to_frame())
            test_df[feature]=scaler.transform(test_df[feature].to_frame())

        if kwargs['base'] == '2':
            train_df[f'{feature}_log2']=np.log2(train_df[feature])
            test_df[f'{feature}_log2']=np.log2(test_df[feature])

        if kwargs['base'] == 'e':
            train_df[f'{feature}_ln']=np.log(train_df[feature])
            test_df[f'{feature}_ln']=np.log(test_df[feature])

        if kwargs['base'] == '10':
            train_df[f'{feature}_log10']=np.log10(train_df[feature])
            test_df[f'{feature}_log10']=np.log10(test_df[feature])

    return train_df, test_df


def ratio_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds every possible ratio feature, replaces divide by zero errors
    with np.nan.'''

    feature_pairs=permutations(features, 2)

    train_features={}
    test_features={}

    for feature_a, feature_b in feature_pairs:

        quotient = np.divide(
            np.array(train_df[feature_a]),
            np.array(train_df[feature_b]),
            out=np.array([kwargs['div_zero_value']]*len(train_df[feature_a])),
            where=np.array(train_df[feature_b]) != 0
        )

        train_features[f'{feature_a}_over_{feature_b}'] = quotient

        quotient = np.divide(
            np.array(test_df[feature_a]),
            np.array(test_df[feature_b]),
            out=np.array([kwargs['div_zero_value']]*len(test_df[feature_a])),
            where=np.array(test_df[feature_b]) != 0
        )

        test_features[f'{feature_a}_over_{feature_b}'] = quotient

    new_train_df=pd.DataFrame.from_dict(train_features)
    new_test_df=pd.DataFrame.from_dict(test_features)

    train_df=pd.concat([train_df, new_train_df], axis=1)
    test_df=pd.concat([test_df, new_test_df], axis=1)

    return train_df, test_df


def exponential_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds exponential features with base 2 or base e.'''

    new_train_features={}
    new_test_features={}

    for feature in features:
        if kwargs['base'] == 'e':
            new_train_features[f'{feature}_exp_base_e'] = e**train_df[feature].astype(float)
            new_train_features[f'{feature}_exp_base_e'] = e**test_df[feature].astype(float)

        elif kwargs['base'] == '2':
            new_train_features[f'{feature}_exp_base_e'] = 2**train_df[feature].astype(float)
            new_test_features[f'{feature}_exp_base_e'] = 2**test_df[feature].astype(float)

    new_train_df=pd.DataFrame.from_dict(new_train_features)
    new_test_df=pd.DataFrame.from_dict(new_test_features)

    train_df=pd.concat([train_df, new_train_df], axis=1)
    test_df=pd.concat([test_df, new_test_df], axis=1)

    return train_df, test_df


def sum_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds sum features for variable number of addends.'''

    if kwargs['n_addends'] > len(features):
        n_addends=len(features)

    else:
        n_addends=kwargs['n_addends']

    new_test_features={}
    new_train_features={}
    addend_sets=combinations(features, n_addends)

    for i, addend_set in enumerate(addend_sets):

        train_sum = [0]*len(train_df)
        test_sum = [0]*len(test_df)

        for addend in addend_set:

            train_sum += train_df[addend]
            test_sum += test_df[addend]

        new_train_features[f'sum_feature_{i}'] = train_sum
        new_test_features[f'sum_feature_{i}'] = test_sum

    new_train_df=pd.DataFrame.from_dict(new_train_features)
    new_test_df=pd.DataFrame.from_dict(new_test_features)

    train_df=pd.concat([train_df, new_train_df], axis=1)
    test_df=pd.concat([test_df, new_test_df], axis=1)

    return train_df, test_df


def difference_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Adds difference features for variable number of subtrahends.'''

    if kwargs['n_subtrahends'] > len(features):
        n_subtrahends=len(features)

    else:
        n_subtrahends=kwargs['n_subtrahends']

    new_test_features={}
    new_train_features={}
    subtrahend_sets=combinations(features, n_subtrahends)

    for i, subtrahend_set in enumerate(subtrahend_sets):

        train_difference = train_df[subtrahend_set[0]]
        test_difference = test_df[subtrahend_set[0]]

        for subtrahend in subtrahend_set[1:]:

            train_difference -= train_df[subtrahend]
            test_difference -= test_df[subtrahend]

        new_train_features[f'difference_feature_{i}'] = train_difference
        new_test_features[f'difference_feature_{i}'] = test_difference

    new_train_df=pd.DataFrame.from_dict(new_train_features)
    new_test_df=pd.DataFrame.from_dict(new_test_features)

    train_df=pd.concat([train_df, new_train_df], axis=1)
    test_df=pd.concat([test_df, new_test_df], axis=1)

    return train_df, test_df
