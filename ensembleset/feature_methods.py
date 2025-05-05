'''Collection of functions to run feature engineering operations.'''

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, SplineTransformer


def onehot_encoding(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict={'sparse_output': False}
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
        kwargs:dict={
            'handle_unknown': 'use_encoded_value',
            'unknown_value': np.nan  
        }
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

    encoded_data=transformer.fit_transform(train_df[features])
    encoded_df=pd.DataFrame(encoded_data, columns=transformer.get_feature_names_out())
    train_df.drop(features, axis=1, inplace=True)
    train_df=pd.concat([train_df, encoded_df], axis=1)

    if test_df is not None:
        encoded_data=transformer.transform(test_df[features])
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

    encoded_data=transformer.fit_transform(train_df[features])
    encoded_df=pd.DataFrame(encoded_data, columns=transformer.get_feature_names_out())
    train_df.drop(features, axis=1, inplace=True)
    train_df=pd.concat([train_df, encoded_df], axis=1)

    if test_df is not None:
        encoded_data=transformer.transform(test_df[features])
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
            scaler=MinMaxScaler()

            train_df[feature]=scaler.fit_transform(train_df[feature].to_frame())
            test_df[feature]=scaler.fit_transform(test_df[feature].to_frame())

        if kwargs['base'] == '2':
            train_df[feature]=np.log2(train_df[feature])
            test_df[feature]=np.log2(test_df[feature])

        if kwargs['base'] == 'e':
            train_df[feature]=np.log(train_df[feature])
            test_df[feature]=np.log(test_df[feature])

        if kwargs['base'] == '10':
            train_df[feature]=np.log10(train_df[feature])
            test_df[feature]=np.log10(test_df[feature])

    return train_df, test_df
