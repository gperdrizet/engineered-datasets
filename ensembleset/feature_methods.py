'''Collection of functions to run feature engineering operations.'''

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, SplineTransformer


def onehot_encoding(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's one hot encoder.'''

    if kwargs is None:
        encoder=OneHotEncoder()
    else:
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
        kwargs:dict=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's label encoder.'''

    if kwargs is None:
        encoder=OrdinalEncoder()
    else:
        encoder=OrdinalEncoder(**kwargs)

    train_df[features]=encoder.fit_transform(train_df[features])

    if test_df is not None:
        test_df[features]=encoder.transform(test_df[features])

    return train_df, test_df


def poly_features(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        features:list,
        kwargs:dict=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's polynomial feature transformer..'''

    if kwargs is None:
        transformer=PolynomialFeatures()
    else:
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
        kwargs:dict=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    '''Runs sklearn's polynomial feature transformer..'''

    if kwargs is None:
        transformer=SplineTransformer()
    else:
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
