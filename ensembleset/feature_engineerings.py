'''Dictionaries containing feature engineering methods for string in
numeric features and parameter argument options.'''

import numpy as np

# String feature encoding methods
STRING_ENCODINGS={
    'onehot_encoding': {'sparse_output': False},
    'ordinal_encoding': {
        'handle_unknown': 'use_encoded_value',
        'unknown_value': np.nan  
    }
}

# Numeric feature engineering methods
NUMERICAL_METHODS={
    'poly_features': {
        'degree': [2, 3],
        'interaction_only': [True, False],
        'include_bias': [True, False]
    },
    'spline_features': {
        'n_knots': [5],
        'degree': [2, 3, 4],
        'knots': ['uniform', 'quantile'],
        'extrapolation': ['constant', 'linear', 'continue', 'periodic'],
        'include_bias': [True, False]
    },
    'log_features': {
        'base': ['2', 'e', '10']
    },
    'ratio_features': {
        'div_zero_value': [np.nan]
    },
    'exponential_features': {
        'base': ['2', 'e']
    },
    'sum_features': {
        'n_addends': [2,3,4]
    },
    'difference_features': {
        'n_subtrahends': [2,3,4]
    },
    'kde_smoothing': {
        'bandwidth': ['scott', 'silverman'],
        'sample_size': [1000]
    },
    'kbins_quantization': {
        'n_bins': [4,8,16],
        'encode': ['ordinal'],
        'strategy': ['uniform', 'quantile', 'kmeans']
    }
}
