'''Configuration and parameters for dataset specific ensembleset
optimization runs with Optuna.'''

import os

#############################################################
# Optuna RDB credentials ####################################
#############################################################

# Provide the following via environment variables:
USER = os.environ['POSTGRES_USER']
PASSWD = os.environ['POSTGRES_PASSWD']
HOST = os.environ['POSTGRES_HOST']
PORT = os.environ['POSTGRES_PORT']

STORAGE = f'postgresql://{USER}:{PASSWD}@{HOST}:{PORT}'


#############################################################
# Optuna optimization parameters ############################
#############################################################

N_TRIALS = 1000
N_DATASETS = (2, 100)
FRAC_FEATURES = (0.01, 1.0)
N_STEPS = (1, 3)


#############################################################
# Kaggle calories dataset parameters ########################
#############################################################

CALORIES_DATA_PATH = 'optimization/data/kaggle_calories.csv'
CALORIES_LABEL = 'Calories'
CALORIES_STRING_FEATURES = ['Sex']
CALORIES_DROP_FEATURES = ['id']
CALORIES_SCORING = 'neg_mean_squared_log_error'
CALORIES_LOSS = 'gamma'
