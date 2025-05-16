'''Optuna optimization of ensembleset generation parameters.'''

import os
import re
import io
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import h5py
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor

import configuration as config # pylint: disable=E0401

sys.path.append('../ensembleset')
from ensembleset.dataset import DataSet # pylint: disable=C0413


def optimization_run(
        study_name:str,
        raw_data_path:str,
        label:str,
        string_features:list,
        drop_features:list,
        scoring:str,
        loss:str
) -> None:

    '''Main function to orchestrate optimization run.'''

    Path('optimization/logs').mkdir(parents=True, exist_ok=True)
    delete_old_logs('optimization/logs', study_name)

    function_logger = logging.getLogger(__name__ + '.optimization_run')

    logging.basicConfig(
        handlers=[RotatingFileHandler(
            f'optimization/logs/{study_name}.log',
            maxBytes=100000, backupCount=10
        )],
        level=logging.DEBUG,
        format='%(levelname)s - %(name)s - %(message)s'
    )

    storage_name = f'{config.STORAGE}/{study_name}'

    study=optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=storage_name,
        load_if_exists=True
    )

    raw_data = pd.read_csv(raw_data_path)
    raw_data.drop(drop_features, axis=1, inplace=True, errors='ignore')
    training_data, validation_data = train_test_split(raw_data, test_size=0.5)

    function_logger.info('Optuna RDB storage: %s', storage_name)
    function_logger.info('Training data:')

    # Capture df.info() output
    buffer = io.StringIO()
    training_data.info(buf=buffer)
    info_output = buffer.getvalue()

    # Log the output
    for line in info_output.splitlines():
        function_logger.info(line)

    study.optimize(
        lambda trial: objective(
            trial,
            training_data,
            validation_data,
            label,
            string_features,
            scoring,
            loss
        ),
        n_trials=config.N_TRIALS
    )


def objective(
        trial:optuna.trial,
        train_df:pd.DataFrame,
        val_df:pd.DataFrame,
        label:str,
        string_features:list,
        scoring:str,
        loss:str
) -> float:

    '''Optuna objective function to optimize ensembleset generation parameters using
    performance of stage II ensemble model as a metric.'''

    objective_logger = logging.getLogger(__name__ + '.objective')

    data_ensemble = DataSet(
        label=label,
        train_data=train_df,
        test_data=val_df,
        string_features=string_features,
        data_directory='optimization/data'
    )

    n_datasets = trial.suggest_int('n_datasets', config.N_DATASETS[0], config.N_DATASETS[1])

    data_ensemble.make_datasets(
        n_datasets=n_datasets,
        frac_features=trial.suggest_float(
            'frac_features',
            config.FRAC_FEATURES[0],
            config.FRAC_FEATURES[1]
        ),
        n_steps=trial.suggest_int(
            'n_steps',
            config.N_STEPS[0],
            config.N_STEPS[1]
        )
    )

    stage_one_models = {}
    stage_one_test_predictions = {}

    with h5py.File('optimization/data/dataset.h5', 'r') as hdf:
        for i in range(n_datasets):

            objective_logger.info('Fitting model %s of %s', i+1, n_datasets)

            stage_one_models[i] = HistGradientBoostingRegressor(loss=loss)
            stage_one_models[i].fit(hdf[f'train/{i}'], hdf['train/labels'])
            stage_one_test_predictions[i] = stage_one_models[i].predict(hdf[f'test/{i}'])

        stage_two_training_df = pd.DataFrame.from_dict(stage_one_test_predictions)
        stage_two_training_df['labels'] = hdf['test/labels']

        objective_logger.info('Stage II training data')

        # Capture df.info() output
        buffer = io.StringIO()
        stage_two_training_df.info(buf=buffer)
        info_output = buffer.getvalue()

        # Log the output
        for line in info_output.splitlines():
            objective_logger.info(line)

        scores = cross_val_score(
            HistGradientBoostingRegressor(loss=loss),
            stage_two_training_df.drop('labels', axis=1),
            stage_two_training_df['labels'],
            scoring=scoring,
            n_jobs=-1,
            cv=7
        )

    cv_score_mean = np.mean(np.sqrt(-scores))
    cv_score_std = np.std(np.sqrt(-scores))

    objective_logger.info(
        'Cross-validation score: %s +/- %s',
        round(cv_score_mean, 4),
        round(cv_score_std, 4)
    )

    return cv_score_mean


def delete_old_logs(directory:str, basename:str) -> None:
    '''Deletes old log files from previous optimization runs on the
    same dataset.'''

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if re.search(basename, filename):
            os.remove(file_path)


if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        choices=['calories'],
        help='dataset to optimize on'
    )

    args=parser.parse_args()

    if args.dataset == 'calories':

        optimization_run(
            study_name='calories',
            raw_data_path=config.CALORIES_DATA_PATH,
            label=config.CALORIES_LABEL,
            string_features=config.CALORIES_STRING_FEATURES,
            drop_features=config.CALORIES_DROP_FEATURES,
            scoring=config.CALORIES_SCORING,
            loss=config.CALORIES_LOSS
        )
