# EnsembleSet

[![PyPI release](https://github.com/gperdrizet/ensembleset/actions/workflows/publish_pypi.yml/badge.svg)](https://github.com/gperdrizet/ensembleset/actions/workflows/publish_pypi.yml) [![Python CI](https://github.com/gperdrizet/ensembleset/actions/workflows/python_ci.yml/badge.svg)](https://github.com/gperdrizet/ensembleset/actions/workflows/python_ci.yml)[![Devcontainer](https://github.com/gperdrizet/ensembleset/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/gperdrizet/ensembleset/actions/workflows/codespaces/create_codespaces_prebuilds)

EnsembleSet generates dataset ensembles by applying a randomized sequence of feature engineering methods to a randomized subset of input features.

## 1. Installation

Install the pre-release alpha from PyPI with:

```bash
pip install ensembleset
```

## 2. Usage

See the [example usage notebook](https://github.com/gperdrizet/ensembleset/blob/main/examples/regression_calorie_burn.ipynb).

Initialize an EnsembleSet class instance, passing in the label name a training DataFrame. Optionally, include a test DataFrame and/or list of any string features. Then call the `make_datasets()` to generate an EnsembleSet, specifying:

1. The number of individual datasets to generate.
2. The number of features to randomly select for each feature engineering step.
3. The number of feature engineering steps to run.

```python
import ensembleset.dataset as ds

data_ensemble=ds.DataSet(
    label='label_column_name',
    train_data=train_df,
    test_data=test_df
    string_features=['string_feature_column_names']
)

data_ensemble.make_datasets(
    n_datasets=10,
    n_features=7,
    n_steps=5
)
```

By default, generated datasets will be saved to HDF5 in `data/dataset.h5` using the following structure:

```text
dataset.h5
├──train
│   ├── labels
|   ├── 1
|   ├── .
|   ├── .
|   ├── .
|   └── n
│
└──test
    ├── labels
    ├── 1
    ├── .
    ├── .
    ├── .
    └── n
```
