'''Plotting functions for notebooks.'''

import matplotlib.pyplot as plt


def model_eval(plot_title, feature_name, predictions, labels):
    '''Plots true vs predicted values and residuals vs
    true values'''

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    fig.suptitle(plot_title)

    axs[0].set_title('Test set predictions')
    axs[0].scatter(labels, predictions, color='black', s=0.5)
    axs[0].set_xlabel(f'true {feature_name}')
    axs[0].set_ylabel(f'predicted {feature_name}')

    axs[1].set_title('Test set residuals')
    axs[1].scatter(labels, labels - predictions, color='black', s=0.5)
    axs[1].set_xlabel(f'true {feature_name}')
    axs[1].set_ylabel(f'{feature_name} (true - predicted)')

    fig.tight_layout()
