from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt
import warnings
import numpy as np

__all__ = ["plot_calibration_and_distribution"]


def plot_calibration_and_distribution(
        df,
        target,
        predictions,
        n_bins=20,
        strategy='quantile',
        x_lim=None,
        y_lim=None,
        show_distribution=True,
        color="#3377bb",
        return_fig=False,
        ax=None):

    """
    Returns
        (1) a calibration curve
        (2) a distribuition plot
    for predicted values

    :param df: a pd.Dataframe that contains target and prediction data
    :type df: pd.Dataframe

    :param target: name of target column
    :type target: string

    :param predictions: name of prediction column
    :type predictions: string

    :param n_bins: number of bins to discretize the [0, x_lim] interval in calibration curve. |default| :code:`20`
    :type n_bins: int, optional

    :param strategy: strategy used in calibration curve: |default| :code:`quantile`
        uniform: the bins have identical widths.
        quantile: The bins have the same number of samples and depend on y_prob.
    :type strategy: string, optional

    :param x_lim: width of x axes in calibration and distribution curve. |default| :code:`None`
    :type x_lim: float, optional

    :param y_lim: width of y ax in calibration curve. |default| :code:`None`
    :type y_lim: float, optional

    :param show_distribution: if distribution graph is wanted |default| :code:`True`
    :type show_distribution: boolean, optional

    :param color: personalized color |default| :code:`#3377bb`
    :type color: str, optional

    :param return_fig: If True return figure object. |default| :code:`True`
    :type return_fig: bool, optional

    :return: Returns the figure object with the plot (*return_fig parameter needs to be set)
    :rtype: Figure

    """
    
    if show_distribution and ax:
        warnings.warn("`ax` is not None and `show_distribution` is True. Ignoring distribution for plotting in personalized axes. To see distribution don't use `ax` parameter.")
        show_distribution = False
    
    if ax:
        # used personalized axes (ax)
        ax1 = ax
        
    elif show_distribution:
        # create subplots for calibration curve and distribution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    else:
        # create only one plot with calibration curve
        fig, ax1 = plt.subplots(figsize=(20, 10))

    if strategy == 'uniform':
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    else:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(df[predictions], quantiles * 100)
        bins[-1] = bins[-1] + 1e-8

    binids = np.digitize(df[predictions], bins) - 1

    bin_sums = np.bincount(binids, weights=df[predictions], minlength=len(bins))
    bin_true = np.bincount(binids, weights=df[target], minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    if x_lim is None:
        x_lim = prob_pred.max() * 1.1

    if y_lim is None:
        y_lim = prob_true.max() * 1.1

    # plot perfectly calibrated
    ax1.plot([0, y_lim], [0, y_lim], label='Perfectly calibrated', linestyle='--', color='black')

    fop_calibrated, mpv_calibrated = calibration_curve(df[target], df[predictions], n_bins=n_bins,
                                                       strategy=strategy)
    ax1.plot(mpv_calibrated, fop_calibrated, marker='.', label=predictions, color=color)

    ax1.set_xlim([0, x_lim])
    ax1.set_ylim([0, y_lim])
    ax1.legend(loc="upper left")
    ax1.grid(True)
    ax1.set_xlabel('Mean prediction value')
    ax1.set_ylabel('Mean target value')

    if show_distribution:
        ax2.hist(df[predictions], histtype="bar", bins=bins, label=predictions, color=color)

        ax2.set_xlim([0, x_lim])
        ax2.legend(loc="upper left")
        ax2.set_xlabel('Mean prediction value')
        ax2.set_ylabel('Quantity')

    if return_fig:
        plt.show()
        plt.close()

        return fig