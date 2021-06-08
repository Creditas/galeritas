import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mstats

from galeritas.precision_recall_threshold_confidence_interval_aux import (threshold_confidence_interval,
                                                                          _get_threshold_metrics_intervals)

from galeritas.utils.creditas_palette import get_palette

sns.set_palette(get_palette())


def plot_precision_and_recall_by_probability_threshold(
        df,
        target_name,
        prediction_column_name,
        target=1,
        n_trials=50,
        sample_size_percent=0.5,
        quantiles=[0.05, 0.5, 0.95],
        x_label="Model probability threshold",
        y_label="Metric's Ratio",
        figsize=(16, 7),
        plot_title=None,
        thresholds_to_highlight=None,
        **legend_kwargs):
    """
    Determines precision, recall e support scores for different thresholds for the positive class, using a data sample with
    replacement.
    Adapted from Insight Data Science's post:
    https://blog.insightdatascience.com/visualizing-machine-learning-thresholds-to-make-better-business-decisions-4ab07f823415

    :param df: Dataframe containing predictions and target columns.
    :type df: DataFrame

    :param target_name: String that indicates the target name.
    :type target_name: str

    :param prediction_column_name: String that indicates the name of the columns where the predictions are.
    :type prediction_column_name: str

    :param target: Indicates the target class. |default| :code:`1`
    :type target: int, optional

    :param n_trials: Indicates the number of times to resample the data and make predictions. |default| :code:`50`
    :type n_trials: int, optional

    :param sample_size_percent: Indicates the percentage of the dataset that needs to be used to perform the sample data. |default| :code:`0.5`
    :type sample_size_percent: float, optional

    :param quantiles: Indicates the upper, median and lower quantiles to be used to plot the graph. |default| :code:`[0.05, 0.5, 0.95]`
    :type quantiles: list, optional

    :param x_label: Text to describe the x-axis label. |default| :code:`"Model probability threshold"`
    :type x_label: str, optional

    :param y_label: Text to describe the y-axis label. |default| :code:`"Metric's Ratio"`
    :type y_label: str, optional

    :param figsize: A tuple that indicates the figure size (respectively, width and height in inches). |default| :code:`(16, 7)`
    :type figsize: tuple, optional

    :param plot_title: Text to describe the plot's title. |default| :code:`None`
    :type plot_title: str, optional

    :param thresholds_to_highlight: Indicates the score(s) where the thresholds will be drawn. |default| :code:`None`
    :type thresholds_to_highlight: list, optional

    :param legend_kwargs: Matplotlib.pyplot's legend arguments such as *bbox_to_anchor* and *ncol*. Further informations `here <http://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend>`__.
    :type legend_kwargs: key, value mappings

    :return: Returns the figure object with the plot
    :rtype: Figure
    """

    uniform_precision_plots, uniform_recall_plots, uniform_support_plots, uniform_support_rate_plots, uniform_thresholds \
        = threshold_confidence_interval(df,
                                        target_name,
                                        prediction_column_name,
                                        target=target,
                                        n_trials=n_trials,
                                        sample_size_percent=sample_size_percent)

    confidence_interval = 100 * round(quantiles[-1] - quantiles[0], 2)
    lower_precision, median_precision, upper_precision = mstats.mquantiles(uniform_precision_plots, quantiles, axis=0)
    lower_recall, median_recall, upper_recall = mstats.mquantiles(uniform_recall_plots, quantiles, axis=0)
    lower_support, median_support, upper_support = mstats.mquantiles(uniform_support_plots,
                                                                     quantiles, axis=0)
    lower_support_rate, median_support_rate, upper_support_rate = mstats.mquantiles(uniform_support_rate_plots,
                                                                                    quantiles, axis=0)
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    plt.plot(uniform_thresholds, median_precision)
    plt.plot(uniform_thresholds, median_recall)
    plt.plot(uniform_thresholds, median_support_rate)

    plt.fill_between(uniform_thresholds, upper_precision, lower_precision, alpha=0.5, linewidth=0)
    plt.fill_between(uniform_thresholds, upper_recall, lower_recall, alpha=0.5, linewidth=0)
    plt.fill_between(uniform_thresholds, upper_support_rate, lower_support_rate, alpha=0.5, linewidth=0)

    plt.legend(
        ('precision', 'recall', 'support'), frameon=True, **legend_kwargs
    )

    plt.text(
        0.05,
        -0.15,
        f"Confidence Interval: {confidence_interval}%",
        horizontalalignment='center',
        verticalalignment='center',
        bbox=dict(boxstyle='round', alpha=0.25, facecolor='gray')
    )

    ax.set_title(plot_title, pad=30)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.grid(True, alpha=0.6, linestyle='--')

    if thresholds_to_highlight:
        label = "A"
        thresholds_metrics_summary_range = {}

        metrics_columns_range_order = [
            "probability_threshold",
            "Median Precision",
            "Precision Range",
            "Median Recall",
            "Recall Range",
            "Median Support",
            "Support Range",
            "Median Support Rate",
            "Support Rate Range"
        ]

        for highlight_threshold in thresholds_to_highlight:
            plt.plot(
                np.repeat(highlight_threshold, len(uniform_thresholds)), uniform_thresholds, "k--"
            )

            thresholds_metrics_summary_range[label] = _get_threshold_metrics_intervals(
                lower_precision,
                median_precision,
                upper_precision,
                lower_recall,
                median_recall,
                upper_recall,
                lower_support,
                median_support,
                upper_support,
                lower_support_rate,
                median_support_rate,
                upper_support_rate,
                highlight_threshold,
                uniform_thresholds
            )
            thresholds_metrics_summary_range[label]["probability_threshold"] = highlight_threshold

            plt.text(highlight_threshold - 0.006, 1.08, label)
            label = chr(ord(label) + 1)

        thresholds_metrics_dataframe = pd.DataFrame(thresholds_metrics_summary_range).T[metrics_columns_range_order]
        thresholds_metrics_dataframe.index.name = "Threshold Label"

        display(thresholds_metrics_dataframe)

    plt.close()

    return fig
