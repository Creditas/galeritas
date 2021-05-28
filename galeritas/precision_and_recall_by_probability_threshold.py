import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mstats

from galeritas.precision_recall_threshold_confidence_interval_aux import (threshold_confidence_interval,
                                                                _get_threshold_metrics_intervals)

from galeritas.utils.creditas_palette import get_palette


sns.set_palette(get_palette())


def plot_precision_and_recall_by_probability_threshold(predictions_with_target, target_name, target=1, n_trials=50, sample_size_percent=0.5,
                                         quantiles=[0.05, 0.5, 0.95], figsize=(20, 10), title=None, thresholds_to_highlight=None):
    """
        Determines precision, recall e support scores for different thresholds for the positive class, using a data sample with
        replacement.
        Adapted from Insight Data Science's post:
        https://blog.insightdatascience.com/visualizing-machine-learning-thresholds-to-make-better-business-decisions-4ab07f823415
        Parameters
        ----------
        test_data : DataFrame, shape (n_samples, n_features)
            Test dataframe, where n_samples is the number of samples and
            n_features is the number of features. Has to contain the target variable;
        target_name : string
            String that indicates the target name
        model : object type that implements the "fit" and "predict" methods.
        target: int
            Indicates the target class.
        n_trials : int
            Indicates the number of times to resample the data and make predictions.
        test_size_percent : float
            Indicates the percentage of the dataset that needs to be used to perform the sample data
        quantiles : array
            Indicates the upper, median and lower quantiles to be used to plot the graph.
        thresholds_to_highlight: array
            Indicates the score(s) where the thresholds will be drawn
    """

    uniform_precision_plots, uniform_recall_plots, uniform_support_plots, uniform_support_rate_plots, uniform_thresholds \
        = threshold_confidence_interval(
        predictions_with_target,
        target_name,
        target=target,
        n_trials=n_trials,
        sample_size_percent=sample_size_percent
    )

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

    base_fontsize = figsize[0]
    leg = plt.legend(
        ('precision', 'recall', 'support'), frameon=True, fontsize=0.75*base_fontsize
    )
    leg.get_frame().set_edgecolor('k')
    plt.text(
        0.05,
        -0.15,
        f"Confidence Interval: {confidence_interval}%",
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=0.75*base_fontsize,
        bbox=dict(boxstyle='round', alpha=0.25, facecolor='gray')
    )
    if title is None:
        title = "Precision, Recall and Support by model probability threshold"

    ax.set_title(title, fontsize=1.25*base_fontsize, pad=30)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=0.75*base_fontsize)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=0.75*base_fontsize)
    ax.set_xlabel("Model probability threshold", fontsize=base_fontsize)
    ax.set_ylabel("Metric's Ratio", fontsize=base_fontsize)
    plt.grid(alpha=0.5)

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
        print(thresholds_metrics_dataframe)

    plt.close()

    return fig

