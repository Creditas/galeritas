import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mstats
from galeritas.utils.creditas_palette import get_palette
from galeritas.precision_recall_threshold_confidence_interval_aux import (threshold_confidence_interval,
                                                                          _get_threshold_metrics_intervals)


def plot_precision_and_recall_by_probability_threshold(
        df,
        prediction_column_name,
        target_name,
        target=1,
        n_trials=50,
        sample_size_percent=0.5,
        quantiles=[0.05, 0.5, 0.95],
        thresholds_to_highlight=None,
        x_label="Model probability threshold",
        y_label="Metric's Ratio",
        plot_title=None,
        colors=None,
        color_palette=None,
        figsize=(16, 7),
        ax=None,
        return_fig=False,
        **legend_kwargs):
    """
    Determines precision, recall e support scores for different thresholds for the positive class, using a data sample with
    replacement.

    Adapted from `Insight Data Science's post <https://blog.insightdatascience.com/visualizing-machine-learning-thresholds-to-make-better-business-decisions-4ab07f823415>`__.

    :param df: Dataframe containing predictions and target columns.
    :type df: DataFrame

    :param prediction_column_name: String that indicates the name of the columns where the predictions are.
    :type prediction_column_name: str

    :param target_name: String that indicates the target name.
    :type target_name: str

    :param target: Indicates the target class. |default| :code:`1`
    :type target: int, optional

    :param n_trials: Indicates the number of times to resample the data and make predictions. |default| :code:`50`
    :type n_trials: int, optional

    :param sample_size_percent: Indicates the percentage of the dataset that needs to be used to perform the sample data. |default| :code:`0.5`
    :type sample_size_percent: float, optional

    :param quantiles: Indicates the upper, median and lower quantiles to be used to plot the graph. |default| :code:`[0.05, 0.5, 0.95]`
    :type quantiles: list, optional

    :param thresholds_to_highlight: Indicates the score(s) where the thresholds will be drawn. |default| :code:`None`
    :type thresholds_to_highlight: list, optional

    :param x_label: Text to describe the x-axis label. |default| :code:`"Model probability threshold"`
    :type x_label: str, optional

    :param y_label: Text to describe the y-axis label. |default| :code:`"Metric's Ratio"`
    :type y_label: str, optional

    :param plot_title: Text to describe the plot's title. |default| :code:`None`
    :type plot_title: str, optional

    :param colors: A list containing the hexadecimal colors of each hue. The number of elements on the list must be the same of hue groups. |default| :code:`None`
    :type colors: list of str, optional

    :param color_palette: If this parameter is set, uses the color_palette to set different colors of the palette for each hue value. If both colors and color_palette parameters are None, uses Galeritas default palette. |default| :code:`None`
    :type color_palette: str, optional

    :param figsize: A tuple that indicates the figure size (respectively, width and height in inches). |default| :code:`(16, 7)`
    :type figsize: tuple, optional
    
    :param ax: Custom figure axes to plot. |default| :code: `None`
    :type ax: matplotlib.axes, optional

    :param return_fig: If True return figure object. |default| :code:`False`
    :type return_fig: bool, optional

    :param legend_kwargs: Matplotlib.pyplot's legend arguments such as *bbox_to_anchor* and *ncol*. Further informations `here <http://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend>`__.
    :type legend_kwargs: key, value mappings

    :return: Returns the figure object with the plot (*return_fig parameter needs to be set)
    :rtype: Figure
    """

    if target_name not in [col for col in df if np.isin(df[col].unique(), [0, 1]).all()]:
        raise ValueError(f'The target must be binary! Column "{target_name}" contains more values.')

    if colors is None:
        colors = get_palette()

    if color_palette:
        colors = sns.color_palette(color_palette, 3)

    if colors is not None and len(colors) < 3:
        raise KeyError(f'Expected 3 colors but only {len(colors)} was/were passed.')

    metric_names = ['precision', 'recall', 'support_rate']

    colormap = dict(zip(metric_names, colors))

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

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=120)

    ax.plot(uniform_thresholds, median_precision, color=colormap['precision'])
    ax.plot(uniform_thresholds, median_recall, color=colormap['recall'])
    ax.plot(uniform_thresholds, median_support_rate, color=colormap['support_rate'])

    ax.fill_between(uniform_thresholds, upper_precision, lower_precision, alpha=0.5, linewidth=0,
                     color=colormap['precision'])
    ax.fill_between(uniform_thresholds, upper_recall, lower_recall, alpha=0.5, linewidth=0, color=colormap['recall'])
    ax.fill_between(uniform_thresholds, upper_support_rate, lower_support_rate, alpha=0.5, linewidth=0,
                     color=colormap['support_rate'])

    ax.legend(
        ('precision', 'recall', 'support'), frameon=True, **legend_kwargs
    )

    ax.text(
        0.05,
        -0.15,
        f"Confidence Interval: {confidence_interval}%",
        horizontalalignment='center',
        verticalalignment='center',
        bbox=dict(boxstyle='round', alpha=0.25, facecolor='gray')
    )

    ax.set_title(plot_title, pad=30)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.grid(True, alpha=0.6, linestyle='--')

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
            ax.plot(
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

            ax.text(highlight_threshold - 0.006, 1.08, label)
            label = chr(ord(label) + 1)

        thresholds_metrics_dataframe = pd.DataFrame(thresholds_metrics_summary_range).T[metrics_columns_range_order]
        thresholds_metrics_dataframe.index.name = "Threshold Label"

        display(thresholds_metrics_dataframe)

    if return_fig:
        plt.show()
        plt.close()

        return fig
