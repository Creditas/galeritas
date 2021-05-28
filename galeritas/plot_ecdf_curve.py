import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "plot_ecdf_curve"
]


def plot_ecdf_curve(
        data,
        column_to_plot,
        plot_title,
        hue=None,
        labels=[1, 0],
        figsize=(16, 6),
        percentiles=(25, 50, 75),
        mark_percentiles=True
):
    """
    Generates an empirical cumulative distribution function.

    Theorical Reference: https://en.wikipedia.org/wiki/Empirical_distribution_function

    :param data: A dataframe containing the dataset.
    :type data: DataFrame

    :param column_to_plot: Column name of the observed data.
    :type column_to_plot: str

    :param hue: Text to describe the category.
    :type hue: str

    :param plot_title: Text to describe the plot's title.
    :type plot_title: str

    :param labels: Possible classes of the binary target. It expects the positive class followed by the negative class.
    :type labels: list, optional

    :param figsize: A tuple that indicates the figure size (respectively, width and height in inches). |default| :code:`(16, 7)`
    :type figsize: tuple, optional

    :param percentiles: A tuple that indicates the percentiles |default| :code:`(25, 50, 75)`
    :type percentiles: tuple, optional

    :param mark_percentiles: If True, shows the percentiles defined in param percentiles. |default| :code:`True`
    :type mark_percentiles: bool, optional

    """

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    fig.subplots_adjust(hspace=0.5)

    data_list = [data]

    x_values_list = []
    y_values_list = []
    x_values, y_values = calculate_ecdf_plot_axis_values(data[column_to_plot])
    x_values_list.append(x_values)
    y_values_list.append(y_values)

    if hue:
        for _, category in enumerate(data[hue].unique()):
            df_hue = data.loc[data[hue] == category]
            data_list.append(df_hue)
            x_values, y_values = calculate_ecdf_plot_axis_values(df_hue[column_to_plot])
            x_values_list.append(x_values)
            y_values_list.append(y_values)

        data_list.pop(0)
        x_values_list.pop(0)
        y_values_list.pop(0)

    for coordinates in enumerate(list(zip(x_values_list, y_values_list))):
        index = coordinates[0]
        x_values, y_values = coordinates[1]
        axes.plot(x_values, y_values, marker='.', alpha=0.7, linestyle='none', label=labels[index])

    axes.set_title(plot_title, weight='bold', fontsize=13)
    axes.set_ylabel("ECDF")

    if mark_percentiles:
        for index, data in enumerate(data_list):
            percentiles_calc = np.percentile(data[column_to_plot], percentiles)
            axes.plot(
                percentiles_calc,
                np.divide(percentiles_calc, 100),
                marker='D',
                color='orange',
                linestyle='none',
                label=f'Percentiles {percentiles_calc} - {labels[index]}'
            )

    axes.legend(loc="upper left")
    plt.close()

    return fig


def calculate_ecdf_plot_axis_values(data):
    data_length = len(data)
    x = np.sort(data)
    y = np.arange(1, data_length + 1) / data_length

    return x, y
