import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from galeritas.utils.creditas_palette import get_palette
import seaborn as sns
import warnings

__all__ = ["plot_ecdf_curve"]


def plot_ecdf_curve(
        df,
        column_to_plot,
        drop_na=True,
        hue=None,
        hue_labels=None,
        colors=None,
        color_palette=None,
        plot_title=None,
        percentiles=(25, 50, 75),
        percentiles_title='Percentiles',
        mark_percentiles=True,
        show_percentile_table=False,
        figsize=(16, 7),
        ax=None,
        return_fig=False,
        **legend_kwargs):
    """
    Generates an empirical cumulative distribution function.
    Theorical Reference can be found `here <https://en.wikipedia.org/wiki/Empirical_distribution_function>`__.

    :param df: A dataframe containing the dataset.
    :type df: DataFrame

    :param column_to_plot: Column name of the observed data.
    :type column_to_plot: str

    :param drop_na: If True, removes the missing values of the column to be plotted. Otherwise, plots the distribution without removing the missing values, but doesn't calculates the percentiles. |default| :code:`True`
    :type drop_na: bool, optional

    :param hue: A string indicating the dataframe's column name containing the categories if is wanted to plot the distribution using the column passed by column_to_plot parameter for each category that appears at the column passed by hue parameter. |default| :code:`None`
    :type hue: str, optional

    :param hue_labels: Parameter to be used if is wanted to show a label of hue categories different from the actual values existing in the column passed by hue parameter. It's necessary to pass a dictionary containing the values to be replaced and the values that will replace them (e.g. {1:'True', 0: 'False'}). |default| :code:`None`
    :type hue_labels: Dict, optional

    :param colors: A list containing the hexadecimal colors of each hue. The number of elements on the list must be the same of hue groups. |default| :code:`None`
    :type colors: list of str, optional

    :param color_palette:  If colors parameter is None, uses the color_palette to set different colors of the palette for each hue value.  If both colors and color_palette parameters are None, then uses the default palette of the library. |default| :code:`None`
    :type color_palette: str, optional

    :param plot_title: Text to describe the plot's title. |default| :code:`None`
    :type plot_title: str, optional

    :param percentiles: A tuple that indicates the percentiles of the distributions. |default| :code:`(25, 50, 75)`
    :type percentiles: tuple, optional

    :param percentiles_title: A string to be used to indicate the percentiles. |default| :code:`Percentiles`
    :type percentiles_title: str, optional

    :param mark_percentiles: If True, shows the percentiles defined in parameter percentiles. |default| :code:`True`
    :type mark_percentiles: bool, optional

    :param show_percentile_table: If True, shows a table with the values for each percentile and category. |default| :code:`False`
    :type show_percentile_table: bool, optional

    :param figsize: A tuple that indicates the figure size (respectively, width and height in inches). |default| :code:`(16, 7)`
    :type figsize: tuple, optional

    :param ax: Custom figure axes to plot. |default| :code: `None`
    :type ax: matplotlib.axes, optional

    :param return_fig: If True return figure object. |default| :code:`Fase`
    :type return_fig: bool, optional

    :param legend_kwargs: Matplotlib.pyplot's legend arguments such as *bbox_to_anchor* and *ncol*. Further informations `here <http://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend>`__.
    :type legend_kwargs: key, value mappings

    :return: Returns the figure object with the plot
    :rtype: Figure
    """
    data = df.copy()

    if data[column_to_plot].isnull().values.any():
        warnings.warn(f'Column "{column_to_plot}" has missing values! If the parameter drop_na is True (which is the '
                      f'default value), the missing values will be removed.')

    if drop_na:
        data = data.dropna(subset=[column_to_plot])
    
    if ax:
        axes=ax
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        fig.subplots_adjust(hspace=0.5)

    data_list = [data]

    x_values_list = []
    y_values_list = []
    x_values, y_values = calculate_ecdf_plot_axis_values(data[column_to_plot])
    x_values_list.append(x_values)
    y_values_list.append(y_values)

    if hue_labels is not None:
        data[hue] = data[hue].apply(
            lambda category: hue_labels[category] if category in hue_labels.keys() else category
        )

    if hue:
        data[hue] = data[hue].astype('str')
        data = data.sort_values(by=hue)
        hue_categories_labels = data[hue].unique()

        for _, category in enumerate(hue_categories_labels):
            df_hue = data.loc[data[hue] == category]
            data_list.append(df_hue)
            x_values, y_values = calculate_ecdf_plot_axis_values(df_hue[column_to_plot])
            x_values_list.append(x_values)
            y_values_list.append(y_values)

        data_list.pop(0)
        x_values_list.pop(0)
        y_values_list.pop(0)
    else:
        hue_categories_labels = [column_to_plot]

    if colors is not None and len(hue_categories_labels) > len(colors):
        raise KeyError(f'The number of colors passed by colors parameter is smaller than the number of categories in "{hue}" column! Expected {len(hue_categories_labels)} colors but only {len(colors)} was/were passed.')

    if colors is None:
        colors = get_palette()

    if color_palette:
        colors = sns.color_palette(color_palette, len(hue_categories_labels))

    colormap = dict(zip(hue_categories_labels, colors))

    for coordinates in enumerate(list(zip(x_values_list, y_values_list))):
        index = coordinates[0]
        x_values, y_values = coordinates[1]
        axes.plot(x_values, y_values, marker='.', markersize=4.5, alpha=0.7, linestyle='none',
                  label=hue_categories_labels[index], color=colormap[hue_categories_labels[index]])

    axes.set_title(plot_title)
    axes.set_ylabel("ECDF")
    axes.set_xlabel(column_to_plot)

    if mark_percentiles:
        percentiles_values = []
        for index, data in enumerate(data_list):
            percentiles_calc = np.percentile(data[column_to_plot], percentiles)
            axes.plot(
                percentiles_calc,
                np.divide(percentiles, 100),
                marker='D',
                markersize=8,
                color=colormap[hue_categories_labels[index]],
                linestyle='none',
                label=f'{percentiles_title} - {hue_categories_labels[index]}'
            )
            percentiles_values.append(percentiles_calc)

        axes.text(
            0.05,
            -0.15,
            f"{percentiles_title} {percentiles}",
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(boxstyle='round', alpha=0.25, facecolor='gray')
        )

    plt.grid(True, alpha=0.6, linestyle='--')

    axes.legend(loc="lower right")

    if bool(legend_kwargs) is True:
        axes.legend(**legend_kwargs)

    if show_percentile_table:
        columns = list(zip([percentiles_title] * len(percentiles), list(percentiles)))
        columns = pd.MultiIndex.from_tuples(columns)
        tabela = pd.DataFrame(percentiles_values, index=hue_categories_labels, columns=columns)

        display(tabela)

    if return_fig:
        plt.show()
        plt.close()

        return fig


def calculate_ecdf_plot_axis_values(data):
    data_length = len(data)
    x = np.sort(data)
    y = np.arange(1, data_length + 1) / data_length

    return x, y
