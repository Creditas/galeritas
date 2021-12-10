import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from galeritas.utils.creditas_palette import get_palette

__all__ = ["stacked_percentage_bar_plot"]


def stacked_percentage_bar_plot(
        df,
        categorical_feature,
        hue,
        hue_labels=None,
        plot_title=None,
        annotate=False,
        show_na=True,
        na_label='Null',
        colors=None,
        color_palette=None,
        figsize=(16, 7),
        ax=None,
        return_fig=False,
        **legend_kwargs
):
    """
    Generates a stacked percentage bar plot.
    It will generate a bar for each given category and inside each bar, will stack each group on the top of the other,
    showing each group representation (proportionally) for each category.

    :param df: A dataframe containing the dataset.
    :type df: DataFrame

    :param categorical_feature: A string indicating the dataframe's column name that will be used to create each plot's bar representing a category.
    :type categorical_feature: str

    :param hue: A string indicating the dataframe's column name of the groups that will be stack for each category.
    :type hue: str

    :param hue_labels: A dictionary describing the labels of each hue group. If None, uses the values of the hue group in the dataframe. |default| :code:`None`
    :type hue_labels: dict, optional

    :param plot_title: Text to describe the plot's title. |default| :code:`None`
    :type plot_title: str, optional

    :param annotate: If True, shows the amount of rows of each hue group inside each category. |default| :code:`False`
    :type annotate: bool, optional

    :param show_na: If True, shows the missing values for both hue group and categories. |default| :code:`True`
    :type show_na: bool, optional

    :param na_label: The label used to identify the missing values. |default| :code:`'Null'`
    :type na_label: str, optional

    :param colors: A list containing the hexadecimal colors of each hue. The number of elements on the list must be the same of hue groups. |default| :code:`None`
    :type colors: list of str, optional

    :param color_palette:  If colors parameter is None, uses the color_palette to set different colors of the palette for each hue value. If both colors and color_palette parameters are None, then uses the default palette of the library. |default| :code:`'pastel'`
    :type color_palette: str, optional

    :param figsize: A tuple that indicates the figure size (respectively, width and height in inches). |default| :code:`(16, 7)`
    :type figsize: tuple, optional

    :param return_fig: If True return figure object. |default| :code:`True`
    :type return_fig: bool, optional

    :param legend_kwargs: Matplotlib.pyplot's legend arguments such as *bbox_to_anchor* and *ncol*. Further informations `here <http://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend>`_.
    :type legend_kwargs: key, value mappings

    :return: Returns the figure object with the plot (*return_fig parameter needs to be set)
    :rtype: Figure

    """
    data = df.copy()

    if show_na:
        data[categorical_feature] = data[categorical_feature].fillna(na_label)
        data[hue] = data[hue].fillna(na_label)
    else:
        data = data.dropna(subset=[categorical_feature, hue])

    crosstab_df, percentage_crosstab_df = _calculate_percentages(data, categorical_feature, hue)
    categories_names = list(percentage_crosstab_df.columns)
    label_names = tuple(percentage_crosstab_df.index)
    bar_bottom_position = np.zeros(percentage_crosstab_df.shape[0])

    if colors is not None and len(categories_names) > len(colors):
        raise KeyError(f'The number of colors passed by colors parameter is smaller than the number of categories in "{hue}" column! Expected {len(categories_names)} colors but only {len(colors)} was/were passed.')

    if colors is None:
        colors = get_palette(n_colors=len(categories_names))

    if color_palette:
        colors = sns.color_palette(color_palette, len(categories_names))

    colormap = dict(zip(categories_names, colors))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=120)

    for n_category, line_category in enumerate(categories_names):
        _plot_stacked_percentage_bars(
            ax,
            n_category,
            line_category,
            percentage_crosstab_df,
            crosstab_df,
            label_names,
            colormap,
            hue_labels,
            bar_bottom_position,
            annotate
        )

    ax.set_title(plot_title)
    ax.set_xlabel(categorical_feature)
    ax.set_ylabel("%")

    ax.legend(loc='upper right', **legend_kwargs)

    plt.grid(True, alpha=0.9, linestyle='--', axis='y')

    if return_fig:
        plt.show()
        plt.close()

        return fig



def _plot_stacked_percentage_bars(
        ax,
        n_category,
        line_category,
        percentage_crosstab_df,
        crosstab_df,
        label_names,
        colormap,
        hue_labels,
        bar_bottom_position,
        annotate
        ):
    if hue_labels:
        label = hue_labels[line_category]
    else:
        label = line_category

    bar = ax.bar(
        label_names,
        percentage_crosstab_df[line_category],
        color=colormap[line_category],
        bottom=bar_bottom_position,
        edgecolor='white',
        alpha=0.7,
        width=0.85,
        label=label
    )

    if annotate:
        _annotate_plot(bar, bar_bottom_position, crosstab_df, n_category, ax)

    bar_bottom_position += percentage_crosstab_df[line_category]


def _calculate_percentages(data, categorical_feature, hue):
    crosstab_df = pd.crosstab(data[categorical_feature].astype('str'), data[hue])
    percentage_crosstab_df = 100 * crosstab_df.div(crosstab_df.sum(axis=1), axis=0)

    return crosstab_df, percentage_crosstab_df


def _annotate_plot(fig, bar_bottom_position, crosstab_df, n_category, ax=None):
    for ix, bar in enumerate(fig):
        if bar.get_height() > 1:
            x_position = bar.get_x() + bar.get_width() / 2
            y_position = bar.get_height() / 2 + bar_bottom_position[ix] - 2.0

            ax.annotate(
                crosstab_df.iloc[ix, n_category],
                xy=(x_position, y_position),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom"
            )
