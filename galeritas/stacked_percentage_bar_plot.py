import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["stacked_percentage_bar_plot"]


def stacked_percentage_bar_plot(
        data,
        categorical_feature,
        hue,
        plot_title,
        hue_labels=None,
        annotate=False,
        colors=None,
        color_palette='pastel',
        figsize=(16, 7),
        **legend_kwargs
):
    """
    Generates a stacked percentage bar plot.
    It will generate a bar for each given category and inside each bar, will stack each group on the top of the other,
    showing each group representation (proportionally) for each category.

    :param data: A dataframe containing the dataset.
    :type data: DataFrame

    :param categorical_feature: A string indicating the dataframe's column name that will be used to create each plot's bar representing a category.
    :type categorical_feature: str

    :param hue: A string indicating the dataframe's column name of the groups that will be stack for each category.
    :type hue: str

    :param plot_title: Text to describe the plot's title
    :type plot_title: str

    :param hue_labels: A dictionary describing the labels of each hue group. If None, uses the values of the hue group in the dataframe. |default| :code:`None`
    :type hue_labels: dict

    :param annotate: If True, shows the amount of rows of each hue group inside each category. |default| :code:`False`
    :type annotate: bool, optional

    :param colors: A list containing the hexadecimal colors of each hue. The number of elements on the list must be the same of hue groups. |default| :code:`None`
    :type colors: list of str, optional

    :param color_palette:  If colors parameter is None, uses the color_palette to set different colors of the palette for each hue value. |default| :code:`'pastel'`
    :type color_palette: str, optional

    :param figsize: A tuple that indicates the figure size (respectively, width and height in inches). |default| :code:`(16, 7)`
    :type figsize: tuple, optional

    :param legend_kwargs: Matplotlib.pyplot's legend arguments such as *bbox_to_anchor* and *ncol*. Further informations `here <http://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend>`_.
    :type legend_kwargs: key, value mappings

    :return: Returns the figure object with the plot
    :rtype: Figure

    """

    crosstab_df, percentage_crosstab_df = _calculate_percentages(data, categorical_feature, hue)
    categories_names = list(percentage_crosstab_df.columns)
    label_names = tuple(percentage_crosstab_df.index)
    bar_bottom_position = np.zeros(percentage_crosstab_df.shape[0])

    if colors is None:
        colors = sns.color_palette(color_palette, len(categories_names))

    colormap = dict(zip(categories_names, colors))
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    for n_category, line_category in enumerate(categories_names):
        _plot_stacked_percentage_bars(
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

    if bool(legend_kwargs) is True:
        ax.legend(**legend_kwargs)
    else:
        ax.legend(bbox_to_anchor=(1.11, 1.01))

    plt.grid(True, alpha=0.9, linestyle='--', axis='y')
    
    plt.close()

    return fig


def _plot_stacked_percentage_bars(
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

    fig = plt.bar(
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
        _annotate_plot(fig, bar_bottom_position, crosstab_df, n_category)

    bar_bottom_position += percentage_crosstab_df[line_category]


def _calculate_percentages(data, categorical_feature, hue):
    crosstab_df = pd.crosstab(data[categorical_feature].astype('str'), data[hue])
    percentage_crosstab_df = 100 * crosstab_df.div(crosstab_df.sum(axis=1), axis=0)

    return crosstab_df, percentage_crosstab_df


def _annotate_plot(fig, bar_bottom_position, crosstab_df, n_category):
    for ix, bar in enumerate(fig):
        if bar.get_height() > 1:
            x_position = bar.get_x() + bar.get_width() / 2
            y_position = bar.get_height() / 2 + bar_bottom_position[ix] - 2.0
            plt.annotate(
                crosstab_df.iloc[ix, n_category],
                xy=(x_position, y_position),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom"
            )