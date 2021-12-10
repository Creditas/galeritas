import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from galeritas.utils.creditas_palette import get_palette

__all__ = ["bar_plot_with_population_proportion"]


def bar_plot_with_population_proportion(df, x, y,
                                        func=np.median,
                                        show_error_bar=True,
                                        show_na=True,
                                        na_label='Null',
                                        circle_diameter=150,
                                        split_variable=False,
                                        colors=None,
                                        color_palette=None,
                                        x_label=None,
                                        y_label=None,
                                        show_qty=True,
                                        qty_label='Quantity',
                                        proportion_label='Percentage',
                                        proportion_format='.2f',
                                        show_population_func=False,
                                        population_format='.0f',
                                        population_func_legend='Median population value',
                                        population_legend='Population %',
                                        up_label='Positive values',
                                        down_label='Negative values',
                                        plot_title=None,
                                        figsize=(16, 7),
                                        ax=None,
                                        return_fig=False,
                                        **legend_kwargs):
    """
    Produces a barplot with an additional dotplot showing the percentage of the dataset population for each category of
    the barplot.

    Sometimes it is useful to split the numeric variable of interest into positive and negative values and plot it as a
    function of the categorical variable.
    This can be controlled with the split_variable parameter.

    :param df: A dataframe containing the dataset.
    :type df: DataFrame

    :param x: A string indicating the dataframe's column name of the x-axis variable. It will be treated as a categorical variable.
    :type x: str

    :param y: A string indicating the dataframe's column name of the y-axis variable. It will be treated as a numeric variable in which an aggregation function (defined in func parameter) will be applied.
    :type y: str

    :param func: Aggregation function to be applied in the y-axis variable. The default function is to calculate the median, but other functions are accepted (see `here <http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html>`_). |default| :code:`np.median`
    :type func: function, optional

    :param show_error_bar: If True, shows the default confidence intervals estimated by Seaborn (for more information, see this `link <https://seaborn.pydata.org/generated/seaborn.barplot.html>`_ with Seaborn's barplot documentation). |default| :code:`True`
    :type show_error_bar: bool, optional

    :param show_na: If True, shows the missing values in the column passed by x parameter. |default| :code:`True`
    :type show_na: bool, optional

    :param na_label: The label used to identify the missing values in the column passed by x parameter. |default| :code:`'Null'`
    :type na_label: str, optional

    :param circle_diameter: Base circle diameter of the percentage dots. You might want to decrease it if there's a category in the x-axis variable that accounts a big proportion of the dataset (e.g. 80%). |default| :code:`150`
    :type circle_diameter: int, optional

    :param split_variable: If True, it splits the y-axis variable into positive and negative values, showing upward bars for positive values and downward bars for negative values. |default| :code:`False`
    :type split_variable: bool, optional

    :param colors: A list containing the hexadecimal colors of each hue. The number of elements on the list must be the same of hue groups. |default| :code:`None`
    :type colors: list of str, optional

    :param color_palette: If this parameter is set, uses the color_palette to set different colors of the palette for each hue value. If both colors and color_palette parameters are None, uses Galeritas default palette. |default| :code:`None`
    :type color_palette: str, optional

    :param x_label: Text to describe the x-axis label. If None, the x value is used. |default| :code:`None`
    :type x_label: str, optional

    :param y_label: Text to describe the y-axis label. If None, the y value is used. |default| :code:`None`
    :type y_label: str, optional

    :param show_qty: If True, shows the quantity of the population for each category below its percentage. |default| :code:`True`
    :type show_qty: bool, optional

    :param qty_label: Sets the label of the quantity that will appear at the right side of the plot. |default| :code:`'Quantity'`
    :type qty_label: str, optional

    :param proportion_label: Sets the label of the percentage that will appear at the right side of the plot. |default| :code:`'Percentage'`
    :type proportion_label: str, optional

    :param proportion_format: Formats the population percentage with exactly n digits following the decimal point. The default value shows 2 digits after the decimal point. |default| :code:`.2f`
    :type proportion_format: str, optional

    :param show_population_func: If True, shows a dashed line describing the aggregation function chosen for the entire population. |default| :code:`False`
    :type show_population_func: bool, optional

    :param population_func_legend: A text that will appear in the legend describing the dashed line of the entire population. |default| :code:`'Median population value'`
    :type population_func_legend: str, optional

    :param population_format: Formats the number resulted of the aggregation function for the entire population that will appear near the dashed line with exactly n digits following the decimal point. The default value shows 0 digits after the decimal point. |default| :code:`'.0f'`
    :type population_format: str, optional

    :param population_legend: Text to describe the circles representing the population percentage. |default| :code:`Population %`
    :type population_legend: str, optional

    :param up_label: Text to describe the up bars. It will only be showed if split_variable is True. |default| :code:`Positive values`
    :type up_label: str, optional

    :param down_label: Text to describe the down bars. It will only be showed if split_variable is True. |default| :code:`Negative values`
    :type down_label: str, optional

    :param plot_title: Text to describe the plot's title. |default| :code:`None`
    :type plot_title: str, optional

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
    data = df.copy()

    if show_na:
        data[x] = data[x].fillna(na_label)
    else:
        data = data.dropna(subset=[x])

    data[x] = data[x].astype('str')
    data = data.sort_values(by=x)
    
    if ax:
        axes = ax
    else:
        fig, axes = plt.subplots(figsize=figsize, dpi=120)

    if colors is None:
        colors = get_palette()

    if color_palette:
        colors = sns.color_palette(color_palette, 3)

    if colors is not None:
        if split_variable and len(colors) < 3:
            raise KeyError(f'Expected 3 colors but only {len(colors)} was/were passed.')
        elif len(colors) < 2:
            raise KeyError(f'Expected 2 colors but only {len(colors)} was/were passed.')

    categories_names = ['df_up', 'population', 'df_down']

    colormap = dict(zip(categories_names, colors))

    _plot_bars(data, x, y, func, split_variable, axes, colormap, up_label, down_label, x_label, show_error_bar)

    _set_ticks_and_annotation(data, x, y, func, axes, circle_diameter, colormap, proportion_format, show_qty, qty_label,
                              proportion_label, show_population_func, population_func_legend, population_format)

    _set_titles_and_labels(axes, colormap, plot_title, population_legend, x, y, y_label, x_label, **legend_kwargs)
    
    if return_fig:
        plt.show()
        plt.close()

        return fig


def _plot_bars(data, x, y, func, split_variable, ax, colormap, up_label, down_label, x_label, show_error_bar):
    if show_error_bar:
        error_bar = 95
    else:
        error_bar = None

    if split_variable:
        df_up = data[(data[y] >= 0)]
        df_down = data[(data[y] < 0)]

        if len(df_up) == 0:
            raise ValueError(f'No positive values found in "{y}" column!')

        if len(df_down) == 0:
            raise ValueError(f'No negative values found in "{y}" column!')

        sns.barplot(x=x, y=y, data=df_up, ax=ax, estimator=func, ci=error_bar, color=colormap['df_up'], label=up_label)
        sns.barplot(x=x, y=y, data=df_down, ax=ax, estimator=func, ci=error_bar, color=colormap['df_down'], label=down_label)
    else:
        sns.barplot(x=x, y=y, data=data, ax=ax, estimator=func, ci=error_bar, color=colormap['df_up'], label=x_label)


def _set_ticks_and_annotation(data, x, y, func, ax, circle_diameter, colormap, proportion_format, show_qty, qty_label,
                              proportion_label, show_population_func, population_func_legend, population_format):
    ticks = plt.yticks()[0]
    tick_interval = ticks[1] - ticks[0]

    point_y_position = ticks[0] - 2 * tick_interval
    text_y_position = ticks[0] - 3.5 * tick_interval
    bottom_y_lim = ticks[0] - 5 * tick_interval

    cat_intervals = data[x].value_counts(normalize=True).sort_index().to_dict()
    cat_int = data[x].value_counts(normalize=False).sort_index().to_dict()

    for i, key in enumerate(cat_intervals):
        ax.plot(i, point_y_position, 'o', markersize=circle_diameter * cat_intervals[key], color=colormap['population'])
        ax.annotate(f"{cat_intervals[key] * 100:{proportion_format}}%", (i, text_y_position), va="top", ha="center")
        if show_qty:
            ax.annotate(f"({cat_int[key]})", (i, text_y_position * 1.13), va="top", ha="center")

    ax.annotate(proportion_label, xy=(0, text_y_position), xycoords=ax.get_yaxis_transform(),
                xytext=(-5, 0), textcoords="offset points", ha="right", va="center")

    if show_qty:
        ax.annotate(qty_label, xy=(0, text_y_position * 1.15), xycoords=ax.get_yaxis_transform(),
                    xytext=(-5, 0), textcoords="offset points", ha="right", va="center")

    if show_population_func:
        population_value = data[y].agg(func)

        plt.axhline(y=population_value, color='grey', linestyle='--', label=population_func_legend)

        ax.text(0.035, population_value + abs(tick_interval / 3),  f"{population_value:{population_format}}", color="grey",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center")

    set_ticks = np.where(ticks >= ticks[0], ticks, ticks[0])
    ax.set_yticks(set_ticks)

    ax.set_ylim(bottom=bottom_y_lim)
        
def _set_titles_and_labels(ax, colormap, plot_title, population_legend, x, y, y_label, x_label, **legend_kwargs):
    ax.set_title(plot_title)

    handles, labels = ax.get_legend_handles_labels()

    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=colormap['population'], markersize=15))

    labels.append(population_legend)

    ax.legend(handles, labels, loc='upper right', **legend_kwargs)

    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(y)

    if x_label:
        ax.set_xlabel(x_label)
    else:
        ax.set_xlabel(x)
