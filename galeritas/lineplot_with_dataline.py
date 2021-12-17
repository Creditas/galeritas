import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from galeritas.utils.creditas_palette import get_palette

__all__ = ["line_plot_with_datapoint"]


def line_plot_with_datapoint(
    data,
    x_column_name,
    y_column_name,
    cumulative=True,
    normalize=True,
    quantize_x=True,
    interest_y_value=None,
    x_label=None,
    y_label=None,
    label_font_size=12,
    datapoint=None,
    datapoint_text=None,
    vertical_line=False,
    horizontal_line=False,
    figsize=(12, 7)
):
    """
    Generates line plot with datapoint.
    Params and descriptions to be added.
    """

    df = data.copy()
    df.sort_values(by=x_column_name, inplace=True)
    temp_y_column_name = y_column_name
    temp_x_column_name = x_column_name

    if cumulative:
        df = generate_cumulative_table(df, temp_y_column_name, interest_y_value)
        temp_y_column_name = f'{temp_y_column_name}_cumulative'

    if normalize is True or normalize == 'y':
        df = normalize_y_data(df, y_column_name, temp_y_column_name, interest_y_value)
        temp_y_column_name = f'{temp_y_column_name}_normalized'

    if normalize is True or normalize == 'x':
        df = normalize_x_data(df, y_column_name, temp_x_column_name, interest_y_value)
        temp_x_column_name = f'{temp_x_column_name}_normalized'

    if quantize_x:
        df = quantize_x_data(df, temp_x_column_name)
        temp_x_column_name = f'{temp_x_column_name}_quantized'

    fig, axs = plt.subplots(figsize=figsize)
    ax = sns.lineplot(data=df, x=temp_x_column_name, y=temp_y_column_name)

    if x_label:
        set_label(ax, label_font_size, label='x', label_name=x_label)

    if y_label:
        set_label(ax, label_font_size, label='y', label_name=y_label)

    if datapoint:
        plot_datapoint(ax, datapoint, datapoint_text)

    #if vertical_line: # precisa arrumar..
    #    plot_vertical_line(df, temp_x_column_name, temp_y_column_name, datapoint)

    #if horizontal_line: # precisa arrumar..
    #    plot_horizontal_line(df, temp_x_column_name, temp_y_column_name, datapoint)

    #ax.legend(loc='lower right', fontsize=15) # precisa arrumar..


def generate_cumulative_table(df, y_col, interest_y_value):
    if interest_y_value is not None:
        df['value_to_sum'] = df[y_col].apply(lambda x: 1 if x == interest_y_value else 0)
        df[f'{y_col}_cumulative'] = df['value_to_sum'].cumsum()

    else:
        df[f'{y_col}_cumulative'] = range(1, len(df)+1)

    return df


def normalize_y_data(df, y_column_name, y_col, interest_y_value):
    if interest_y_value is not None:
        df[f'{y_col}_normalized'] = df[y_col] / len(df.loc[df[y_column_name] == interest_y_value])

    else:
        df[f'{y_col}_normalized'] = df[y_col] / len(df)

    return df


def normalize_x_data(df, y_column_name, x_col, interest_y_value):
    if interest_y_value is not None:
        df[f'{x_col}_normalized'] = df[x_col] / len(df.loc[df[y_column_name] == interest_y_value])

    else:
        df[f'{x_col}_normalized'] = df[x_col] / len(df)

    return df

# Função 'quantize_x_data' ainda precisa ser revisada e o método mais apropriado selecionado.
def quantize_x_data(df, x_col):
    # df[f'{x_col}_quantized'] = pd.qcut(df[x_col], 100, labels=False) / 100
    df[f'{x_col}_quantized'] = [i / len(df) for i in range(1, len(df) + 1)]

    return df


def set_label(ax, label_font_size, label=None, label_name=None):
    if label == 'x':
        ax.set_xlabel(label_name, fontsize=label_font_size)

    elif label == 'y':
        ax.set_ylabel(label_name, fontsize=label_font_size)

    else:
        pass


def plot_datapoint(ax, datapoint, datapoint_text):
    ax.plot([datapoint[0]], [datapoint[1]], marker='o', markersize=8, color="Green")
    datapoint_text_location = (datapoint[0]*1.01, datapoint[1]*1.02)
    if datapoint_text is True:
        datapoint_text = datapoint
    if datapoint_text is not None and datapoint_text is not True:
        ax.annotate(datapoint_text, datapoint_text_location)


def plot_vertical_line(df, temp_x_column_name, temp_y_column_name, datapoint):
    datapoint_x = datapoint[0]
    datapoint_y = datapoint[1]
    closest_value_index = df[temp_x_column_name].sub(datapoint_x).abs().idxmin()
    line_y = df.loc[closest_value_index][temp_y_column_name]
    plt.axvline(x=datapoint_x, ymin=min(datapoint_y, line_y), ymax=max(datapoint_y, line_y), color='b',
                label='vline test')


def plot_horizontal_line(df, temp_x_column_name, temp_y_column_name, datapoint):
    datapoint_x = datapoint[0]
    datapoint_y = datapoint[1]
    closest_value_index = df[temp_y_column_name].sub(datapoint_y).abs().idxmin()
    line_x = df.loc[closest_value_index][temp_x_column_name]
    plt.axhline(y=datapoint_y, xmin=min(datapoint_x, line_x), xmax=max(datapoint_x, line_x), color='b',
                label='vline test')
