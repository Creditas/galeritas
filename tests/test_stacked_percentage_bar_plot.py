import pytest
import pandas as pd
from galeritas import stacked_percentage_bar_plot
from matplotlib import pyplot as plt

@pytest.fixture(scope='module')
def load_data():
    data = pd.read_csv("tests/data/zoo.csv")

    return data


@pytest.mark.mpl_image_compare
def test_should_generate_stacked_percentage_bar_plot_correctly(load_data):
    df = load_data

    return stacked_percentage_bar_plot(
        categorical_feature='legs',
        df=df,
        hue='type',
        plot_title='Zoo',
        annotate=True,
        color_palette='husl',
        bbox_to_anchor=(1.13, 1.01),
        figsize=(20, 8),
        return_fig=True
    )


def test_should_return_figure_with_axes(load_data):
    df = load_data

    fig = stacked_percentage_bar_plot(
        categorical_feature='legs',
        df=df,
        hue='type',
        plot_title='Zoo',
        annotate=True,
        color_palette='husl',
        bbox_to_anchor=(1.13, 1.01),
        figsize=(20, 8),
        return_fig=True
    )

    assert fig.get_axes() is not None


def test_should_raise_exception_when_colors_is_smaller_number_categories(load_data):
    df = load_data

    with pytest.raises(KeyError):
        stacked_percentage_bar_plot(
            categorical_feature='legs',
            df=df,
            hue='type',
            colors=['blue'],
            return_fig=True
        )


def test_should_return_none_object_if_return_fig_param_is_not_configured(load_data):
    df = load_data

    fig = stacked_percentage_bar_plot(
        categorical_feature='legs',
        df=df,
        hue='type',
        plot_title='Zoo',
        annotate=True,
        color_palette='husl',
        bbox_to_anchor=(1.13, 1.01),
        figsize=(20, 8)
    )

    assert fig is None

@pytest.mark.mpl_image_compare
def test_should_generate_subplot_stacked_percentage_bar_plot_correctly(load_data):
    df = load_data

    f, axes = plt.subplots(1,2)

    stacked_percentage_bar_plot(
        categorical_feature='legs',
        df=df,
        hue='type',
        plot_title='Zoo',
        annotate=True,
        color_palette='husl',
        bbox_to_anchor=(1.13, 1.01),
        figsize=(20, 8),
        ax=axes[1]
    )

    return f