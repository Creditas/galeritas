import pytest
import pandas as pd
from matplotlib import pyplot as plt

from galeritas import bar_plot_with_population_proportion


@pytest.fixture(scope='module')
def load_data():
    data = pd.read_csv("tests/data/zoo.csv")

    return data


@pytest.mark.mpl_image_compare
def test_should_generate_bar_plot_with_population_proportion_correctly(load_data):
    df = load_data

    return bar_plot_with_population_proportion(
        df=df,
        x='type',
        y='legs',
        return_fig=True
    )


def test_should_return_figure_with_axes_ecdf(load_data):
    df = load_data

    fig = bar_plot_with_population_proportion(
        df=df,
        x='type',
        y='legs',
        return_fig=True
    )

    assert fig.get_axes() is not None


def test_should_raise_exception_when_colors_less_than_necessary(load_data):
    df = load_data

    with pytest.raises(KeyError):
        bar_plot_with_population_proportion(
            df=df,
            x='type',
            y='legs',
            colors=['blue'],
            return_fig=True
        )


def test_should_raise_exception_when_column_without_negative_values(load_data):
    df = load_data

    with pytest.raises(ValueError):
        bar_plot_with_population_proportion(
            df=df,
            x='type',
            y='legs',
            split_variable=True,
            return_fig=True
        )


def test_should_return_none_object_if_return_fig_param_is_not_configured(load_data):
    df = load_data

    fig = bar_plot_with_population_proportion(
        df=df,
        x='type',
        y='legs'
    )

    assert fig is None

@pytest.mark.mpl_image_compare
def test_should_generate_subplot_with_bar_plot_with_population_proportion_correctly(load_data):
    df = load_data

    f, axes = plt.subplots(1,2)

    bar_plot_with_population_proportion(df=df, x='type', y='legs', ax=axes[1])

    return f