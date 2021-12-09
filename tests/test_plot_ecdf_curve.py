import pytest
import pandas as pd
from galeritas import plot_ecdf_curve
from matplotlib import pyplot as plt

@pytest.fixture(scope='module')
def load_data():
    data = pd.read_csv("tests/data/titanic.csv")

    return data


@pytest.mark.mpl_image_compare
def test_should_generate_plot_ecdf_curve_correctly(load_data):
    df = load_data

    return plot_ecdf_curve(
        df=df,
        column_to_plot='predict_proba',
        hue='pclass',
        plot_title='Prediction distribution by PClass',
        figsize=(16, 6),
        percentiles=(25, 50, 75, 90),
        mark_percentiles=True,
        return_fig=True
    )


def test_should_return_figure_with_axes_ecdf(load_data):
    df = load_data

    fig = plot_ecdf_curve(
        df=df,
        column_to_plot='predict_proba',
        hue='pclass',
        plot_title='Prediction distribution by PClass',
        figsize=(16, 6),
        percentiles=(25, 50, 75, 90),
        mark_percentiles=True,
        return_fig=True
    )

    assert fig.get_axes() is not None


def test_should_raise_warning_about_missing_values(load_data):
    df = load_data

    with pytest.warns(UserWarning):
        plot_ecdf_curve(
            df=df,
            column_to_plot='age',
            return_fig=True
        )


def test_should_raise_exception_when_colors_is_smaller_number_categories(load_data):
    df = load_data

    with pytest.raises(KeyError):
        plot_ecdf_curve(
            df=df,
            column_to_plot='fare',
            hue='pclass',
            colors=['blue'],
            return_fig=True
        )


def test_should_return_none_object_if_return_fig_param_is_not_configured(load_data):
    df = load_data

    fig = plot_ecdf_curve(
        df=df,
        column_to_plot='predict_proba',
        hue='pclass',
        plot_title='Prediction distribution by PClass',
        figsize=(16, 6),
        percentiles=(25, 50, 75, 90),
        mark_percentiles=True
    )

    assert fig is None

@pytest.mark.mpl_image_compare
def test_should_generate_subplot_plot_ecdf_curve_correctly(load_data):
    df = load_data
    
    f, axes = plt.subplots(1,2)

    plot_ecdf_curve(
        df=df,
        column_to_plot='predict_proba',
        hue='pclass',
        plot_title='Prediction distribution by PClass',
        figsize=(16, 6),
        percentiles=(25, 50, 75, 90),
        mark_percentiles=True,
        ax=axes[1]
    )

    return f