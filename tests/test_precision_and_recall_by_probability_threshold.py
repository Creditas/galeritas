import pytest
import pandas as pd
from galeritas.precision_and_recall_by_probability_threshold import plot_precision_and_recall_by_probability_threshold


@pytest.fixture(scope='module')
def load_data():
    data = pd.read_csv("tests/data/titanic.csv")

    return data


@pytest.mark.mpl_image_compare
def test_should_generate_plot_precision_and_recall_by_probability_threshold_correctly(load_data):
    df = load_data

    return plot_precision_and_recall_by_probability_threshold(
        df,
        prediction_column_name='predict_proba',
        target_name='survived',
        n_trials=5
    )


def test_should_return_figure_with_axes_ecdf(load_data):
    df = load_data

    fig = plot_precision_and_recall_by_probability_threshold(
        df,
        prediction_column_name='predict_proba',
        target_name='survived',
        n_trials=5
    )

    assert fig.get_axes() is not None


def test_should_raise_exception_when_colors_less_than_necessary(load_data):
    df = load_data

    with pytest.raises(KeyError):
        plot_precision_and_recall_by_probability_threshold(
            df,
            prediction_column_name='predict_proba',
            target_name='survived',
            colors=['blue']
        )


def test_should_raise_exception_when_target_is_not_binary(load_data):
    df = load_data

    with pytest.raises(ValueError):
        plot_precision_and_recall_by_probability_threshold(
            df,
            prediction_column_name='predict_proba',
            target_name='class'
        )
