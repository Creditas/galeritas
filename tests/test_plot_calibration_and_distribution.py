import pytest
import pandas as pd

from matplotlib import pyplot as plt
from galeritas import plot_calibration_and_distribution


@pytest.fixture(scope='module')
def load_data():
    data = pd.read_csv("tests/data/titanic.csv")
    data['y_pred'] = data['predict_proba']
    data['y_true'] = data['survived']
    return data


@pytest.mark.mpl_image_compare
def test_should_generate_plot_calibration_and_distribution_correctly(load_data):
    df = load_data

    return plot_calibration_and_distribution(df, 'y_true', 'y_pred', return_fig=True)

@pytest.mark.mpl_image_compare
def test_should_generate_subplot_with_calibration_and_distribution_correctly(load_data):
    df = load_data

    f, axes = plt.subplots(1,2)

    plot_calibration_and_distribution(df, 'y_true', 'y_pred', ax=axes[0])

    return f


def test_should_return_type_error_when_a_parameter_is_missing(load_data):
    df = load_data

    with pytest.raises(TypeError):
        plot_calibration_and_distribution(df, 'y_true', return_fig=True)


def test_should_raise_error_when_strategy_is_invalid(load_data):
    df = load_data

    with pytest.raises(ValueError):
        plot_calibration_and_distribution(df, 'y_true', 'y_pred', strategy='alo', return_fig=True)

@pytest.mark.mpl_image_compare
def test_should_raise_warning_when_ax_and_show_distribution_passed(load_data):
    df = load_data
    
    f, axes = plt.subplots(1,2)

    with pytest.warns(UserWarning):
        plot_calibration_and_distribution(df, 'y_true', 'y_pred', show_distribution=True, ax=axes[0])

    return f