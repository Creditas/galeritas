import pytest
import pandas as pd

from galeritas import plot_ks_classification


@pytest.fixture(scope='module')
def load_data():
    data = pd.read_csv("tests/data/titanic.csv")
    data['y_pred'] = data['predict_proba']
    data['y_true'] = data['survived']
    return data


@pytest.mark.mpl_image_compare
def test_should_generate_plot_ks_classification_correctly(load_data):
    df = load_data

    return plot_ks_classification(y_pred=df['y_pred'],
                   y_true=df['y_true'])

def test_should_raise_exception_when_min_max_scale_should_be_passed(load_data):
    df = load_data
    df['y_pred_score'] = df['y_pred']*1000
    with pytest.raises(ValueError):
        plot_ks_classification(y_pred=df['y_pred_score'],
                               y_true=df['y_true'])

def test_should_raise_exception_when_y_true_has_more_than_2_unique_values(load_data):
    df = load_data
    df.loc[0, 'y_true'] = 2
    with pytest.raises(ValueError):
        plot_ks_classification(y_pred=df['y_pred_score'],
                               y_true=df['y_true'])

def test_should_raise_exception_when_y_true_outside_range(load_data):
    df = load_data
    df.loc[df.y_true == 1, 'y_true'] = 0.2
    with pytest.raises(ValueError):
        plot_ks_classification(y_pred=df['y_pred_score'],
                               y_true=df['y_true'])
