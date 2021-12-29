import pytest
import pandas as pd

from galeritas import plot_ks_classification
from matplotlib import pyplot as plt

@pytest.fixture(scope='module')
def load_data():
    data = pd.read_csv("tests/data/titanic.csv")
    data['y_pred'] = data['predict_proba']
    data['y_true'] = data['survived']
    
    return data


@pytest.mark.mpl_image_compare
def test_should_generate_plot_ks_classification_correctly(load_data):
    df = load_data.copy()

    return plot_ks_classification(df=df,
                                  y_pred='y_pred',
                                  y_true='y_true',
                                  return_fig=True)
                                  
@pytest.mark.mpl_image_compare
def test_should_generate_plot_ks_classification_with_scaling_correctly(load_data):
    df = load_data.copy()

    return plot_ks_classification(df=df,
                           y_pred='y_pred',
                           y_true='y_true',
                           min_max_scale=(0.8,1), 
                           return_fig=True)

@pytest.mark.mpl_image_compare
def test_should_generate_subplot_plot_ks_classification_correctly(load_data):
    df = load_data.copy()

    f, axes = plt.subplots(1,2)

    plot_ks_classification(df=df,
                           y_pred='y_pred',
                           y_true='y_true',
                           ax=axes[1])

    return f

def test_should_return_none_object_if_return_fig_param_is_not_configured(load_data):
    df = load_data.copy()

    fig = plot_ks_classification(df=df,
                                 y_pred='y_pred',
                                 y_true='y_true'
                                 )

    assert fig is None


def test_should_raise_exception_when_min_max_scale_should_be_passed(load_data):
    df = load_data.copy()
    df['y_pred_score'] = df['y_pred'] * 1000
    with pytest.raises(ValueError):
        plot_ks_classification(df=df,
                               y_pred='y_pred_score',
                               y_true='y_true',
                               return_fig=True)


def test_should_raise_exception_when_y_true_has_more_than_2_unique_values(load_data):
    df = load_data.copy()
    df.loc[0, 'y_true'] = 2
    with pytest.raises(ValueError):
        plot_ks_classification(df=df,
                               y_pred='y_pred',
                               y_true='y_true',
                               return_fig=True)


def test_should_raise_exception_when_y_true_outside_range(load_data):
    df = load_data.copy()
    df.loc[df.y_true == 1, 'y_true'] = 0.2
    with pytest.raises(ValueError):
        plot_ks_classification(df=df,
                               y_pred='y_pred',
                               y_true='y_true',
                               return_fig=True)


    
