import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp

__all__ = ["plot_ks_classification"]

def plot_ks_classification(df,
            y_pred,
            y_true,
            min_max_scale=None,
            show_p_value=True,
            pos_value=1,
            neg_value=0,
            pos_label='1',
            neg_label='0',
            pos_color="#3377bb",
            neg_color="#b33d3d",
            figsize=(12, 7),
            plot_title="Kolmogorov–Smirnov (KS) Metric",
            x_label="Predicted Probability",
            ax=None,
            return_fig=False
            ):
    """
    Produces a KS plot for predicted values (or scores) vs true value (0/1)

    :param df: a pd.Dataframe that contains y_pred and y_true columns
    :type df: pd.Dataframe

    :param y_pred: column name in df corresponding to predictions
    :type y_pred: float

    :param y_true: column name in df corresponding to target values (0 or 1)
    :type y_true: integer

    :param min_max_scale: Tuple containing (min, max) values for scaling y_pred |default| :code:`None`
    :type min_max_scale: tuple, optional

    :param show_p_value: If True plot p-value for the KS together with curves |default| :code:`True`
    :type show_p_value: bool, optional

    :param pos_value: Integer 0/1 indicating which is the positive value in the y_true (in some applications 0 may indicate a 'bad behavior', like default) |default| :code:`1`
    :type pos_value: integer, optional

    :param neg_value: Integer 0/1 indicating which is the negative value in the y_true (in some applications 0 may indicate a 'bad behavior', like default) |default| :code:`0`
    :type pos_value: integer, optional

    :param pos_label: personalized label for positive value |default| :code:`1`
    :type pos_label: str, optional

    :param neg_label: personalized label for negative value |default| :code:`0`
    :type neg_label: str, optional

    :param pos_color: personalized color for positive value |default| :code:`#3377bb`
    :type pos_color: str, optional

    :param neg_color: personalized color for negative value |default| :code:`#b33d3d`
    :type neg_color: str, optional

    :param figsize: tuple containing (height, width) for plot size |default| :code:`(12, 7)`
    :type figsize: tuple, optional

    :param plot_title: main title of plot |default| :code:`Kolmogorov-Smirnov (KS) Metric`
    :type plot_title: str, optional

    :param x_label: personalized x_label |default| :code:`Predicted Probability`
    :type x_label: str, optional

    :param ax: Custom figure axes to plot. |default| :code: `None`
    :type ax: matplotlib.axes, optional

    :param return_fig: If True return figure object. |default| :code:`True`
    :type return_fig: bool, optional

    :return: Returns the figure object with the plot (*return_fig parameter needs to be set)
    :rtype: Figure

    """
    y_pred = df[y_pred]
    y_true = df[y_true]

    y_pred_outside_range = (max(y_pred) > 1 or min(y_pred) < 0)
    if y_pred_outside_range and min_max_scale is None:
        raise ValueError(f'y_pred outside (0,1) range, min_max_scale should be passed')

    # test if y_true contains only 0,1
    if len(y_true.unique()) > 2:
        raise ValueError(f'y_true has {len(y_true.unique())} unique values, it should be an [0, 1] array')

    y_true_is_not_0_and_1_only = (np.sort(y_true.unique()) != np.array([0, 1])).any()
    if y_true_is_not_0_and_1_only:
        raise ValueError(f'y_true has values different than 0 or 1, it should be an [0, 1] array')

    # scale y_pred if is not in range (0, 1)
    if min_max_scale:
        if (min(y_pred) > 1) or (max(y_pred) > 1):
            y_pred = (y_pred- min_max_scale[0])/(min_max_scale[1] - min_max_scale[0])

    pos_data = y_pred[y_true == pos_value]
    neg_data = y_pred[y_true == neg_value]

    # Compute KS
    ks_res = ks_2samp(pos_data, neg_data)
    p_value = round(ks_res.pvalue, 7)

    # Define curve
    bins = 1000
    th = np.linspace(0, 1, bins)
    pos = np.array([np.mean(pos_data <= t) for t in th])
    neg = np.array([np.mean(neg_data <= t) for t in th])
    xmax = abs(neg - pos).argmax()
    ks_text = round(100. * (neg - pos)[xmax], 2)

    # Plot
    if ax:
        axes = ax
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)

    axes.plot(th, pos, pos_color, label=pos_label)
    axes.plot(th, neg, neg_color, label=neg_label)
    axes.plot((th[xmax], th[xmax]), (pos[xmax], neg[xmax]), "ks--")
    axes.legend(loc="upper left")
    axes.set_xlabel(x_label, fontsize=10)
    if min_max_scale:
        xticks = plt.xticks()[0]
        xticks = (xticks * (min_max_scale[1] - min_max_scale[0])) + min_max_scale[0]
        axes.set_xticklabels(["{:0.2f}".format(x) for x in xticks])

    axes.set_title(plot_title, fontsize=12)
    axes.text(0.5, 0.1, f"KS={ks_text}%", fontsize=16)
    if show_p_value:
        axes.text(0.5, 0.03, f"p-value={p_value}", fontsize=12)
    axes.set_ylabel('Cumulative Probability', fontsize=10)
    
    if return_fig:
        plt.show()
        plt.close()

        return fig
    