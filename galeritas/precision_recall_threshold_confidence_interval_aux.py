import numpy as np
import bisect

import warnings

warnings.filterwarnings("ignore")

global thresholds 
thresholds = np.linspace(0, 1, 1000)


def threshold_confidence_interval(predictions_with_target, target_name, prediction_column_name, target=1, n_trials=50, sample_size_percent=0.5):
    plot_data = []

    for trial in range(n_trials):
        predictions_sample = predictions_with_target.sample(frac=sample_size_percent, random_state=42+trial)
        plot_data.append(
            calculate_precision_recall_support_by_threshold(
                predictions_sample[target_name],
                predictions_sample[prediction_column_name],
                target
            )
        )

    uniform_precision_plots = []
    uniform_recall_plots = []
    uniform_support_plots = []
    uniform_support_rate_plots = []

    for p in plot_data:
        uniform_precision = []
        uniform_recall = []
        uniform_support = []
        uniform_support_rate = []
        for ut in thresholds:
            index = bisect.bisect_left(p['thresholds'], ut)
            uniform_precision.append(p['precision'][index])
            uniform_recall.append(p['recall'][index])
            uniform_support.append(p['support'][index])
            uniform_support_rate.append(p['support_percent'][index])

        uniform_precision_plots.append(uniform_precision)
        uniform_recall_plots.append(uniform_recall)
        uniform_support_plots.append(uniform_support)
        uniform_support_rate_plots.append(uniform_support_rate)

    return uniform_precision_plots, uniform_recall_plots, uniform_support_plots,\
           uniform_support_rate_plots, thresholds


def calculate_precision_recall_support_by_threshold(y_test, y_pred_prob_test, target):
    true_probabilities = y_pred_prob_test[y_test == target]
    total_class_data = y_test[y_test == target].shape[0]

    if target == 0:
        total_data_within_threshold = np.array([np.sum(y_pred_prob_test <= t) for t in thresholds])
        true_data_within_threshold = np.array([np.sum(true_probabilities <= t) for t in thresholds])
        support = np.array([np.sum(y_pred_prob_test <= t) for t in thresholds])
        support_percent = np.array([np.mean(y_pred_prob_test <= t) for t in thresholds])
    else:
        total_data_within_threshold = np.array([np.sum(y_pred_prob_test >= t) for t in thresholds])
        true_data_within_threshold = np.array([np.sum(true_probabilities >= t) for t in thresholds])
        support = np.array([np.sum(y_pred_prob_test >= t) for t in thresholds])
        support_percent = np.array([np.mean(y_pred_prob_test >= t) for t in thresholds])

    precision = np.round(np.divide(true_data_within_threshold, total_data_within_threshold), 2)
    recall = np.round(np.divide(true_data_within_threshold, total_class_data), 2)

    return {
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'support': support,
        'support_percent': support_percent
    }


def _get_threshold_metrics_intervals(lower_precision, median_precision, upper_precision,
                                     lower_recall, median_recall, upper_recall,
                                     lower_support, median_support, upper_support,
                                     lower_support_rate, median_support_rate, upper_support_rate,
                                     threshold, uniform_thresholds):
    max_index_value = uniform_thresholds[uniform_thresholds <= threshold].max()
    ix = np.where(uniform_thresholds == max_index_value)

    lower_precision_value = round(float(lower_precision[ix]), 2)
    median_precision_value = round(float(median_precision[ix]), 2)
    upper_precision_value = round(float(upper_precision[ix]), 2)

    lower_recall_value = round(float(lower_recall[ix]), 2)
    median_recall_value = round(float(median_recall[ix]), 2)
    upper_recall_value = round(float(upper_recall[ix]), 2)

    lower_support_value = round(float(lower_support[ix]), 2)
    median_support_value = round(float(median_support[ix]), 2)
    upper_support_value = round(float(upper_support[ix]), 2)

    lower_support_rate_value = round(float(lower_support_rate[ix]), 2)
    median_support_rate_value = round(float(median_support_rate[ix]), 2)
    upper_support_rate_value = round(float(upper_support_rate[ix]), 2)

    return {
        "Median Precision": median_precision_value,
        "Precision Range": f"{lower_precision_value} - {upper_precision_value}",
        "Median Recall": median_recall_value,
        "Recall Range": f"{lower_recall_value} - {upper_recall_value}",
        "Median Support": median_support_value,
        "Support Range": f"{lower_support_value} - {upper_support_value}",
        "Median Support Rate": median_support_rate_value,
        "Support Rate Range": f"{lower_support_rate_value} - {upper_support_rate_value}"
    }
