import numpy as np


### FOR REGRESSION
def pearson_correlation_3d(actual_values, predicted_values):
    correlation_coefficients = []
    for dim in range(actual_values.shape[0]):
        actual_dim = actual_values[dim]
        predicted_dim = predicted_values[dim]
        actual_mean = np.mean(actual_dim)
        predicted_mean = np.mean(predicted_dim)
        cov = np.mean((actual_dim - actual_mean) * (predicted_dim - predicted_mean))
        actual_std = np.std(actual_dim)
        predicted_std = np.std(predicted_dim)
        correlation_coefficient = cov / (actual_std * predicted_std)
        correlation_coefficients.append(correlation_coefficient)
    return correlation_coefficients


def calculate_regression_metrics_3d(actual_values, predicted_values):
    mse = np.mean((actual_values - predicted_values) ** 2, axis = 1)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values - predicted_values), axis = 1)
    r = pearson_correlation_3d(actual_values, predicted_values)
    return mse, rmse, mae, r




### FOR CLASSIFICATION
def one_hot_to_labels(one_hot_matrix):
    num_samples = one_hot_matrix.shape[1]
    labels = []
    for i in range(num_samples):
        index = np.argmax(one_hot_matrix[:, i])
        labels.append(index)
    return np.array(labels).reshape(1,-1)

def labels_to_one_hot(labels):
    labels_flat = labels.flatten()
    unique_labels = sorted(set(labels_flat))
    num_samples = len(labels_flat)
    num_classes = len(unique_labels)
    one_hot_matrix = np.zeros((num_classes, num_samples))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    for i, label in enumerate(labels_flat):
        one_hot_matrix[label_to_index[label], i] = 1
    return one_hot_matrix


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = np.sum((y_true == i) & (y_pred == j))
    return cm

def f1_score_np(y_true, y_pred, num_classes):
    f1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        true_positives = np.sum((y_true == i) & (y_pred == i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        false_negatives = np.sum((y_true == i) & (y_pred != i))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_scores[i] = 2 * (precision * recall) / (precision + recall)
    return f1_scores

def get_acc(y_true, y_pred):
    y_pred = one_hot_to_labels(y_pred)
    num_correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return num_correct / total


def get_roc(y_true, y_score, num_thresh = 100):
    fpr, tpr = [], []
    thresh = np.linspace(0, 1, num_thresh)
    best_thr, best_thr_val = None, -np.inf

    for thr in thresh:
        true_pos = np.sum((y_true == 1) & (y_score >= thr))
        false_pos = np.sum((y_true == 0) & (y_score >= thr))
        true_neg = np.sum((y_true == 0) & (y_score < thr))
        false_neg = np.sum((y_true == 1) & (y_score < thr))

        curr_fpr = false_pos / (false_pos + true_neg)
        curr_tpr = true_pos / (true_pos + false_neg)

        fpr.append(curr_fpr)
        tpr.append(curr_tpr)

        if curr_tpr - curr_fpr > best_thr_val:
            best_thr = thr
            best_thr_val = curr_tpr - curr_fpr

    return fpr, tpr, best_thr

def get_auc_score(fpr, tpr):
    auc = np.trapz(tpr[::-1], fpr[::-1])
    return auc


def confusion_matrix_np(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=0)
    n_classes = y_pred_proba.shape[0]
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = np.sum(np.logical_and(y_true == i, y_pred == j))
    return cm

def f1_score_np(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=0)
    n_classes = y_pred_proba.shape[0]
    f1_scores = np.zeros(n_classes)
    for i in range(n_classes):
        true_positives = np.sum(np.logical_and(y_true == i, y_pred == i))
        false_positives = np.sum(np.logical_and(y_true != i, y_pred == i))
        false_negatives = np.sum(np.logical_and(y_true == i, y_pred != i))
        precision = true_positives / (true_positives + false_positives + 1e-15)
        recall = true_positives / (true_positives + false_negatives + 1e-15)
        f1_scores[i] = 2 * precision * recall / (precision + recall + 1e-15)
    return f1_scores

def roc_curve_np(y_true, y_pred_proba, class_pair):
    class1, class2 = class_pair
    y_true_bin = np.logical_or(y_true == class1, y_true == class2)
    y_pred_prob_bin = y_pred_proba[[class1, class2], :]
    thresholds = np.linspace(0, 1, 100)
    tpr = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)
    for i, thresh in enumerate(thresholds):
        y_pred_bin = y_pred_prob_bin[1, :] > thresh
        tp = np.sum(np.logical_and(y_true_bin, y_pred_bin))
        fp = np.sum(np.logical_and(np.logical_not(y_true_bin), y_pred_bin))
        tn = np.sum(np.logical_and(np.logical_not(y_true_bin), np.logical_not(y_pred_bin)))
        fn = np.sum(np.logical_and(y_true_bin, np.logical_not(y_pred_bin)))
        tpr[i] = tp / (tp + fn + 1e-15)
        fpr[i] = fp / (fp + tn + 1e-15)
    return fpr, tpr

def evaluate_classification_np(y_true, y_pred_proba):
    cm = confusion_matrix_np(y_true, y_pred_proba)
    f1_scores = f1_score_np(y_true, y_pred_proba)
    n_classes = y_pred_proba.shape[0]
    roc_curves = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            class_pair = (i, j)
            fpr, tpr = roc_curve_np(y_true, y_pred_proba, class_pair)
            roc_curves.append({'class_pair': class_pair, 'fpr': fpr, 'tpr': tpr})
    return cm, f1_scores, roc_curves