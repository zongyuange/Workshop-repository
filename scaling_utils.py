# Utility functions for getting calibration results
import numpy as np
import pickle

def softmax(x):
    """
    Compute softmax values for each set of scores in x.

    Args:
        x: numpy array, array containing m samples with n-dimentions, with shape (m, n)

    Return:
        x_softmax: numpy array, softmaxed values for initial (m, n) array

    """
    # Task 1-1: write softmax function
    ##################################

    ##################################


def unpickle_logits(logits_file, verbose=0):
    with open(logits_file, "rb") as f:
        (y_logits_val, y_val), (y_logits_test, y_test) = pickle.load(f, encoding="bytes")
        
    if verbose:
        print("y_logits_val is in shape:", y_logits_val.shape)
        print("y_val is in shape:", y_val.shape)
        print("y_logits_test is in shape:", y_logits_test.shape)
        print("y_test is in shape:", y_test.shape)
        
    return ((y_logits_val, y_val), (y_logits_test, y_test))


def get_preds_confs(y_probs):
    """
    Get the predictions and confidence from input probability array, with shape (num_samples, num_classes)
    """
    y_preds = np.argmax(y_probs, axis=1)
    #y_confs = np.take_along_axis(y_probs, np.expand_dims(y_preds, 1), axis=1)
    y_confs = np.max(y_probs, axis=1)

    return y_preds, y_confs


def compute_bin_info(conf_thresh_lower, conf_thresh_upper, confs, preds, labels):
    """
    Computs accuracy and average confidence for a bin defined by conf_thresh_lower and conf_thresh_upper

    Args:
        conf_thresh_lower: float, lower threshold of confidence interval
        conf_thresh_upper: float, upper threshold of confidence interval
        confs: list of confidences
        preds: list of predictions
        labels: list of labels

    Return:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin
    """
    filtered_tuples = [x for x in zip(preds, labels, confs) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        # How many correct labels
        correct = len([x for x in filtered_tuples if x[0] == x[1]]) 
        # How many elements fall into a given bin
        len_bin = len(filtered_tuples) 
        # Average confidence of a given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin
        # Accuracy of a given bin
        acc = float(correct) / len_bin 

        return acc, avg_conf, len_bin
    

def ECE(confs, preds, labels, bin_size=0.1):
    """
    Expected Calibration Error

    Args:
        confs: list of confidences
        preds: list of predictions
        labels: list of labels
        bin_size: float, size of one bin

    Return:
        ece: expected calibration error
    """
    upper_bounds = np.arange(bin_size, bin_size + 1, bin_size)

    n = len(confs)
    ece = 0
    
    # Task 3: 
    # Go through bounds to find accuracies and confidences
    #######################################


    #######################################

    return ece


def MCE(confs, preds, labels, bin_size=0.1):
    """
    Maximum Calibration Error

    Args:
        confs: list of confidences
        preds: list of predictions
        labels: list of labels
        bin_size: float, size of one bin

    Return:
        mce: maximum calibration error
    """
    upper_bounds = np.arange(bin_size, bin_size + 1, bin_size)

    calibration_errors = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_bin_info(conf_thresh - bin_size, conf_thresh, confs, preds, labels)
        calibration_errors.append(np.abs(acc - avg_conf))

    return max(calibration_errors)


def get_bins_info(confs, preds, labels, bin_size=0.1):
    """
    Get accuracy, confidence and elements in bin for all the bins.

    Args:
        confs: list of confidences
        preds: list of predictions
        labels: list of labels
        bin_size: float, size of one bin

    Return:
        (accuracies, confidences, bin_lengths): tuple containing all the necessary info for reliability diagrams
    """
    upper_bounds = np.arange(bin_size, bin_size + 1, bin_size)

    accuracies = []
    confidences = []
    len_bins = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_bin_info(conf_thresh - bin_size, conf_thresh, confs, preds, labels)
        accuracies.append(acc)
        confidences.append(avg_conf)
        len_bins.append(len_bin)

    return accuracies, confidences, len_bins

    



