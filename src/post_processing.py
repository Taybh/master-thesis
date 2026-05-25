import numpy as np

from scipy.ndimage.measurements import label


def do_nothing(pred_left, pred_right):
    return pred_left, pred_right


def largest_concomp(pred_left, pred_right):
    labeled, ncomponents = label(pred_left)
    lc_masks_left = np.zeros(pred_left.shape)
    maximum = 0
    maximum_label = 0

    # process left mask
    for l in range(1, ncomponents + 1):
        tmp = np.zeros(lc_masks_left.shape)
        tmp[np.logical_and(labeled > l - 0.5, labeled < l + 0.5)] = 1
        if np.sum(tmp) > maximum:
            maximum_label = l
            maximum = np.sum(tmp)
    lc_masks_left[np.logical_and(labeled > maximum_label - 0.5, labeled < maximum_label + 0.5)] = 1

    # process right mask
    labeled, ncomponents = label(pred_right)
    lc_masks_right = np.zeros(pred_left.shape)
    maximum = 0
    for l in range(1, ncomponents + 1):
        tmp = np.zeros(lc_masks_right.shape)
        tmp[np.logical_and(labeled > l - 0.5, labeled < l + 0.5)] = 1
        if np.sum(tmp) > maximum:
            maximum_label = l
            maximum = np.sum(tmp)
    lc_masks_right[np.logical_and(labeled > maximum_label - 0.5, labeled < maximum_label + 0.5)] = 1

    return lc_masks_left, lc_masks_right


def size_threshold(pred_left, pred_right, threshold=50):
    labeled, ncomponents = label(pred_left)
    lc_masks_left = np.zeros(pred_left.shape)
    for l in range(1, ncomponents + 1):
        tmp = np.zeros(lc_masks_left.shape)
        tmp[np.logical_and(labeled > l - 0.5, labeled < l + 0.5)] = 1
        if np.sum(tmp) > threshold:
            lc_masks_left[np.logical_and(tmp > 1 - 0.5, tmp < 1 + 0.5)] = 1
    labeled, ncomponents = label(pred_right)
    lc_masks_right = np.zeros(pred_right.shape)
    for l in range(1, ncomponents + 1):
        tmp = np.zeros(lc_masks_right.shape)
        tmp[np.logical_and(labeled > l - 0.5, labeled < l + 0.5)] = 1
        if np.sum(tmp) > threshold:
            lc_masks_right[np.logical_and(tmp > 1 - 0.5, tmp < 1 + 0.5)] = 1
    return lc_masks_left, lc_masks_right


def boundary_crop(pred_left, pred_right, cropping=(200, 150, 50)):
    x_crop = cropping[0]
    y_crop = cropping[1]
    z_crop = cropping[2]
    lc_masks_left = pred_left[x_crop:pred_left.shape[0] - x_crop,
                    y_crop:pred_left.shape[1] - y_crop, z_crop:pred_left.shape[2] - z_crop]
    lc_masks_right = pred_right[x_crop:pred_right.shape[0] - x_crop,
                     y_crop:pred_right.shape[1] - y_crop, z_crop:pred_right.shape[2] - z_crop]
    lc_masks_left = np.pad(lc_masks_left, ((x_crop, x_crop), (y_crop, y_crop), (z_crop, z_crop)), mode='constant')
    lc_masks_right = np.pad(lc_masks_right, ((x_crop, x_crop), (y_crop, y_crop), (z_crop, z_crop)), mode='constant')
    return lc_masks_left, lc_masks_right


def boundary_plus_thresh(pred_left, pred_right, cropping=(200, 150, 50), threshold=50):
    pred_left, pred_right = boundary_crop(pred_left, pred_right, cropping)
    return size_threshold(pred_left, pred_right, threshold)


def apply_mask(pred_left, pred_right, mask):
    raise NotImplementedError
