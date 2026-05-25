import os
import numpy as np
import nibabel as nib

from scipy.ndimage.measurements import label


def load_gt_lc(index):
    masks = np.load(gt_lc_dir + os.sep + 'lc_data_{:02d}.npz'.format(index))['gt']
    return masks[0], masks[1]


def load_gt_lc_r2(index):
    r2_left = []
    r2_right = []
    for r, d, f in os.walk(gt_lc_dir):
        for x in f:
            if 'left' in x:
                r2_left.append(x)
            elif 'right' in x:
                r2_right.append(x)
    r2_left = sorted(r2_left)
    r2_right = sorted(r2_right)
    #print(r2_left)
    #print(len(r2_left))
    #print(r2_right)
    #print(len(r2_right))

    lc_gt_left = nib.load(gt_lc_dir + os.sep + r2_left[index]).get_data()
    lc_gt_right = nib.load(gt_lc_dir + os.sep + r2_right[index]).get_data()
    lc_gt_left[lc_gt_left > 0.5] = 1
    lc_gt_right[lc_gt_right > 0.5] = 1

    return lc_gt_left, lc_gt_right


if __name__ == '__main__':
    gt_lc_dir = '/home/max/git/LCSN-Seg/data/lc_processed'
    gt_lc = load_gt_lc
    #gt_lc_dir = '/home/max/git/LCSN-Seg/data/lc/BETTS82_Rater2'
    #gt_lc = load_gt_lc_r2
    num_of_samples = 82

    # 3d-diagonal-neighbourhood
    str_3D = np.array([[[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]],
                 [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]],
                 [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]], dtype='uint8')
    print('R1')
    for i in range(num_of_samples):
        print(i)
        lc_gt_masks = gt_lc(i)

        for m in lc_gt_masks:
            labeled, ncomponents = label(m, structure=str_3D)
            if ncomponents is not 1:
                print('FAULT IN: ' + str(i))