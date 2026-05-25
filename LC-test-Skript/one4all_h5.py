import os
import numpy as np
import nibabel as nib
import rpy2.robjects as robjects
import h5py
from post_processing import boundary_plus_thresh


def load_pred_masks(index):
    lc_pred_left = nib.load(test_lc_dir + os.sep + '{:02d}'.format(index) + '_0_' + model_name + '.nii.gz').get_fdata()
    lc_pred_right = nib.load(test_lc_dir + os.sep + '{:02d}'.format(index) + '_1_' + model_name +'.nii.gz').get_fdata()
    return lc_pred_left, lc_pred_right


def load_original_data(index):
    img_files = []
    for r, d, f in os.walk(orig_data_dir):
        for x in f:
            if 'left' not in x and 'right' not in x and 'REF' not in x and 'MB' not in x \
                    and not x.startswith('.') \
                    and not x.endswith('txt'):
                img_files.append(x)

    #print('img index: '+list(hf['img'])[index])
    img_files = sorted(img_files)
    #return hf['img'][index]
    return nib.load(orig_data_dir + os.sep + img_files[index]).get_fdata()


# TODO: write methods for all different data sets: R2, AD, MCI, PD, ... or check if it works already
def load_gt_lc_r1(index):
    gt_lc_dir = '/home/tayebeh/PycharmProjects/LC/data/sn/BETTS82'
    r1_left = []
    r1_right = []
    for r, d, f in os.walk(gt_lc_dir):
        for x in f:
            if 'LEFT' in x:
                r1_left.append(x)
            elif 'RIGHT' in x:
                r1_right.append(x)

    #print('SN index: '+list(hf['SN'])[index])
    r1_left = sorted(r1_left)
    r1_right = sorted(r1_right)
    print(r1_left)
    print(len(r1_left))
    print(r1_right)
    print(len(r1_right))
    #lc_gt_left = hf['SN'][index][0]
    #lc_gt_right = hf['SN'][index][1]
    lc_gt_left = nib.load(gt_lc_dir + os.sep + r1_left[index]).get_fdata()
    lc_gt_right = nib.load(gt_lc_dir + os.sep + r1_right[index]).get_fdata()
    lc_gt_left[lc_gt_left > 0.5] = 1
    lc_gt_right[lc_gt_right > 0.5] = 1
    return lc_gt_left, lc_gt_right

'''
def load_gt_lc_r2(index):
    gt_lc_dir = '/home/max/git/LCSN-Seg/data/lc/BETTS82_Rater2'
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
    print(r2_left)
    print(len(r2_left))
    print(r2_right)
    print(len(r2_right))

    lc_gt_left = nib.load(gt_lc_dir + os.sep + r2_left[index]).get_data()
    lc_gt_right = nib.load(gt_lc_dir + os.sep + r2_right[index]).get_data()
    lc_gt_left[lc_gt_left > 0.5] = 1
    lc_gt_right[lc_gt_right > 0.5] = 1

    return lc_gt_left, lc_gt_right


def load_gt_lc_r3(index):
    gt_lc_dir = '/home/max/git/LCSN-Seg/data/lc/BETTS82_Rater3'
    r3_left = []
    r3_right = []
    for r, d, f in os.walk(gt_lc_dir):
        for x in f:
            if 'LEFT' in x:
                r3_left.append(x)
            elif 'RIGHT' in x:
                r3_right.append(x)
    r3_left = sorted(r3_left)
    r3_right = sorted(r3_right)
    print(r3_left)
    print(len(r3_left))
    print(r3_right)
    print(len(r3_right))
    assert len(r3_left) == len(r3_right)

    lc_gt_left = np.asanyarray(nib.load(gt_lc_dir + os.sep + r3_left[index]).dataobj)
    lc_gt_right = np.asanyarray(nib.load(gt_lc_dir + os.sep + r3_right[index]).dataobj)
    lc_gt_left[lc_gt_left > 0.5] = 1
    lc_gt_right[lc_gt_right > 0.5] = 1

    return lc_gt_left, lc_gt_right
'''

def load_gt_ref(index):
    ref_left = []
    ref_right = []
    for r, d, f in os.walk(gt_ref_dir):
        for x in f:
            if 'left' in x and 'REF' in x:
                ref_left.append(x)
            elif 'right' in x and 'REF' in x:
                ref_right.append(x)
    r1_left = sorted(ref_left)
    r1_right = sorted(ref_right)
    print(ref_left)
    print(len(ref_left))
    print(ref_right)
    print(len(ref_right))

    ref_masks_left = nib.load(gt_ref_dir + os.sep + ref_left[index]).get_fdata()
    ref_masks_right = nib.load(gt_ref_dir + os.sep + ref_right[index]).get_fdata()

    #ref_masks = np.load(gt_ref_dir + os.sep + 'LC_{:02d}_REF_left.nii.gz'.format(index))['ref']
    return ref_masks_left, ref_masks_right


def dsc(gt, pred):
    intersection = np.sum(pred * gt)
    net_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    return float(2 * intersection) / (net_sum + gt_sum)


def cr(data, lc_mask_left, lc_mask_right, ref_mask_left, ref_mask_right):
    lc_left = data[lc_mask_left > 0.5]
    lc_right = data[lc_mask_right > 0.5]
    ref_left = data[ref_mask_left > 0.5]
    ref_right = data[ref_mask_right > 0.5]

    # get median and max values
    med_lc_left = np.median(lc_left)
    med_lc_right = np.median(lc_right)
    med_ref_left = np.median(ref_left)
    med_ref_right = np.median(ref_right)
    max_lc_left = np.max(lc_left)
    max_lc_right = np.max(lc_right)

    # calculate ratios
    r_med_left = med_lc_left / med_ref_left
    r_med_right = med_lc_right / med_ref_right
    r_max_left = max_lc_left / med_ref_left
    r_max_right = max_lc_right / med_ref_right

    return r_med_left, r_med_right, r_max_left, r_max_right


if __name__ == '__main__':
    hf = h5py.File('/home/tayebeh/PycharmProjects/LC/data/SN-LC-3R-data.h5','r')
    test_name = 'one4all_SN_result'
    orig_data_dir = '/home/tayebeh/PycharmProjects/LC/data/lc/BETTS82'
    # gt_lc_dir = '/home/max/git/LCSN-Seg/data/lc/BETTS82'
    gt_ref_dir = '/home/tayebeh/PycharmProjects/LC/data/lc/BETTS82'
    model_name = 'SN_LC_03_2R_2ndPart_'
    test_lc_dir = '/home/tayebeh/PycharmProjects/LC/LCSN-Seg-f4f50f3/results/lc2/SN_LC_03_2R_2ndPart_best/'
    #test_ref_dir = '/home/max/git/LCSN-Seg/data/lc_refs'
    num_of_samples = 82
    conf_for_icc = 0.95
    write_pp_masks = True


    dsc_csv = 'No,DSC_left,DSC_right,DSC_all\n'
    cr_csv = 'No,CR_median_left,CR_median_right,CR_max_left,CR_max_right\n'
    cr_lists = {'median_left': [],
                'median_right': [],
                'max_left': [],
                'max_right': []}
    cr_gt_csv = 'No,CR_median_left,CR_median_right,CR_max_left,CR_max_right\n'
    cr_gt_lists = {'median_left': [],
                'median_right': [],
                'max_left': [],
                'max_right': []}
    icc_csv = 'CR_median_left,CR_median_right,CR_max_left,CR_max_right\n'

    # create results directory
    results_dir = 'EVAL-' + test_name + '_' + model_name
    if os.path.isdir(results_dir):
        count = 1
        results_dir = 'EVAL_' + str(count) + '-' + test_name + '_' + model_name
        while os.path.isdir(results_dir):
            count += 1
            results_dir = 'EVAL_' + str(count) + '-' + test_name + '_' + model_name
    os.makedirs(results_dir)
    os.makedirs(results_dir + os.sep + 'ICC_tmp')
    os.makedirs(results_dir + os.sep + 'post_processed_lc_masks')

    for i in range(num_of_samples):
        print(i)
        # # TODO: Find a better solution for R2!
        # j = i
        # if i == 40:
        #     continue
        # if i > 40:
        #     j = i-1
        data = load_original_data(i)
        lc_masks = load_pred_masks(i)
        lc_ref = load_gt_ref(i)
        lc_gt_masks = load_gt_lc_r1(i)
        #lc_gt_ref = load_gt_ref(i)

        # post-processing
        # lc_masks = boundary_plus_thresh(lc_masks[0], lc_masks[1])
        #lc_ref = boundary_plus_thresh(lc_ref[0], lc_ref[1])

        # write masks
        # if write_pp_masks:
        #     tmp_file = nib.Nifti1Image(lc_masks[0], np.eye(4))
        #     nib.save(tmp_file, results_dir + os.sep + 'post_processed_lc_masks' + os.sep + '{:02d}_0_'.format(i) + model_name + 'post_processed.nii.gz')
        #     tmp_file = nib.Nifti1Image(lc_masks[1], np.eye(4))
        #     nib.save(tmp_file, results_dir + os.sep + 'post_processed_lc_masks' + os.sep + '{:02d}_1_'.format(i) + model_name + 'post_processed.nii.gz')

        # calculate DSC
        dsc_all = dsc(lc_gt_masks[0] + lc_gt_masks[1], lc_masks[0] + lc_masks[1])
        dsc_0 = dsc(lc_gt_masks[0], lc_masks[0])
        dsc_1 = dsc(lc_gt_masks[1], lc_masks[1])
        dsc_csv += str(i) + ',' + str(dsc_0) + ',' + str(dsc_1) + ',' + str(dsc_all) + '\n'

        # calculate contrast ratios
        cr_automated = cr(data, lc_masks[0], lc_masks[1], lc_ref[0], lc_ref[1])
        cr_csv += str(i) + ',' + str(cr_automated[0]) + ',' + str(cr_automated[1]) + ',' + str(cr_automated[2]) + ',' + str(cr_automated[3]) + '\n'
        cr_lists['median_left'].append(cr_automated[0])
        cr_lists['median_right'].append(cr_automated[1])
        cr_lists['max_left'].append(cr_automated[2])
        cr_lists['max_right'].append(cr_automated[3])

        cr_manual = cr(data, lc_gt_masks[0], lc_gt_masks[1], lc_gt_ref[0], lc_gt_ref[1])
        cr_gt_csv += str(i) + ',' + str(cr_manual[0]) + ',' + str(cr_manual[1]) + ',' + str(cr_manual[2]) + ',' + str(cr_manual[3]) + '\n'
        cr_gt_lists['median_left'].append(cr_manual[0])
        cr_gt_lists['median_right'].append(cr_manual[1])
        cr_gt_lists['max_left'].append(cr_manual[2])
        cr_gt_lists['max_right'].append(cr_manual[3])

    # calculate ICC for contrast ratios
    robjects.r('library(DescTools)')
    cr_types = ['median_left', 'median_right', 'max_left', 'max_right']
    for t in cr_types:
        assert len(cr_lists[t]) == len(cr_gt_lists[t])
        tmp_file = results_dir + os.sep + 'ICC_tmp' + os.sep + t + '.tsv'
        with open(tmp_file, 'w') as f:
            f.write('gt-' + t + '\t' + model_name + t + '\n')
            for v in range(len(cr_lists[t])):
                f.write(str(cr_gt_lists[t][v]) + '\t' + str(cr_lists[t][v]) + '\n')
        robjects.r(t + ' <- read.csv("' + tmp_file + '", sep="\t")')
        iccs = str(robjects.r('ICC(' + t + ', conf.level=' + str(conf_for_icc) + ')'))
        with open(results_dir + os.sep + 'ICC_tmp' + os.sep + model_name + '_CR_ICC_R-Output_' + t, 'w') as f:
            f.write(iccs)
        for i in iter(iccs.splitlines()):
            if 'ICC3' in i and 'ICC3k' not in i:
                icc = [x for x in i.split(' ') if x != ''][2]
        icc_csv += icc + ','
    icc_csv += '\n'

    with open(results_dir + os.sep + model_name + '_DSC.csv', 'w') as f:
        f.write(dsc_csv)
    with open(results_dir + os.sep + model_name + '_CRs_automated.csv', 'w') as f:
        f.write(cr_csv)
    with open(results_dir + os.sep + model_name + '_CRs_manual.csv', 'w') as f:
        f.write(cr_gt_csv)
    with open(results_dir + os.sep + model_name + '_CRs_ICCs_conf-' + str(conf_for_icc) + '.csv', 'w') as f:
        f.write(icc_csv)
