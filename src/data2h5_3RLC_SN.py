import os
import numpy as np
import nibabel as nib
import h5py as h5


def load_original_data(index):
    img_files = []
    for r, d, f in os.walk(orig_data_dir):
        for x in f:
            if 'left' not in x and 'right' not in x\
                    and not x.startswith('.')\
                    and not x.endswith('txt'):
                img_files.append(x)
    img_files = sorted(img_files)
    # print(img_files)
    print(len(img_files))

    print(img_files[index])

    # load and normalize data
    img = np.asanyarray(nib.load(orig_data_dir + os.sep + img_files[index]).dataobj)
    img = img - np.min(img)
    img = img / np.max(img)
    name = img_files[index].split('_')[-1].split('.')[0]

    return name, img


def load_gt_lc_r1(index):
    r1_left = []
    r1_right = []
    for r, d, f in os.walk(gt_r1_dir):
        for x in f:
            if 'left_mod' in x:
                r1_left.append(x)
            elif 'right_mod' in x:
                r1_right.append(x)
    r1_left = sorted(r1_left)
    r1_right = sorted(r1_right)

    # print(len(r1_left))
    # print(len(r1_right))
    assert len(r1_left) == len(r1_right)

    lc_gt_left = np.asanyarray(nib.load(gt_r1_dir + os.sep + r1_left[index]).dataobj)
    lc_gt_right = np.asanyarray(nib.load(gt_r1_dir + os.sep + r1_right[index]).dataobj)
    lc_gt_left[lc_gt_left > 0.5] = 1
    lc_gt_right[lc_gt_right > 0.5] = 1

    print((gt_r1_dir + os.sep + r1_left[index], gt_r1_dir + os.sep + r1_right[index]))

    return lc_gt_left, lc_gt_right


def load_gt_lc_r2(index):
    r2_left = []
    r2_right = []
    for r, d, f in os.walk(gt_r2_dir):
        for x in f:
            if 'left' in x:
                r2_left.append(x)
            elif 'right' in x:
                r2_right.append(x)
    r2_left = sorted(r2_left)
    r2_right = sorted(r2_right)
    # print(r2_left)
    # print(len(r2_left))
    # print(r2_right)
    # print(len(r2_right))
    assert len(r2_left) == len(r2_right)

    lc_gt_left = np.asanyarray(nib.load(gt_r2_dir + os.sep + r2_left[index]).dataobj)
    lc_gt_right = np.asanyarray(nib.load(gt_r2_dir + os.sep + r2_right[index]).dataobj)
    lc_gt_left[lc_gt_left > 0.5] = 1
    lc_gt_right[lc_gt_right > 0.5] = 1

    print((gt_r2_dir + os.sep + r2_left[index], gt_r2_dir + os.sep + r2_right[index]))

    return lc_gt_left, lc_gt_right


def load_gt_lc_r3(index):
    r3_left = []
    r3_right = []
    for r, d, f in os.walk(gt_r3_dir):
        for x in f:
            if 'LEFT' in x:
                r3_left.append(x)
            elif 'RIGHT' in x:
                r3_right.append(x)
    r3_left = sorted(r3_left)
    r3_right = sorted(r3_right)
    # print(r3_left)
    # print(len(r3_left))
    # print(r3_right)
    # print(len(r3_right))
    assert len(r3_left) == len(r3_right)

    lc_gt_left = np.asanyarray(nib.load(gt_r3_dir + os.sep + r3_left[index]).dataobj)
    lc_gt_right = np.asanyarray(nib.load(gt_r3_dir + os.sep + r3_right[index]).dataobj)
    lc_gt_left[lc_gt_left > 0.5] = 1
    lc_gt_right[lc_gt_right > 0.5] = 1

    print((gt_r3_dir + os.sep + r3_left[index], gt_r3_dir + os.sep + r3_right[index]))

    return lc_gt_left, lc_gt_right

def load_gt_SN(index):
    SN_left = []
    SN_right = []
    for r, d, f in os.walk(gt_SN_dir):
        for x in f:
            if 'LEFT' in x:
                SN_left.append(x)
            elif 'RIGHT' in x:
                SN_right.append(x)
    SN_left = sorted(SN_left)
    SN_right = sorted(SN_right)
    print(SN_left)
    print(len(SN_left))
    print(SN_right)
    print(len(SN_right))

    assert len(SN_left) == len(SN_right)

    SN_gt_left = np.asanyarray(nib.load(gt_SN_dir + os.sep + SN_left[index]).dataobj)
    SN_gt_right = np.asanyarray(nib.load(gt_SN_dir + os.sep + SN_right[index]).dataobj)
    SN_gt_left[SN_gt_left > 0.5] = 1
    SN_gt_right[SN_gt_right > 0.5] = 1

    print((gt_SN_dir + os.sep + SN_left[index], gt_SN_dir + os.sep + SN_right[index]))

    return SN_gt_left, SN_gt_right



def get_bbox(left_mask, right_mask):
    whole = left_mask + right_mask
    segments = np.where(whole > 0.5)
    bbox = (np.min(segments[0]),
            np.min(segments[1]),
            np.min(segments[2]),
            np.max(segments[0]),
            np.max(segments[1]),
            np.max(segments[2]))
    return bbox


if __name__ == "__main__":
    num_of_samples = 2
    orig_data_dir = 'data/lc/BETTS82'
    gt_r1_dir = 'data/lc/BETTS82'
    gt_r2_dir = 'data/lc/BETTS82_Rater2'
    gt_r3_dir = 'data/lc/BETTS82_Rater3'
    target_dir = 'data/'
    gt_SN_dir = 'data/sn/BETTS82'


    hf = h5.File(target_dir + os.sep + 'new_SN-LC-3R-data.h5', 'w')
    hf.create_group('img')
    hf.create_group('R1')
    hf.create_group('R2')
    hf.create_group('R3')
    hf.create_group('SN')
    for i in range(num_of_samples):
        # # TODO: USE corrected masks from Matt!
        # if i == 40:
        #     continue

        name, img = load_original_data(i)
        r1_left, r1_right = load_gt_lc_r1(i)
        r2_left, r2_right = load_gt_lc_r2(i)
        r3_left, r3_right = load_gt_lc_r3(i)
        SN_left, SN_right = load_gt_SN(i)

        hf['img'].create_dataset(name, data=img, dtype='float32')
        hf['R1'].create_dataset(name, data=(r1_left, r1_right), dtype='uint8')
        hf['R1'][name].attrs['bbox'] = get_bbox(r1_left, r1_right)
        hf['R2'].create_dataset(name, data=(r2_left, r2_right), dtype='uint8')
        hf['R2'][name].attrs['bbox'] = get_bbox(r2_left, r2_right)
        hf['R3'].create_dataset(name, data=(r3_left, r3_right), dtype='uint8')
        hf['R3'][name].attrs['bbox'] = get_bbox(r3_left, r3_right)
        hf['SN'].create_dataset(name, data=(SN_left, SN_right), dtype='uint8')
        hf['SN'][name].attrs['bbox'] = get_bbox(SN_left, SN_right)

    hf.close()
