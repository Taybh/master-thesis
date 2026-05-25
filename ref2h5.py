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
    print(img_files)
    # print(img_files)
    print(len(img_files))

    #print(img_files[index])

    # load and normalize data
    img = np.asanyarray(nib.load(orig_data_dir + os.sep + img_files[index]).dataobj)
    img = img - np.min(img)
    img = img / np.max(img)
    name = img_files[index].split('_')[-1].split('.')[0]

    return name, img

def load_gt_ref(index):

    ref_left = []
    ref_right = []
    for r, d, f in os.walk(gt_ref_dir):
        for x in f:
            if 'left' in x and 'REF' in x:
                ref_left.append(x)
            elif 'right' in x and 'REF' in x:
                ref_right.append(x)
    ref_left = sorted(ref_left)
    ref_right = sorted(ref_right)
    print(ref_left)
    print(len(ref_left))
    print(ref_right)
    print(len(ref_right))

    assert len(ref_left) == len(ref_right)

    ref_gt_left = np.asanyarray(nib.load(gt_ref_dir + os.sep + ref_left[index]).dataobj)
    ref_gt_right = np.asanyarray(nib.load(gt_ref_dir + os.sep + ref_right[index]).dataobj)
    ref_gt_left[ref_gt_left > 0.5] = 1
    ref_gt_right[ref_gt_right > 0.5] = 1

    print((gt_ref_dir + os.sep + ref_left[index], gt_ref_dir + os.sep + ref_right[index]))


    #ref_masks_left = nib.load(gt_ref_dir + os.sep + r1_left[index]).get_fdata()
    #ref_masks_right = nib.load(gt_ref_dir + os.sep + r1_right[index]).get_fdata()

    #ref_masks = np.load(gt_ref_dir + os.sep + 'LC_{:02d}_REF_left.nii.gz'.format(index))['ref']
    return ref_gt_left, ref_gt_right

'''
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
'''

if __name__ == "__main__":
    num_of_samples = 82
    orig_data_dir = 'data/lc/BETTS82'
    target_dir = '.'

    gt_ref_dir = '/home/tayebeh/PycharmProjects/LC/data/lc/BETTS82'

    hf = h5.File(target_dir + os.sep + 'ref.h5', 'w')
    hf.create_group('img')
    hf.create_group('ref')

    for i in range(num_of_samples):
        # # TODO: USE corrected masks from Matt!
        # if i == 40:
        #     continue

        name, img = load_original_data(i)
        ref_left, ref_right = load_gt_ref(i)


        hf['img'].create_dataset(name, data=img, dtype='float32')
        hf['ref'].create_dataset(name, data=(ref_left, ref_right), dtype='uint8')
        #hf['ref'][name].attrs['bbox'] = get_bbox(r1_left, r1_right)

    hf.close()
