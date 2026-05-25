import os
import numpy as np
import nibabel as nib

if __name__ == "__main__":
    r1_dir = '../data/lc/BETTS82/'
    r2_dir = '../data/lc/BETTS82_Rater2/'

    img_files = []
    for r, d, f in os.walk(r1_dir):
        for x in f:
            if 'left' not in x and 'right' not in x\
                    and not x.startswith('.')\
                    and not x.endswith('txt'):
                img_files.append(x)
    img_files = sorted(img_files)
    print(img_files)
    print(len(img_files))

    r1_left = []
    for r, d, f in os.walk(r1_dir):
        for x in f:
            if 'left' in x and 'REF' not in x \
                    and not x.startswith('.') \
                    and not x.endswith('txt'):
                r1_left.append(x)
    r1_left = sorted(r1_left)
    print(r1_left)
    print(len(r1_left))

    r1_right = []
    for r, d, f in os.walk(r1_dir):
        for x in f:
            if 'right' in x and 'REF' not in x \
                    and not x.startswith('.') \
                    and not x.endswith('txt'):
                r1_right.append(x)
    r1_right = sorted(r1_right)
    print(r1_right)
    print(len(r1_right))

    r2_left = []
    r2_right = []
    for r, d, f in os.walk(r2_dir):
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

    combined = []
    for i in range(len(r1_left)):
        combined.append((r1_left[i], r1_right[i], r2_left[i], r2_right[i], img_files[i]))
        #print((r1_left[i], r1_right[i], r2_left[i], r2_right[i]))
    print(combined)

    all_together = []
    images_together = []
    for t in combined:
        idx = int(t[0].split('_')[1])
        print(idx)
        for s in range(len(t)):
            if s == len(t) - 1:
                if int(t[s].split('_')[1].split('.')[0]) is not idx:
                    raise Exception
            else:
                if int(t[s].split('_')[1]) is not idx:
                    raise Exception

        r1_left_mask = nib.load(r1_dir + t[0]).get_data()
        #r1_right_mask = nib.load(r1_dir + t[1]).get_data()
        r2_left_mask = nib.load(r2_dir + t[2]).get_data()
        #r2_right_mask = nib.load(r2_dir + t[3]).get_data()

        img = nib.load(r1_dir + t[4]).get_data()

        r1_left_mask[r1_left_mask > 0.5] = 1
        #r1_right_mask[r1_right_mask > 0.5] = 1
        r2_left_mask[r2_left_mask > 0.5] = 1
        #r2_right_mask[r2_right_mask > 0.5] = 1

        #both_left = np.zeros(r1_left_mask.shape)
        both_left = r1_left_mask + (2 * r2_left_mask)
        '''for x in range(both_left.shape[0]):
            for y in range(both_left.shape[1]):
                for z in range(both_left.shape[2]):
                    if r1_left_mask[x, y, z] == 1 and r2_left_mask[x, y, z] == 1:
                        both_left[x, y, z] = 3
                    if r1_left_mask[x, y, z] == 1 and r2_left_mask[x, y, z] == 0:
                        both_left[x, y, z] = 1
                    if r1_left_mask[x, y, z] == 0 and r2_left_mask[x, y, z] == 1:
                        both_left[x, y, z] = 2'''

        #all_together.append(both_left)
        #images_together.append(img)
        nii_file = nib.Nifti1Image(both_left, np.eye(4))
        nib.save(nii_file, "/media/max/Data_Max_2/lc-analysis/left_{}.nii.gz".format(idx))
        nii_file = nib.Nifti1Image(img, np.eye(4))
        nib.save(nii_file, "/media/max/Data_Max_2/lc-analysis/imgs_{}.nii.gz".format(idx))