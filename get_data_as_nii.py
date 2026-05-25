import numpy as np
import nibabel as nib


if __name__=='__main__':
    # load the respective data
    data_dir = 'data/lc_processed/'
    i = 78
    data = np.load(data_dir + 'lc_data_{:02d}.npz'.format(i))
    img = np.array(data['img'], dtype=np.float32)
    gt = data['gt']

    print(np.unique(img))
    print(np.unique(gt))

    gt = np.array(gt, dtype=np.float32)
    print(np.unique(gt))

    data_file = nib.Nifti1Image(img, np.eye(4))
    nib.save(data_file, '{:02d}_data.nii.gz'.format(i))
    gt0_file = nib.Nifti1Image(gt[0], np.eye(4))
    nib.save(gt0_file, '{:02d}_gt0.nii.gz'.format(i))
    gt1_file = nib.Nifti1Image(gt[1], np.eye(4))
    nib.save(gt1_file, '{:02d}_gt1.nii.gz'.format(i))
