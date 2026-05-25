import nibabel as nib
import numpy as np

if __name__ == '__main__':
    index = 10
    idx_gt = index +1
    model_name = 'SN_LC_03_2R_2ndPart_'
    gt0 = nib.load('/home/tayebeh/PycharmProjects/LC/data/sn/BETTS82/'+'LC_{:02d}_SN_LEFT_MS.nii.gz'.format(idx_gt)).get_fdata()
    gt1 = nib.load('/home/tayebeh/PycharmProjects/LC/data/sn/BETTS82/'+'LC_{:02d}_SN_RIGHT_MS.nii.gz'.format(idx_gt)).get_fdata()
    pred0 = nib.load('/home/tayebeh/PycharmProjects/LC/EVAL-one4all_SN_result_SN_LC_03_2R_2ndPart_/post_processed_lc_masks/'+'{:02d}_0_{}post_processed.nii.gz'.format(index, model_name)).get_fdata()
    pred1 = nib.load('/home/tayebeh/PycharmProjects/LC/EVAL-one4all_SN_result_SN_LC_03_2R_2ndPart_/post_processed_lc_masks/'+'{:02d}_1_{}post_processed.nii.gz'.format(index, model_name)).get_fdata()

    gt = gt0 + gt1
    pred = pred0 + pred1

    combined = np.zeros(gt.shape)
    # False-Negative: 1
    combined[np.logical_and(gt > 0.5, pred < 0.5)] = 1
    # False-Positve: 3
    combined[np.logical_and(gt < 0.5, pred > 0.5)] = 3
    # Overlap: 2
    combined[np.logical_and(gt > 0.5, pred > 0.5)] = 2

    nii_file = nib.Nifti1Image(combined, np.eye(4))
    nib.save(nii_file, 'RESVIS_' + model_name + '_' + str(index) + '.nii.gz')
