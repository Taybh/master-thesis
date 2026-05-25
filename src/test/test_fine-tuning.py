import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import torch
import nibabel as nib

from sklearn.model_selection import KFold
from networks.unet import UNet3D
from networks.ResNet import ResnetGenerator, get_norm_layer


# from networks.attention_unet import AttentionUnet


def dsc(gt, pred):
    intersection = np.sum(pred * gt)
    net_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    return float(2 * intersection) / (net_sum + gt_sum)


def get_patch_borders(x, overlap, max):
    x_l = x * int(patchsize / overlap)
    x_r = x * int(patchsize / overlap) + patchsize
    if x_r >= max:
        x_l = max - patchsize
        x_r = max
    return x_l, x_r


def test_fold(test_idx, fold=0, fold2=0, wdh=0):
    # load the weights and create the net
    net = UNet3D(in_channels=1, n_classes=2, depth=4, wf=5, padding=True, batch_norm=True)
    net.cuda()

    model_n = model_name + 'f' + str(fold) + '_f2-' + str(fold2) + '_wdh' + str(wdh) if wdh is not 0 else model_name + 'f' + str(fold) + '_f2-' + str(fold2)
    checkpoint = torch.load(model_dir + model_n + '/checkpoint_' + model_type)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    for tid in test_idx:
        # load the data
        data = np.load(data_dir + 'lc_data_{:02d}.npz'.format(tid))
        img = np.array(data['img'], dtype=np.float32)
        gt = data['gt']

        # normalize volume
        img = img / np.max(img)

        result = np.zeros(gt.shape, dtype=np.float32)
        img_shape = img.shape
        overlap_factor = 2
        x_range = overlap_factor * (
            img_shape[0] / patchsize if img_shape[0] % patchsize == 0 else int(img_shape[0] / patchsize) + 1)
        y_range = overlap_factor * (
            img_shape[1] / patchsize if img_shape[1] % patchsize == 0 else int(img_shape[1] / patchsize) + 1)
        z_range = overlap_factor * (
            img_shape[2] / patchsize if img_shape[2] % patchsize == 0 else int(img_shape[2] / patchsize) + 1)
        for x in range(x_range):
            for y in range(y_range):
                for z in range(z_range):
                    x_l, x_r = get_patch_borders(x, overlap_factor, img_shape[0])
                    y_l, y_r = get_patch_borders(y, overlap_factor, img_shape[1])
                    z_l, z_r = get_patch_borders(z, overlap_factor, img_shape[2])

                    net_input = np.expand_dims(np.expand_dims(img[x_l:x_r, y_l:y_r, z_l:z_r], axis=0), axis=0)
                    net_input = torch.from_numpy(net_input).cuda()
                    output = net(net_input)
                    output = np.squeeze(output.cpu().detach().numpy())
                    output[output < 0.5] = 0
                    output[output > 0.5] = 1
                    # nii_file = nib.Nifti1Image(output[0], np.eye(4))
                    # nib.save(nii_file, ('/pool/max/results/lc/' + model_name + '/' + str(
                    #     tid) + 'PATCH-{}-{}-{}_0_' + model_name + '.nii.gz').format(x, y, z))
                    # nii_file = nib.Nifti1Image(output[1], np.eye(4))
                    # nib.save(nii_file, ('/pool/max/results/lc/' + model_name + '/' + str(
                    #     tid) + 'PATCH-{}-{}-{}_1_' + model_name + '.nii.gz').format(x, y, z))
                    result[:, x_l:x_r, y_l:y_r, z_l:z_r] += output

        result[result < 0.5] = 0
        result[result > 0.5] = 1

        nii_file = nib.Nifti1Image(result[0], np.eye(4))
        nib.save(nii_file, '/pool/max/results/lc/' + model_name + model_type + '/' + '{:02d}'.format(
            tid) + '_0_' + model_name + '.nii.gz')
        nii_file = nib.Nifti1Image(result[1], np.eye(4))
        nib.save(nii_file, '/pool/max/results/lc/' + model_name + model_type + '/' + '{:02d}'.format(
            tid) + '_1_' + model_name + '.nii.gz')
        nii_file = nib.Nifti1Image(gt[0], np.eye(4))
        nib.save(nii_file, '/pool/max/results/lc/' + model_name + model_type + '/' + '{:02d}'.format(
            tid) + '_' + model_name + '_GT0.nii.gz')
        nii_file = nib.Nifti1Image(gt[1], np.eye(4))
        nib.save(nii_file, '/pool/max/results/lc/' + model_name + model_type + '/' + '{:02d}'.format(
            tid) + '_' + model_name + '_GT1.nii.gz')
        nii_file = nib.Nifti1Image(img, np.eye(4))
        nib.save(nii_file, '/pool/max/results/lc/' + model_name + model_type + '/' + '{:02d}'.format(
            tid) + '_' + model_name + '_IMG.nii.gz')

        gt = np.array(gt, dtype=np.float32)
        if np.max(gt) is not 1.0:
            print(np.max(gt))

        dsc_all = dsc(gt, result)
        dsc_0 = dsc(gt[0], result[0])
        dsc_1 = dsc(gt[1], result[1])

        text_file.write(str(tid) + ',' + str(dsc_0) + ',' + str(dsc_1) + ',' + str(dsc_all) + '\n')


if __name__ == '__main__':
    patchsize = 128
    model_name = "LC_03_PD-tuned_"
    model_dir = "/pool/max/train_logs/lc/"
    num_wdh = 1
    num_kfolds = 2
    indices = range(20)
    model_type = 'best'  # 'best', 'latest'
    data_dir = 'data/lc_pd_processed/'

    text_file = open("DSC-test-folds_" + model_name + model_type + ".csv", "w")
    text_file.write('subject,DSC_0,DSC_1,DSC_all\n')
    for w in range(num_wdh):
        kf = KFold(num_kfolds, shuffle=True, random_state=131294 + w)
        for counter, (train_val_indices, test_indices) in enumerate(kf.split(indices)):
            train_val_indices = [indices[i] for i in train_val_indices]
            test_indices = [indices[i] for i in test_indices]
            for counter2, (train_indices, val_indices) in enumerate(kf.split(train_val_indices)):
                train_indices = [train_val_indices[i] for i in train_indices]
                val_indices = [train_val_indices[i] for i in val_indices]
                print(test_indices)
                print(model_name, counter, counter2, w)
                test_fold(test_indices, fold=counter, fold2=counter2, wdh=w)

    text_file.close()
