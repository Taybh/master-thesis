import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def test_fold(test_idx, counter):
    # load the weights and create the net
    net = UNet3D(in_channels=1, n_classes=2, depth=4, wf=5, padding=True, batch_norm=True)
    #net = ResnetGenerator(input_nc=1, output_nc=2, ngf=64, norm_layer=get_norm_layer('batch'), use_dropout=False,
                          #n_blocks=9, padding_type='zero')
    # net = AttentionUnet(in_channels=1, out_channels=2, is_leaky=True)
    net.cuda()

    checkpoint = torch.load(model_dir + model_name + 'f' + str(counter) + '/checkpoint_' + model_type)
    #checkpoint = torch.load(model_dir + model_name + '/checkpoint_' + model_type)
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
        nib.save(nii_file, '/pool/max/results/lc_PD-test/' + model_name + model_type + '/' + '{:02d}'.format(
            tid) + '_0_' + model_name + '.nii.gz')
        nii_file = nib.Nifti1Image(result[1], np.eye(4))
        nib.save(nii_file, '/pool/max/results/lc_PD-test/' + model_name + model_type + '/' + '{:02d}'.format(
            tid) + '_1_' + model_name + '.nii.gz')
        nii_file = nib.Nifti1Image(gt[0], np.eye(4))
        nib.save(nii_file, '/pool/max/results/lc_PD-test/' + model_name + model_type + '/' + '{:02d}'.format(
            tid) + '_' + model_name + '_GT0.nii.gz')
        nii_file = nib.Nifti1Image(gt[1], np.eye(4))
        nib.save(nii_file, '/pool/max/results/lc_PD-test/' + model_name + model_type + '/' + '{:02d}'.format(
            tid) + '_' + model_name + '_GT1.nii.gz')
        nii_file = nib.Nifti1Image(img, np.eye(4))
        nib.save(nii_file, '/pool/max/results/lc_PD-test/' + model_name + model_type + '/' + '{:02d}'.format(
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
    model_name = "LC_03_"
    model_dir = "/pool/max/train_logs/lc/"
    indices = range(20)
    model_type = 'latest'  # 'best'
    data_dir = 'data/lc_pd_processed/'

    # # load the data
    # data_dir = 'data/lc_processed/'
    # img = []
    # gt = []
    # for i in indices:
    #     #if i == 29:
    #     data = np.load(data_dir + 'lc_data_{:02d}.npz'.format(i))
    #     img.append(np.array(data['img'], dtype=np.float32))
    #     gt.append(data['gt'])
    #
    # # normalize each volume
    # for i in range(len(indices)):
    #     img[i] = img[i] / np.max(img[i])

    text_file = open("DSC-test-PD_" + model_name + model_type + ".csv", "w")
    text_file.write('subject,DSC_0,DSC_1,DSC_all\n')

    test_fold(indices, 4)

    text_file.close()
