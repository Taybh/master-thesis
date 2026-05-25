import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
import nibabel as nib
import h5py as h5

from sklearn.model_selection import train_test_split

from networks.unet import UNet3D


def get_patch_borders(x, overlap, max):
    x_l = x * int(patchsize / overlap)
    x_r = x * int(patchsize / overlap) + patchsize
    if x_r >= max:
        x_l = max - patchsize
        x_r = max
    return x_l, x_r


def test(test_idx):
    for counter in range(num_kfolds):
        # load the weights and create the net
        net = UNet3D(in_channels=1, n_classes=2, depth=4, wf=5, padding=True, batch_norm=True)
        # net = ResnetGenerator(input_nc=1, output_nc=2, ngf=64, norm_layer=get_norm_layer('batch'), use_dropout=False,
        #                       n_blocks=9, padding_type='zero')
        # net = AttentionUnet(in_channels=1, out_channels=2, is_leaky=True)
        net.cuda()

        checkpoint = torch.load(model_dir + model_name + 'f' + str(counter) + '/checkpoint_' + model_type)
        #checkpoint = torch.load(model_dir + model_name + '/checkpoint_' + model_type)
        net.load_state_dict(checkpoint['model0_state_dict'])
        net.eval()

        for tid in test_idx:
            # load the data
            data = hf['img'][names[tid]]
            img = np.array(data, dtype=np.float32)
            res_shape = [2]
            res_shape.extend(list(img.shape))

            result = np.zeros(res_shape, dtype=np.float32)
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
            nib.save(nii_file, output_dir + '{:02d}'.format(tid) + '_0_' + model_name + 'f' + str(counter) + '.nii.gz')
            nii_file = nib.Nifti1Image(result[1], np.eye(4))
            nib.save(nii_file, output_dir + '{:02d}'.format(tid) + '_1_' + model_name + 'f' + str(counter) + '.nii.gz')


if __name__ == '__main__':
    rnd_seed = 131294
    patchsize = 128
    model_name = "LC_B_R1-R3_"
    model_dir = "/pool/max/train_logs/lc-v2/"
    num_kfolds = 3
    indices = range(82)
    model_type = 'latest'
    output_dir = '/pool/max/results/lc2/' + model_name + model_type + '/'
    data_file = 'data/LC-3R-data.h5'
    hf = h5.File(data_file, 'r', swmr=True)
    names = list(hf['img'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_val_s, test_s = train_test_split(indices, test_size=0.27, shuffle=True, random_state=rnd_seed)
    print(('test', test_s))
    test(test_s)
