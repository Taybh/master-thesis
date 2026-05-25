import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import nibabel as nib
import h5py as h5
import random
from datasets.LC_1rater import LC1Rater_Set
from sklearn.model_selection import train_test_split
#from post_processing import largest_concomp #khodam vared kardam
import rpy2.robjects as robjects
#from post_processing import boundary_plus_thresh

from sklearn.model_selection import KFold
from networks.unet import UNet3D
from datasets.SN import SNSet
#from networks.ResNet import ResnetGenerator, get_norm_layer
#from networks.attentionUnet import AttU_Net
#from networks.UnetPlusPlus import NestedUNet

torch.backends.cudnn.benchmark = True
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




'''
def load_gt_ref(index):

    ref_masks_left = nib.load(gt_ref_dir + os.sep + r1_left[index]).get_fdata()
    ref_masks_right = nib.load(gt_ref_dir + os.sep + r1_right[index]).get_fdata()

    #ref_masks = np.load(gt_ref_dir + os.sep + 'LC_{:02d}_REF_left.nii.gz'.format(index))['ref']
    return ref_masks_left, ref_masks_right
'''

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



def test_fold(test_idx, counter,origin_data, target,dsc_csv,cr_csv,cr_lists,
                          cr_gt_csv,cr_gt_lists,icc_csv):
    # load the weights and create the net
    net = UNet3D(in_channels=1, n_classes=2, depth=4, wf=5, padding=True, batch_norm=True)
    #net = AttU_Net(img_ch=1, output_ch=2)
    #net = NestedUNet()

    # net = ResnetGenerator(input_nc=1, output_nc=2, ngf=64, norm_layer=get_norm_layer('batch'), use_dropout=False,
    #                       n_blocks=9, padding_type='zero')
    # net = AttentionUnet(in_channels=1, out_channels=2, is_leaky=True)
    net.cuda()
    checkpoint = torch.load(model_dir + model_name + 'f' + str(counter) + '_wdh' +str(num_wdh)+ '/checkpoint_' + model_type)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    for tid in test_idx:
        print(str(tid))
        # load the data
        hf = h5.File('../data/SN-LC-3R-data.h5', 'r')  # ag error dad relavant bezan
        hf_ref = h5.File('../data/ref.h5', 'r')
        names = list(hf['img'])
        print('names: ', names)

        names_ref = list(hf_ref['ref'])
        #print('names_ref: ', names_ref)

        #print(hf.keys())
        data = hf['img'][names[tid]]
        img = np.array(data, dtype=np.float32)

        if target == 'SN':
            gt = hf['SN'][names[tid]]
        else:
            gt = hf['R3'][names[tid]]

        ref =hf_ref['ref'][names_ref[tid]]
        gt_ref = np.array(ref, dtype=np.float32)

        # normalize volume
        img = img / np.max(img)

        pred_masks = np.zeros(gt.shape, dtype=np.float32)
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
                    pred_masks[:, x_l:x_r, y_l:y_r, z_l:z_r] += output

        pred_masks[pred_masks < 0.5] = 0
        pred_masks[pred_masks > 0.5] = 1

        '''
        nii_file = nib.Nifti1Image(pred_masks[0], np.eye(4))
        nib.save(nii_file, output_dir  + '{:02d}'.format(tid) + '_0_' + model_name + '.nii.gz')
        nii_file = nib.Nifti1Image(pred_masks[1], np.eye(4))
        nib.save(nii_file, output_dir  + '{:02d}'.format(tid) + '_1_' + model_name + '.nii.gz')
        nii_file = nib.Nifti1Image(gt[0], np.eye(4))
        nib.save(nii_file, output_dir  + '{:02d}'.format(tid) + '_' + model_name + '_GT0.nii.gz')
        nii_file = nib.Nifti1Image(gt[1], np.eye(4))
        nib.save(nii_file, output_dir  + '{:02d}'.format(tid) + '_' + model_name + '_GT1.nii.gz')
        nii_file = nib.Nifti1Image(img, np.eye(4))
        nib.save(nii_file, output_dir  + '{:02d}'.format(tid) + '_' + model_name+ '_IMG.nii.gz')
'''

        gt = np.array(gt, dtype=np.float32)

        if np.max(gt) is not 1.0:
            print('gt max: ', np.max(gt))

        if np.max(gt_ref) is not 1.0:
            print('ref max: ', np.max(gt_ref))


        # calculate DSC
        dsc_all = dsc(gt[0] + gt[1], pred_masks[0] + pred_masks[1])
        dsc_0 = dsc(gt[0], pred_masks[0])
        dsc_1 = dsc(gt[1], pred_masks[1])
        print(dsc_all)
        print(dsc_0)
        print(dsc_1)

        dsc_csv += str(tid) + ',' + str(counter) + ',' + str(dsc_0) + ',' + str(dsc_1) + ',' + str(dsc_all) + '\n'

        # calculate contrast ratios
        cr_automated = cr(img, pred_masks[0], pred_masks[1], gt_ref[0], gt_ref[1])
        cr_csv += str(tid) + ',' + str(counter)+',' + str(cr_automated[0]) + ',' + str(cr_automated[1]) + ',' + str(
            cr_automated[2]) + ',' + str(cr_automated[3]) + '\n'
        cr_lists['median_left'].append(cr_automated[0])
        cr_lists['median_right'].append(cr_automated[1])
        cr_lists['max_left'].append(cr_automated[2])
        cr_lists['max_right'].append(cr_automated[3])

        cr_manual = cr(img, gt[0], gt[1], gt_ref[0], gt_ref[1])
        cr_gt_csv += str(tid) + ',' + str(counter)+',' + str(cr_manual[0]) + ',' + str(cr_manual[1]) + ',' + str(cr_manual[2]) + ',' + str(
            cr_manual[3]) + '\n'
        cr_gt_lists['median_left'].append(cr_manual[0])
        cr_gt_lists['median_right'].append(cr_manual[1])
        cr_gt_lists['max_left'].append(cr_manual[2])
        cr_gt_lists['max_right'].append(cr_manual[3])

        '''
        # calculate ICC for contrast ratios
        robjects.r('library(DescTools)')
        cr_types = ['median_left', 'median_right', 'max_left', 'max_right']
        for t in cr_types:
            assert len(cr_lists[t]) == len(cr_gt_lists[t])
            tmp_file = results_dir + os.sep + 'ICC_tmp'+os.sep
            if not os.path.exists(tmp_file):
                os.makedirs(tmp_file)

            with open(os.path.join(tmp_file , t +'.tsv'), 'w') as f:
                f.write('gt-' + t + '\t' + model_name + t + '\n')
                for v in range(len(cr_lists[t])):
                    f.write(str(cr_gt_lists[t][v]) + '\t' + str(cr_lists[t][v]) + '\n')
            robjects.r(t + ' <- read.csv("' + tmp_file + t + '.tsv' + '", sep="\t")')
            iccs = str(robjects.r('ICC(' + t + ', conf.level=' + str(conf_for_icc) + ')'))
            with open(tmp_file + os.sep + model_name + '_CR_ICC_R-Output_' + t, 'w') as f:
                f.write(iccs)
            for i in iter(iccs.splitlines()):
                if 'ICC3' in i and 'ICC3k' not in i:
                    icc = [x for x in i.split(' ') if x != ''][2]
            icc_csv += icc + ','
        icc_csv += '\n'
        '''

        with open(results_dir + os.sep + model_name + '_DSC.csv', 'w') as f:
            f.write(dsc_csv)
        with open(results_dir + os.sep + model_name + '_CRs_automated.csv', 'w') as f:
            f.write(cr_csv)
        with open(results_dir + os.sep + model_name + '_CRs_manual.csv', 'w') as f:
            f.write(cr_gt_csv)
        #with open(results_dir + os.sep + model_name + '_CRs_ICCs_conf-' + str(conf_for_icc) + '.csv', 'w') as f:
        #    f.write(icc_csv)

        hf.close()
        hf_ref.close()

# reproducibility stuff
random.seed(131294)
np.random.seed(131294)
torch.manual_seed(131294)
torch.cuda.manual_seed(131294)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    patchsize = 128
    origin = ['sn','lc']
    target_test = ['SN','R3']
    num_samples = [22,42,62]
    num_wdh = 1 #train repetitions
    indices_total = range(81)
    model_type = 'best'  # 'latest'

    #orig_data_dir = '/home/tayebeh/PycharmProjects/LC/data/lc/BETTS82'
    # gt_lc_dir = '/home/max/git/LCSN-Seg/data/lc/BETTS82'
    #gt_ref_dir = '/home/tayebeh/PycharmProjects/LC/data/lc/BETTS82'

    conf_for_icc = 0.95
    write_pp_masks = True

    num_folds = 5
    for num in num_samples:
        for org in origin:
            for tar in target_test:
                indices = indices_total
                print(indices)

                train_val_s, test_s = train_test_split(indices_total, test_size=0.27, shuffle=True,
                                                       random_state=131294)
                print(('test', test_s))

                if org =='lc':
                    model_dir = "logs/scratch" + str(num)+ "/lc/"
                    model_name = 'unet_LC_Scratch'
                else:
                    model_dir = "logs/scratch" + str(num)+ "/sn/"
                    model_name = 'unet_SN_Scratch'

                # create results directory
                results_dir = 'EVAL_scratch' + str(num)+'_' +str(org) +'_' + 'to'+ '_'+str(tar)

                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                #if os.path.isdir(results_dir):
                #    count = 1
                #    results_dir = 'EVAL_scratch' + str(count) + '_'+ str(org) + '_' + 'to'+ '_' +str(tar)
                #    while os.path.isdir(results_dir):
                #        count += 1
                #        results_dir = 'EVAL_scratch' + str(count) + '_' + str(org) + '_' + 'to'+ '_' + str(tar)
                #os.makedirs(results_dir)


                dsc_csv = 'Subject,fold,DSC_left,DSC_right,DSC_all\n'
                cr_csv = 'Subject,fold,CR_median_left,CR_median_right,CR_max_left,CR_max_right\n'
                cr_lists = {'median_left': [],
                            'median_right': [],
                            'max_left': [],
                            'max_right': []}
                cr_gt_csv = 'Subject,fold,CR_median_left,CR_median_right,CR_max_left,CR_max_right\n'
                cr_gt_lists = {'median_left': [],
                               'median_right': [],
                               'max_left': [],
                               'max_right': []}
                icc_csv = 'CR_median_left,CR_median_right,CR_max_left,CR_max_right\n'

                for i in range(num_folds):
                    print('number of samples:'+str(num)+'  origin:'+str(org)+'  target:'+str(tar)+'  fold:'+str(i))
                    test_fold(test_s, counter=i,origin_data=org, target=tar,dsc_csv=dsc_csv,cr_csv=cr_csv,cr_lists=cr_lists,
                              cr_gt_csv=cr_gt_csv,cr_gt_lists=cr_gt_lists,icc_csv=icc_csv)

