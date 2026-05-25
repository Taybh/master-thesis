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
#from post_processing import boundary_plus_thresh

from sklearn.model_selection import KFold
from networks.unet import UNet3D
from datasets.SN import SNSet
from random import sample
#from networks.ResNet import ResnetGenerator, get_norm_layer
#from networks.attentionUnet import AttU_Net
#from networks.UnetPlusPlus import NestedUNet

torch.backends.cudnn.benchmark = True
# from networks.attention_unet import AttentionUnet

#import rpy2.robjects as robjects
#import rpy2.robjects.packages as rpackages
#import rpy2.robjects.vectors as StrVector

import sys

import rpy2
from rpy2.robjects import IntVector, pandas2ri
from rpy2.robjects.packages import importr

#    Calculates Interclass Correlation Coefficient (3,1) as defined in
#    P. E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses in
#    Assessing Rater Reliability". Psychological Bulletin 86 (2): 420-428.
# args. ground_truth, prediction

# def ICC31():
#     ground_truth = np.random.uniform(low=0.5, high=13.3, size=(10,))
#     prediction = np.random.uniform(low=0.5, high=20.1, size=(10,))
#     print('ground_truth =' + str(ground_truth))
#     print('prediction =' + str(prediction))
#     print("---------------------------")
#     # remove NaN values
#     idx = np.squeeze(~np.isnan(ground_truth))
#     print('idx =' + str(idx))
#     ground_truth = ground_truth[idx]
#     print('ground_truth =' + str(ground_truth))
#     prediction = prediction[idx]
#     print('prediction =' + str(prediction))

#     dat = np.column_stack((ground_truth, prediction))
#     # number of raters/ratings
#     k = 2;
#     # number of targets
#     n = dat.shape[0];
#     print('n =' + str(n))
#     # mean per target
#     mpt = np.mean(dat, axis=1);
#     print('mpt' + str(mpt))
#     mptshap = mpt.shape = (n,1);
#     print('mpt.shap =' + str(mptshap))
#     # mean per rater/rating
#     mpr = np.mean(dat, axis=0);
#     print('mpr =' + str(mpr))
#     # get total mean
#     tm = np.mean(mpt);
#     print('tm =' + str(tm))
#     # within target sum sqrs
#     WSS = np.sum(np.sum((dat-mpt)**2));
#     print('WSS =' + str(WSS))
#     # within target mean sqrs
#     WMS = WSS / (n * (k - 1));
#     print('WMS =' + str(WMS))
#     # between rater sum sqrs
#     RSS = np.sum((mpr - tm)**2) * n;
#     print('RSS =' + str(RSS))
#     # between rater mean sqrs
#     RMS = RSS / (k - 1);
#     print('RMS =' + str(RMS))
#     # between target sum sqrs
#     BSS = np.sum((mpt - tm)**2) * k;
#     print('BSS = ' + str(BSS))
#     # between targets mean squares
#     BMS = BSS / (n - 1);
#     print('BMS =' + str(BMS))
#     # residual sum of squares
#     ESS = WSS - RSS;
#     print('ESS =' + str(ESS))
#     # residual mean sqrs
#     EMS = ESS / ((k - 1) * (n - 1));
#     print('EMS =' + str(EMS))
#     # ICC(3,1)
#     return print ((BMS - EMS) / (BMS + (k - 1) * EMS))


# run it by --> interclass_correlation_coefficient.py [1,2,3,4,5 ],[1,2,3,4,5]


def ICC31(ground_truth, prediction):
    # remove NaN values
    #idx = np.squeeze(~np.isnan(ground_truth))
    #ground_truth = ground_truth[idx]
    #prediction = prediction[idx]

    dat = np.column_stack((ground_truth, prediction))
    # number of raters/ratings
    k = 2
    # number of targets
    n = dat.shape[0]
    # mean per target
    mpt = np.mean(dat, axis=1)
    mpt.shape = (n, 1)
    # mean per rater/rating
    mpr = np.mean(dat, axis=0)
    # get total mean
    tm = np.mean(mpt)
    # within target sum sqrs
    WSS = np.sum(np.sum((dat - mpt) ** 2))
    # within target mean sqrs
    WMS = WSS / (n * (k - 1))
    # between rater sum sqrs
    RSS = np.sum((mpr - tm) ** 2) * n
    # between rater mean sqrs
    RMS = RSS / (k - 1)
    # between target sum sqrs
    BSS = np.sum((mpt - tm) ** 2) * k
    # between targets mean squares
    BMS = BSS / (n - 1)
    # residual sum of squares
    ESS = WSS - RSS
    # residual mean sqrs
    EMS = ESS / ((k - 1) * (n - 1))
    # ICC(3,1)

    return (BMS - EMS) / (BMS + (k - 1) * EMS)


#def main():
 #   return ICC31([1, 2, 3, 4, 5][1, 2, 3, 4, 5])  # *sys.argv[1:]



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



def test_fold(test_idx, counter,target):
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

    hf = h5.File('../data/SN-LC-3R-data.h5', 'r')  # ag error dad relavant bezan
    hf_ref = h5.File('../data/ref.h5', 'r')
    names = list(hf['img'])
    # print('names: ', names)

    names_ref = list(hf_ref['ref'])

    for tid in test_idx:
        print(str(tid))
        # load the data

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

        '''
        # calculate DSC
        dsc_all = dsc(gt[0] + gt[1], pred_masks[0] + pred_masks[1])
        dsc_0 = dsc(gt[0], pred_masks[0])
        dsc_1 = dsc(gt[1], pred_masks[1])
        print(dsc_all)
        print(dsc_0)
        print(dsc_1)

        dsc_csv.write(str(tid) + ',' + str(counter) + ',' + str(dsc_0) + ',' + str(dsc_1) + ',' + str(dsc_all) + '\n')
'''
        # calculate contrast ratios
        cr_automated = cr(img, pred_masks[0], pred_masks[1], gt_ref[0], gt_ref[1])
        #cr_csv.write(str(tid) + ',' + str(counter)+',' + str(cr_automated[0]) + ',' + str(cr_automated[1]) + ',' + str(
        #    cr_automated[2]) + ',' + str(cr_automated[3]) + '\n')
        cr_lists['median_left'].append(cr_automated[0])
        cr_lists['median_right'].append(cr_automated[1])
        cr_lists['max_left'].append(cr_automated[2])
        cr_lists['max_right'].append(cr_automated[3])

        cr_manual = cr(img, gt[0], gt[1], gt_ref[0], gt_ref[1])
        #cr_gt_csv.write(str(tid) + ',' + str(counter)+',' + str(cr_manual[0]) + ',' + str(cr_manual[1]) + ',' + str(cr_manual[2]) + ',' + str(
        #    cr_manual[3]) + '\n')
        cr_gt_lists['median_left'].append(cr_manual[0])
        cr_gt_lists['median_right'].append(cr_manual[1])
        cr_gt_lists['max_left'].append(cr_manual[2])
        cr_gt_lists['max_right'].append(cr_manual[3])


        r_icc = importr("ICC")
        df = DataFrame({"automated": FloatVector(cr_lists['median_left']),
                        "manual": FloatVector(cr_gt_lists['median_left'])})
        icc_res = r_icc.ICCbare("automated", "manual", data=df)
        icc_val = icc_res[0]  # icc_val now holds the icc value

        # check whether icc value equals reference value
        print(isclose(icc_val, 0.728, abs_tol=0.001))



        #ICC_median_left = ICC31(cr_gt_lists['median_left'],cr_lists['median_left'])
        #print(ICC_median_left)
        #ICC_median_right =
        #ICC_max_left =
        #ICC_max_right =


        '''
    # calculate ICC for contrast ratios
    #robjects.r('library(DescTools)')
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
'''


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
    num_samples = [22, 42, 62]
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
                indices = sample(indices_total, num)
                print(indices)

                train_val_s, test_s = train_test_split(indices, test_size=0.27, shuffle=True,
                                                       random_state=131294)

                print(('test', test_s))

                if org =='lc':
                    model_dir = "logs/scratch" + str(num)+ "/lc/"
                    model_name = 'unet_lc_Scratch'
                else:
                    model_dir = "logs/scratch" + str(num)+ "/sn/"
                    model_name = 'unet_sn_Scratch'

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

                cr_lists = {'median_left': [],
                            'median_right': [],
                            'max_left': [],
                            'max_right': []}

                cr_gt_lists = {'median_left': [],
                               'median_right': [],
                               'max_left': [],
                               'max_right': []}

                #dsc_csv = open(results_dir + os.sep + model_name + '_DSC.csv', 'w')
                #dsc_csv.write('Subject,fold,DSC_left,DSC_right,DSC_all\n')


                #cr_csv = open(results_dir + os.sep + model_name + '_CRs_automated.csv', 'w')
                #cr_csv.write('Subject,fold,CR_median_left,CR_median_right,CR_max_left,CR_max_right\n')


                #cr_gt_csv = open(results_dir + os.sep + model_name + '_CRs_manual.csv', 'w')
                #cr_gt_csv.write('Subject,fold,CR_median_left,CR_median_right,CR_max_left,CR_max_right\n')


                icc_csv = open(results_dir + os.sep + model_name + '_CRs_ICCs_conf-' + str(conf_for_icc) + '.csv', 'w')
                icc_csv.write('CR_median_left,CR_median_right,CR_max_left,CR_max_right\n')

                for i in range(num_folds):
                    print('number of samples:'+str(num)+'  origin:'+str(org)+'  target:'+str(tar)+'  fold:'+str(i))
                    test_fold(test_s, counter=i, target=tar)

                #dsc_csv.close()
                #cr_csv.close()
                #cr_gt_csv.close()
                icc_csv.close()



