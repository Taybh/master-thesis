import numpy as np
import transforms as nptransforms
import nibabel as nib
import time
import h5py as h5
import random

from torch.utils.data import Dataset
from torchvision import transforms


class LCSetTwoRaters(Dataset):
    '''Locus Coeruleus Segmentation Dataset'''

    def __init__(self, indices, augmentation_params, data_file='data/LC-R2-data.h5', validation=False):
        # get the data handles
        self.hf = h5.File(data_file, 'r')
        self.img = []
        self.gt1 = []
        self.gt2 = []
        names = list(self.hf['img'])
        for i in indices:
            # data = np.load(data_dir + 'lc_data_{:02d}.npz'.format(i))
            # self.img.append(np.array(data['img'], dtype=np.float32))
            # self.gt.append(data['gt'])
            self.img.append(self.hf['img'][names[i]])
            self.gt1.append(self.hf['R1'][names[i]])
            self.gt2.append(self.hf['R2'][names[i]])

        # get augmentation parameters
        if not augmentation_params:
            self.aug_params = {'patch_size': augmentation_params.get('patch_size', 64.),
                               'fg_rate': augmentation_params.get('fg_rate', 0.5)}
        else:
            self.aug_params = augmentation_params

        # add padding once for later patch extraction
        self.gt1_boxes = [mask.attrs['bbox'] for mask in self.gt1]
        self.gt2_boxes = [mask.attrs['bbox'] for mask in self.gt2]

        self.validation = validation

        self.transform = transforms.Compose([
            nptransforms.ToTensorTwoTargetClasses()
        ])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        choose_R1 = random.random() <= 0.5
        if choose_R1:
            patch = self.extract_patch(self.img[item], self.gt1[item], self.gt1_boxes[item])
        else:
            patch = self.extract_patch(self.img[item], self.gt2[item], self.gt2_boxes[item])
        # (self.data[item][0][..., 125][64:576, 64:576], self.data[item][1][..., 125][64:576, 64:576])
        # if not self.validation:
        # patch = self.augment_patch(self.extract_patch(self.data[item]))
        return self.transform(patch)

    def extract_patch(self, img, gt, bbox):
        ps2 = int(self.aug_params['patch_size'] / 2)

        # choose position of fg or bg patch
        fg = np.random.random() < self.aug_params['fg_rate']
        if fg:
            # choose random patch that intersects well with the bounding box of the segments
            overlap = 0.5
            center = [np.random.randint(bbox[0] - ps2 + overlap * (bbox[3] - bbox[0]),
                                        bbox[3] + ps2 - overlap * (bbox[3] - bbox[0])),
                      np.random.randint(bbox[1] - ps2 + overlap * (bbox[4] - bbox[1]),
                                        bbox[4] + ps2 - overlap * (bbox[4] - bbox[1])),
                      np.random.randint(bbox[2] - ps2 + overlap * (bbox[5] - bbox[2]),
                                        bbox[5] + ps2 - overlap * (bbox[5] - bbox[2]))]
        else:
            # TODO: Think of a better solution for randomly drawing a background sample
            # choose random background patch
            intersect = True
            while intersect:
                center = [np.random.randint(0 + ps2, img.shape[0] - ps2),
                          np.random.randint(0 + ps2, img.shape[1] - ps2),
                          np.random.randint(0 + ps2, img.shape[2] - ps2)]
                intersect = self.boxes_intersect(center, ps2, bbox)

        # ensure to stay within bounds
        if center[0] < ps2:
            center[0] = ps2
        elif center[0] > img.shape[0]-ps2:
            center[0] = img.shape[0]-ps2
        if center[1] < ps2:
            center[1] = ps2
        elif center[1] > img.shape[0]-ps2:
            center[1] = img.shape[0]-ps2
        if center[2] < ps2:
            center[2] = ps2
        elif center[2] > img.shape[0]-ps2:
            center[2] = img.shape[0]-ps2

        patch_img = img[center[0] - ps2:center[0] + ps2,
                        center[1] - ps2:center[1] + ps2,
                        center[2] - ps2:center[2] + ps2]
        patch_gt = gt[:,
                      center[0] - ps2:center[0] + ps2,
                      center[1] - ps2:center[1] + ps2,
                      center[2] - ps2:center[2] + ps2]

        # nii_file = nib.Nifti1Image(patch_img, np.eye(4))
        # nib.save(nii_file, 'img.nii')
        # nii_file = nib.Nifti1Image(patch_gt, np.eye(4))
        # nib.save(nii_file, 'gt.nii')
        return patch_img, patch_gt

    def fast_zero_pad(self, img, pad_size):
        img_pad = np.zeros((img.shape[0] + 2 * pad_size, img.shape[1] + 2 * pad_size, img.shape[2] + 2 * pad_size),
                           dtype=np.float32)
        img_pad[pad_size:img_pad.shape[0] - pad_size,
        pad_size:img_pad.shape[1] - pad_size,
        pad_size:img_pad.shape[2] - pad_size] = img
        return img_pad

    def boxes_intersect(self, center, ps2, seg_box):
        return seg_box[0] - ps2 < center[0] < seg_box[3] + ps2 and \
               seg_box[1] - ps2 < center[1] < seg_box[4] + ps2 and \
               seg_box[2] - ps2 < center[2] < seg_box[5] + ps2

    def augment_patch(self, patch):
        return patch


class LCSet(Dataset):
    '''Locus Coeruleus Segmentation Dataset'''

    def __init__(self, indices, augmentation_params, data_dir='data/lc_processed/', validation=False):
        # load the respective data
        self.img = []
        self.gt = []
        for i in indices:
            data = np.load(data_dir + 'lc_data_{:02d}.npz'.format(i))
            self.img.append(np.array(data['img'], dtype=np.float32))
            self.gt.append(data['gt'])

        # normalize each volume
        for i in range(len(indices)):
            self.img[i] = self.img[i] / np.max(self.img[i])
        # self.data = data

        # get augmentation parameters
        if not augmentation_params:
            self.aug_params = {'patch_size': augmentation_params.get('patch_size', 64.),
                               'fg_rate': augmentation_params.get('fg_rate', 0.5)}
        else:
            self.aug_params = augmentation_params

        # add padding once for later patch extraction
        self.gt_boxes = []
        ps2 = int(self.aug_params['patch_size'] / 2)
        for idx, d in enumerate(self.gt):
            gt = d[0] + d[1]
            segments = np.where(gt > 0.5)
            bbox = (np.min(segments[0]),
                    np.min(segments[1]),
                    np.min(segments[2]),
                    np.max(segments[0]),
                    np.max(segments[1]),
                    np.max(segments[2]))
            self.gt_boxes.append(bbox)

        self.validation = validation

        self.transform = transforms.Compose([
            nptransforms.ToTensorTwoTargetClasses()
        ])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        patch = self.extract_patch(self.img[item], self.gt[item], self.gt_boxes[item])
        # (self.data[item][0][..., 125][64:576, 64:576], self.data[item][1][..., 125][64:576, 64:576])
        # if not self.validation:
        # patch = self.augment_patch(self.extract_patch(self.data[item]))
        return self.transform(patch)

    def extract_patch(self, img, gt, bbox):
        ps2 = int(self.aug_params['patch_size'] / 2)

        # choose position of fg or bg patch
        fg = np.random.random() < self.aug_params['fg_rate']
        if fg:
            # choose random patch that intersects well with the bounding box of the segments
            overlap = 0.5
            center = [np.random.randint(bbox[0] - ps2 + overlap * (bbox[3] - bbox[0]),
                                        bbox[3] + ps2 - overlap * (bbox[3] - bbox[0])),
                      np.random.randint(bbox[1] - ps2 + overlap * (bbox[4] - bbox[1]),
                                        bbox[4] + ps2 - overlap * (bbox[4] - bbox[1])),
                      np.random.randint(bbox[2] - ps2 + overlap * (bbox[5] - bbox[2]),
                                        bbox[5] + ps2 - overlap * (bbox[5] - bbox[2]))]
        else:
            # TODO: Think of a better solution for randomly drawing a background sample
            # choose random background patch
            intersect = True
            while intersect:
                center = [np.random.randint(0 + ps2, img.shape[0] - ps2),
                          np.random.randint(0 + ps2, img.shape[1] - ps2),
                          np.random.randint(0 + ps2, img.shape[2] - ps2)]
                intersect = self.boxes_intersect(center, ps2, bbox)

        # ensure to stay within bounds
        if center[0] < ps2:
            center[0] = ps2
        elif center[0] > img.shape[0]-ps2:
            center[0] = img.shape[0]-ps2
        if center[1] < ps2:
            center[1] = ps2
        elif center[1] > img.shape[0]-ps2:
            center[1] = img.shape[0]-ps2
        if center[2] < ps2:
            center[2] = ps2
        elif center[2] > img.shape[0]-ps2:
            center[2] = img.shape[0]-ps2

        patch_img = img[center[0] - ps2:center[0] + ps2,
                        center[1] - ps2:center[1] + ps2,
                        center[2] - ps2:center[2] + ps2]
        patch_gt = gt[:,
                      center[0] - ps2:center[0] + ps2,
                      center[1] - ps2:center[1] + ps2,
                      center[2] - ps2:center[2] + ps2]

        # nii_file = nib.Nifti1Image(patch_img, np.eye(4))
        # nib.save(nii_file, 'img.nii')
        # nii_file = nib.Nifti1Image(patch_gt, np.eye(4))
        # nib.save(nii_file, 'gt.nii')
        return patch_img, patch_gt

    def fast_zero_pad(self, img, pad_size):
        img_pad = np.zeros((img.shape[0] + 2 * pad_size, img.shape[1] + 2 * pad_size, img.shape[2] + 2 * pad_size),
                           dtype=np.float32)
        img_pad[pad_size:img_pad.shape[0] - pad_size,
        pad_size:img_pad.shape[1] - pad_size,
        pad_size:img_pad.shape[2] - pad_size] = img
        return img_pad

    def boxes_intersect(self, center, ps2, seg_box):
        return seg_box[0] - ps2 < center[0] < seg_box[3] + ps2 and \
               seg_box[1] - ps2 < center[1] < seg_box[4] + ps2 and \
               seg_box[2] - ps2 < center[2] < seg_box[5] + ps2

    def augment_patch(self, patch):
        return patch
