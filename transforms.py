import torch
import torch.nn.functional as F
import numpy as np
import numbers


class DownSampling(object):
    def __init__(self, factor=4):
        self.factor = factor

    def __call__(self, sample):
        img, gt = sample
        factor = self.factor
        return img[:, ::factor, ::factor], gt[::factor, ::factor]


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, gt = sample

        theta = torch.Tensor([
            [1, 0, 0],
            [0, -1 if np.random.rand() < self.p else 1, 0]])

        def unsqueeze(x):
            return torch.unsqueeze(x, 0)

        grid = F.affine_grid(unsqueeze(theta), unsqueeze(img).shape)
        img = F.grid_sample(unsqueeze(img), grid, mode='nearest')[0]
        gt = F.grid_sample(unsqueeze(gt), grid, mode='nearest')[0]

        return img, gt


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, gt = sample
        theta = torch.Tensor([
            [-1 if np.random.rand() < self.p else 1, 0, 0],
            [0, 1, 0]])

        def unsqueeze(x):
            return torch.unsqueeze(x, 0)

        grid = F.affine_grid(unsqueeze(theta), unsqueeze(img).shape)
        img = F.grid_sample(unsqueeze(img), grid, mode='nearest')[0]
        gt = F.grid_sample(unsqueeze(gt), grid, mode='nearest')[0]

        return img, gt


class RandomDepthFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, gt = sample
        theta = torch.Tensor([
            [-1 if np.random.rand() < self.p else 1, 0, 0],
            [0, 1, 0]])

        def unsqueeze(x):
            return torch.unsqueeze(x, 0)

        grid = F.affine_grid(unsqueeze(theta), unsqueeze(img).shape)
        img = F.grid_sample(unsqueeze(img), grid, mode='nearest')[0]
        gt = F.grid_sample(unsqueeze(gt), grid, mode='nearest')[0]

        return img, gt


class RandomPixelTranslation(object):
    def __init__(self, translate):
        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
        self.translate = translate

    def _sample_translation(self):
        max_dx = self.translate[0]
        max_dy = self.translate[1]
        translation = (np.round(np.random.uniform(-max_dx, max_dx)),
                       np.round(np.random.uniform(-max_dy, max_dy)))
        return translation

    def __call__(self, sample):
        img, gt = sample

        translation = self._sample_translation()
        theta = torch.Tensor([
            [1, 0, translation[0] / float(img.shape[1])],
            [0, 1, translation[1] / float(img.shape[2])]])

        def unsqueeze(x):
            return torch.unsqueeze(x, 0)

        grid = F.affine_grid(unsqueeze(theta), unsqueeze(img).shape)
        img = F.grid_sample(unsqueeze(img), grid, padding_mode='reflection', mode='nearest')[0]
        gt = F.grid_sample(unsqueeze(gt), grid, padding_mode='reflection', mode='nearest')[0]

        return img, gt


class RandomRotation(object):
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def _sample_angle(self):
        return np.random.rand() * (self.degrees[1] - self.degrees[0]) + self.degrees[0]

    def __call__(self, sample):
        img, gt = sample

        angle = self._sample_angle() * np.pi / 180.
        theta = torch.Tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0]])

        def unsqueeze(x):
            return torch.unsqueeze(x, 0)

        grid = F.affine_grid(unsqueeze(theta), unsqueeze(img).shape)
        img = F.grid_sample(unsqueeze(img), grid, padding_mode='border')[0]
        gt = F.grid_sample(unsqueeze(gt), grid, padding_mode='border')[0]
        gt[gt < 0.5] = 0
        gt[gt >= 0.5] = 1

        return img, gt


class RandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, flip=None, shear=None, mode='bilinear'):
        self.inter_mode = mode
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translates = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scales = scale

        if flip is not None:
            assert isinstance(flip, (tuple, list)) and len(flip) == 2, \
                "flip should be a list or tuple and it must be of length 2."
            assert isinstance(flip[0], bool) and isinstance(flip[1], bool), \
                "flip elements must be booleans"
        self.flips = flip

        if shear is not None:
            raise NotImplementedError()
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shears = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shears = shear
        else:
            self.shears = shear

    def _sample_values(self):
        angle = np.random.uniform(self.degrees[0], self.degrees[1]) * np.pi / 180.
        if self.translates is not None:
            max_dx = self.translates[0]
            max_dy = self.translates[1]
            translation = (np.round(np.random.uniform(-max_dx, max_dx)),
                           np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translation = (0, 0)
        if self.scales is not None:
            scale = np.random.uniform(self.scales[0], self.scales[1])
        else:
            scale = 1.0
        if self.flips is not None:
            should_flipx = self.flips[0]
            should_flipy = self.flips[1]
            flip = (np.random.choice([-1, 1] if should_flipx else 1),
                    np.random.choice([-1, 1] if should_flipy else 1))
        if self.shears is not None:
            shear = np.random.uniform(self.shears[0], self.shears[1])
        else:
            shear = 0.0

        return angle, translation, scale, flip, shear

    def __call__(self, sample):
        img, gt = sample

        angle, translation, scale, flip, _ = self._sample_values()
        matR = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]], dtype=torch.float32)
        matT = torch.tensor([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]], dtype=torch.float32)
        matS = torch.tensor([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]], dtype=torch.float32)
        matF = torch.tensor([
            [flip[0], 0, 0],
            [0, flip[1], 0],
            [0, 0, 1]], dtype=torch.float32)

        transMat = matT.mm(matF.mm(matS.mm(matR)))
        theta = transMat[:2]

        def unsqueeze(x):
            return torch.unsqueeze(x, 0)

        grid = F.affine_grid(unsqueeze(theta), unsqueeze(img).shape)
        img = F.grid_sample(unsqueeze(img), grid, padding_mode='border', mode=self.inter_mode)[0]
        gt = F.grid_sample(unsqueeze(gt), grid, padding_mode='border', mode=self.inter_mode)[0]

        if self.inter_mode != 'nearest':
            gt[gt < 0.5] = 0
            gt[gt >= 0.5] = 1

        return img, gt


class ToTensorTwoTargetClasses(object):
    def __call__(self, sample):
        img, gt = sample

        gt = np.array(gt, dtype=np.float32)
        img = torch.from_numpy(np.expand_dims(img, axis=0))
        gt = torch.from_numpy(gt)
        return img, gt
