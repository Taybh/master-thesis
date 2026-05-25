import torch
from torch.nn import CrossEntropyLoss


def dice_loss(prediction, target):
    smooth = 1.
    pflat = prediction.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (pflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (pflat.sum() + tflat.sum() + smooth))


def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=4. / 3.):
    smooth = 1.

    nclasses = pred.shape[1]
    ftl = 0.
    for c in range(nclasses):
        pflat = pred[:, c].contiguous().view(-1)
        gflat = target[:, c].contiguous().view(-1)

        intersection = (pflat * gflat).sum()
        non_p_g = ((1. - pflat) * gflat).sum()
        p_non_g = (pflat * (1. - gflat)).sum()

        ti = (intersection + smooth) / (intersection + alpha * non_p_g + beta * p_non_g + smooth)
        ftl += (1. - ti) ** (1. / gamma)
    return ftl / nclasses


# Tversky Loss: alpha = 0.5 -> Dice, alpha = beta = 1 -> Tanimoto/Jaccard
def tversky_loss(pred, target, alpha=0.7):
    smooth = 1.
    beta = 1. - alpha

    target = target.view(-1)
    pred = pred.view(-1)

    xy = torch.sum(pred * target)
    xmy = torch.sum(pred * (1. - target))
    ymx = torch.sum((1. - pred) * target)

    return 1 - ((xy + smooth) / (xy + alpha * xmy + beta * ymx + smooth))


def two_class_balanced_cross_entropy(pred, target):
    target = target.view(-1)
    pred = pred.view(-1)

    bg_weight = torch.sum(target) / target.size()[0]
    fg_weight = 1. - bg_weight
    weights = torch.tensor([bg_weight, fg_weight]).cuda()

    pred = torch.stack((1. - pred, pred), -1)

    return CrossEntropyLoss(weight=weights)(pred, target.long())
