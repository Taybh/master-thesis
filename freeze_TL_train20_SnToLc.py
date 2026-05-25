import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.optim as optim
import torchvision.utils as vutils
import time
import random
import numpy as np

from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary
# from metrics import iou_score
from collections import OrderedDict
# import sys
# sys.path.insert(0, '/home/tayebeh/PycharmProjects/LC/LCSN-Seg-f4f50f3/datasets/')
from datasets.SN import SNSet
from datasets.LC_1rater import LC1Rater_Set
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
# from networks.Modified_3DUnet import Modified3DUNet
# from util import init_weights, count_param
# from layers import unetConv3, unetUp

# from networks import UnetPlusPlus
# from networks.attentionUnet import AttU_Net

from networks.unet import UNet3D

# from networks.attention_unet import AttentionUnet
from networks.ResNet import ResnetGenerator, get_norm_layer
from loss_functions import dice_loss, tversky_loss, focal_tversky_loss, two_class_balanced_cross_entropy
from random import sample

# from loss_PlusPlus import BCEDiceLoss

torch.backends.cudnn.benchmark = True


def forward_default(net_ft, inputs, targets, criterion):
    outputs = net_ft(inputs)
    return criterion(outputs, targets), outputs


def forward_attention_unet(net, inputs, targets, criterion):
    outputs = net(inputs)
    loss = (criterion(torch.sigmoid(outputs[0]), targets[:, :, ::8, ::8, ::8])
            + criterion(torch.sigmoid(outputs[1]), targets[:, :, ::4, ::4, ::4])
            + criterion(torch.sigmoid(outputs[2]), targets[:, :, ::2, ::2, ::2])
            + criterion(torch.sigmoid(outputs[3]), targets)) / 4.
    return loss, outputs


# TODO: check dtypes of img and gt in all stages!
def train(train_indices, val_indices, fold=0, wdh=0, pattern='', lr_type='', origin=''):
    if origin == 'LcToSn':
        org = SNSet
        model_dir = "LC_weights/"
        data_name = "unet_LC_Scratch"
        print('transfer learning from LC to SN')
        checkpoint = torch.load(model_dir + data_name + 'f' + str(counter) + '_wdh1' + '/checkpoint_' + model_type)
    else:
        org = LC1Rater_Set
        model_dir = "SN_weights/"
        data_name = "unet_SN_Scratch"
        print('transfer learning from SN to LC')
        checkpoint = torch.load(model_dir + data_name + 'f' + str(counter) + '_wdh1' + '/checkpoint_' + model_type)
##################################
######### define the network######
##################################
    net_ft = UNet3D(in_channels=1, n_classes=2, depth=4, wf=5, padding=True, batch_norm=True)
    net_ft.cuda()
    # checkpoint = torch.load(model_dir + model_name + '/checkpoint_' + model_type)
    net_ft.load_state_dict(checkpoint['model_state_dict'])

    for name, child in net_ft.named_children():
        print(name)
    print(net_ft)

####################################
############ conditions ############
####################################

    if lr_type == 'decay':
        if pattern == 'veryLast':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters(), "lr":train_params['lr']}
                ],
                lr=train_params['lr_freeze']
            )
            print('very last layer is tuned & decay lr for the rest')

        elif pattern == 'last':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters(), "lr":train_params['lr']},
                    {"params": net_ft.veryLast.parameters(), "lr":train_params['lr']}
                ],
                lr=train_params['lr_freeze']
            )
            print(' last layer is tuned & decay lr for the rest')

        elif pattern == 'up2':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.last.parameters(), "lr":train_params['lr']},
                    {"params": net_ft.veryLast.parameters(), "lr":train_params['lr']}
                ],
                lr=train_params['lr_freeze']
            )
            print('up2 layer is tuned & decay lr for the rest')

        elif pattern == 'up1':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[2].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.last.parameters(), "lr":train_params['lr']},
                    {"params": net_ft.veryLast.parameters(), "lr":train_params['lr']}
                ],
                lr=train_params['lr_freeze']
            )
            print('up1 layer is tuned & decay lr for the rest')

        elif pattern == 'up0':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[1].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[2].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.last.parameters(), "lr":train_params['lr']},
                    {"params": net_ft.veryLast.parameters(), "lr":train_params['lr']}
                ],
                lr=train_params['lr_freeze']
            )
            print('up0 layer is tuned & decay lr for the rest')

        elif pattern == 'down3':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[0].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[1].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[2].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.last.parameters(), "lr":train_params['lr']},
                    {"params": net_ft.veryLast.parameters(), "lr":train_params['lr']}
                ],
                lr=train_params['lr_freeze']
            )
            print('down3 layer is tuned & decay lr for the rest')

        elif pattern == 'down2':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.down_path[3].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[0].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[1].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[2].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.last.parameters(), "lr":train_params['lr']},
                    {"params": net_ft.veryLast.parameters(), "lr":train_params['lr']}
                ],
                lr=train_params['lr_freeze']
            )
            print('down2 layer is tuned & decay lr for the rest')

        elif pattern == 'down1':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.down_path[2].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.down_path[3].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[0].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[1].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.up_path[2].parameters(), "lr":train_params['lr']},
                    {"params": net_ft.last.parameters(), "lr":train_params['lr']},
                    {"params": net_ft.veryLast.parameters(), "lr":train_params['lr']}
                ],
                lr=train_params['lr_freeze']
            )
            print('down1 layer is tuned & decay lr for the rest')

        else: #whole
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters()}
                ],
                lr=train_params['lr']
            )
            print('all layers is tuned & decay lr for the rest')

    else:

        if pattern == 'veryLast':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.veryLast.parameters()}
                ],
                lr=train_params['lr']
            )
            print('very last layer is tuned & rest if freezed')

        elif pattern == 'last':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters()}
                ],
                lr=train_params['lr']
            )
            print('last layer is tuned & rest if freezed')

        elif pattern == 'up2':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters()}
                ],
                lr=train_params['lr']
            )
            print('up2 layer is tuned & rest if freezed')

        elif pattern == 'up1':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters()}
                ],
                lr=train_params['lr']
            )
            print('up1 layer is tuned & rest if freezed')

        elif pattern == 'up0':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters()}
                ],
                lr=train_params['lr']
            )
            print('up0 layer is tuned & rest if freezed')

        elif pattern == 'down3':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters()},
                ],
                lr=train_params['lr']
            )
            print('down3 layer is tuned & rest if freezed')

        elif pattern == 'down2':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters()}
                ],
                lr=train_params['lr']
            )
            print('down2 layer is tuned & rest if freezed')

        elif pattern == 'down1':
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters()}
                ],
                lr=train_params['lr']
            )
            print('down1 layer is tuned & rest if freezed')

        else: #whole
            optimizer_ft = optim.Adam(
                [
                    {"params": net_ft.down_path[0].parameters()},
                    {"params": net_ft.down_path[1].parameters()},
                    {"params": net_ft.down_path[2].parameters()},
                    {"params": net_ft.down_path[3].parameters()},
                    {"params": net_ft.up_path[0].parameters()},
                    {"params": net_ft.up_path[1].parameters()},
                    {"params": net_ft.up_path[2].parameters()},
                    {"params": net_ft.last.parameters()},
                    {"params": net_ft.veryLast.parameters()}
                ],
                lr=train_params['lr']
            )
            print('all layers is tuned & rest if freezed')

    #  for parameter in net_ft.parameters():
    #     print(parameter)

    print((train_indices, val_indices, fold))
    scaler = GradScaler()
    ######################
    ######data prep########
    #######################

    #model_name = train_params['model_name']
    #model_name = model_name + 'f' + str(fold) + '_wdh' + str(wdh) if wdh is not 0 else model_name + 'f' + str(fold)
    model_name = origin +'/'+ lr_type + '/' + pattern + '/' + 'f' + str(fold) + '_wdh' + str(wdh) if wdh is not 0 else origin +'/'+ lr_type + '/' + pattern + '/' + 'f' + str(fold)
    log_dir = train_params['log_dir']
    num_epochs = train_params['epochs']
    num_repetitions = train_params['rep_per_epoch']

    writer = SummaryWriter(log_dir=log_dir + model_name)

    augmentation_params = {'patch_size': 64,
                           'fg_rate': 0.5,
                           'flip_horizontal': 0.5,
                           'flip_vertical': 0.5,
                           'flip_depth': 0.5}
    ps2 = int(augmentation_params['patch_size'] / 2)

    train_set = org(train_indices, augmentation_params, data_file=train_params['data_dir'], validation=False)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_set = org(val_indices, augmentation_params, data_file=train_params['data_dir'], validation=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    print('data loaded')

    # optimizer = optim.Adam(net_ft.parameters())
    # net = torch.nn.DataParallel(net).cuda()
    # summary(net, input_size=(1, augmentation_params['patch_size'],
    #                        augmentation_params['patch_size'],
    #                       augmentation_params['patch_size']))

    if train_params['loss'] == 'dsc':
        forward = forward_default
        criterion = dice_loss
    # elif train_params['loss'] == 'loss_PlusPlus':
    ##forward = forward_default
    # criterion = BCEDiceLoss
    elif train_params['loss'] == 'tversky':
        forward = forward_default
        criterion = tversky_loss
    elif train_params['loss'] == 'focal_tversky':
        forward = forward_attention_unet
        criterion = focal_tversky_loss
    elif train_params['loss'] == 'ce':
        forward = forward_default
        criterion = two_class_balanced_cross_entropy

   # def count_parameters(model):
     #   return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(count_parameters(net_ft))

    # print(net_ft)
    ##print(net.parameters)

    # for parameter in net.parameters():
    #    print(len(parameter))

    best_val_loss = 999999999
    num_mini_batches = len(train_loader)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for r in range(num_repetitions):
            net_ft.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                print(i)
                # get the inputs
                inputs, targets = data
                # print(inputs.size())
                # print(targets.size())
                inputs = inputs.cuda()
                targets = targets.cuda()

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                #try:
                    #with torch.autograd.detect_anomaly():
                '''
                if train_params['network'] == 'UNetPlusPlus':
                    outputs = net_ft(inputs)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, targets)
                    loss /= len(outputs)
                else:
                '''
                # forward
                with autocast():
                    loss, outputs = forward(net_ft, inputs, targets, criterion)
                # backward + optimize
                scaler.scale(loss).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer_ft)

                # Updates the scale for next iteration.
                scaler.update()

                #loss.backward()
                # torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                #optimizer_ft.step()


                running_loss += loss.item()
                epoch_loss += loss.item()

            # print statistics
            # writer.add_scalar('loss/train', running_loss / num_mini_batches, (epoch * num_repetitions) + r)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, r + 1, running_loss / num_mini_batches))

        # calculate training loss
        writer.add_scalar('loss/train', epoch_loss / num_mini_batches / num_repetitions, epoch)

        # calculate validation loss
        net_ft.eval()
        mean_val_loss = 0
        with torch.no_grad():
            for r in range(num_repetitions):
                for j, val_data in enumerate(val_loader):
                    val_inputs, val_targets = val_data
                    val_inputs = val_inputs.cuda()
                    val_targets = val_targets.cuda()

                    if train_params['network'] == 'UNetPlusPlus':
                        val_outputs = net_ft(val_inputs)
                        val_loss = 0
                        for output in val_outputs:
                            val_loss += criterion(output, val_targets)
                        val_loss /= len(val_outputs)
                        # iou = iou_score(outputs[-1], val_targets)
                    else:
                        # val_loss = criterion(val_outputs, val_targets)
                        val_loss, val_outputs = forward(net_ft, val_inputs, val_targets, criterion)
                    mean_val_loss += val_loss.item()
            mean_val_loss /= num_repetitions
            writer.add_scalar('loss/val', mean_val_loss, epoch)

        # save model if it has a better validation loss than all before
        if mean_val_loss < best_val_loss:
            torch.save({'epoch': epoch,
                        'model_state_dict': net_ft.state_dict(),
                        'optimizer_state_dict': optimizer_ft.state_dict(),
                        'loss': loss}, log_dir + model_name + '/checkpoint_best')
            best_val_loss = mean_val_loss
        # save latest model
        torch.save({'epoch': epoch,
                    'model_state_dict': net_ft.state_dict(),
                    'optimizer_state_dict': optimizer_ft.state_dict(),
                    'loss': loss}, log_dir + model_name + '/checkpoint_latest')

        # write images
        writer.add_image('train/input', vutils.make_grid(inputs[0, 0, ..., ps2], normalize=True, scale_each=True),
                         epoch)
        writer.add_image('train/output', vutils.make_grid(outputs[0, 0, ..., ps2], normalize=True, scale_each=True),
                         epoch)
        writer.add_image('train/target', vutils.make_grid(targets[0, 0, ..., ps2], normalize=True, scale_each=True),
                         epoch)
        writer.add_image('val/input', vutils.make_grid(val_inputs[0, 0, ..., ps2], normalize=True, scale_each=True),
                         epoch)
        writer.add_image('val/output',
                         vutils.make_grid(val_outputs[0, 0, ..., ps2], normalize=True, scale_each=True),
                         epoch)
        writer.add_image('val/target', vutils.make_grid(val_targets[0, 0, ..., ps2], normalize=True, scale_each=True),
                         epoch)
    writer.close()

    print('finished training fold #' + str(fold))


train_params = {
    'rnd_seed': 131294,
    #'model_name': 'unet_SN_LC',
    'network': 'unet',
    'loss': 'dsc',
    'epochs': 1000,
    'log_dir': 'logs/unet20/',
    'rep_per_epoch': 10,
    'train_repetitions': 1,
    'num_kfolds': 5,
    'data_dir': '../data/SN-LC-3R-data.h5',
    'num_samples': 81,
    'lr': 0.001,
    'lr_freeze': 0.0001
}

# reproducibility stuff
random.seed(train_params['rnd_seed'])
np.random.seed(train_params['rnd_seed'])
torch.manual_seed(train_params['rnd_seed'])
torch.cuda.manual_seed(train_params['rnd_seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    pattern_total = ['down2', 'down1','whole', 'down3'] #'last','up1' ,'whole','up0','down3','up2' , 'down2', 'down1'
    lr_type_total = ['freeze'] #,  'decay'
    origin_total = ['SnToLc'] #,'LcToSn'
    model_type = "best"
    num_wdh = train_params['train_repetitions']
    indices_total = range(train_params['num_samples'])
    dataset_num = [20]

    train_val_81, test_81 = train_test_split(indices_total, test_size=0.27, shuffle=True,
                                           random_state=train_params['rnd_seed'])
    print(('test', test_81))
    print('train_val', train_val_81)

    #indices_40 = sample(train_val_81, 40)
    #train_val_40, test_40 = train_test_split(indices_40, test_size=0.27, shuffle=True,
     #                                        random_state=train_params['rnd_seed'])

    indices_30 = sample(train_val_81, 30)
    train_val_30, test_30 = train_test_split(indices_30, test_size=0.27, shuffle=True,
                                             random_state=train_params['rnd_seed'])

    indices_20 = sample(train_val_30, 20)
    train_val_20, test_20 = train_test_split(indices_20, test_size=0.27, shuffle=True,
                                             random_state=train_params['rnd_seed'])

    indices_10 = sample(train_val_20, 10)
    train_val_10, test_10 = train_test_split(indices_10, test_size=0.27, shuffle=True,
                                             random_state=train_params['rnd_seed'])
    ##

    kf = KFold(n_splits=train_params['num_kfolds'], shuffle=True, random_state=train_params['rnd_seed'])

    train_val_s = train_val_20


    for j, org in enumerate(origin_total):
        for k, lr_typ in enumerate(lr_type_total):
            for i, patt in enumerate(pattern_total):
                print('patt: '+ str(patt)+ '  lr_type: '+ str(lr_typ) + '  origin: ' + str(org))
                for w in range(train_params['train_repetitions']):
                    for counter, (train_indices, val_indices) in enumerate(kf.split(train_val_s)):
                        train_set = [train_val_s[i] for i in train_indices]
                        val_set = [train_val_s[i] for i in val_indices]
                        train(train_set, val_set, fold=counter, wdh=num_wdh, pattern=patt, lr_type=lr_typ, origin=org)





#logs/unet20/SnToLc/decay/whole/f2_wdh1/checkpoint_latest
#logs/unet20/SnToLc/decay/up0/f2_wdh1/checkpoint_latest
#logs/unet20/SnToLc/decay/down3/f2_wdh1/checkpoint_latest
#logs/unet20/SnToLc/decay/down3/f0_wdh1/checkpoint_latest
#logs/unet20/SnToLc/decay/up2/f3_wdh1/checkpoint_latest

'''
        train_length = int(0.8 * len(indices))
        print(train_length)
        val_length = int(0.3 * train_length)
        test_length = len(indices) - train_length
        train_indices, test_indices = torch.utils.data.random_split(indices, [train_length, test_length])
        print('train indices:' + str(list(train_indices)))
        print('test indices:' + str(list(test_indices)))
        train_indices_tmp, val_indices = torch.utils.data.random_split(train_indices, [train_length-val_length, val_length])
       # print("f:" + str(counter) + "train_indices:" + str( train_indices)  + "val_indices:" + str(val_indices))
        print('train indices:' +str(list(train_indices)) + '\n' + 'val indices:' + str(list(val_indices))+ '\n' + 'test indices:' + str(list(test_indices)))
        train(train_indices, val_indices, wdh=w, fold=1)

'''

