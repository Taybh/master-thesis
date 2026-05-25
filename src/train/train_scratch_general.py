import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.optim as optim
import torchvision.utils as vutils
import time
import random
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary
#from metrics import iou_score
from collections import OrderedDict
#import sys
#sys.path.insert(0, '/home/tayebeh/PycharmProjects/LC/LCSN-Seg-f4f50f3/datasets/')
from datasets.LC_1rater import LC1Rater_Set
from datasets.SN import SNSet
#from networks.Modified_3DUnet import Modified3DUNet
#from util import init_weights, count_param
#from layers import unetConv3, unetUp

#from networks import UnetPlusPlus
#from networks.attentionUnet import AttU_Net

from networks.unet import UNet3D
from random import sample
#from networks.attention_unet import AttentionUnet
from networks.ResNet import ResnetGenerator, get_norm_layer
from loss_functions import dice_loss, tversky_loss, focal_tversky_loss, two_class_balanced_cross_entropy

#from loss_PlusPlus import BCEDiceLoss

torch.backends.cudnn.benchmark = True


def forward_default(net, inputs, targets, criterion):
    outputs = net(inputs)
    return criterion(outputs, targets), outputs

def forward_attention_unet(net, inputs, targets, criterion):
    outputs = net(inputs)
    loss = (criterion(torch.sigmoid(outputs[0]), targets[:, :, ::8, ::8, ::8])
            + criterion(torch.sigmoid(outputs[1]), targets[:, :, ::4, ::4, ::4])
            + criterion(torch.sigmoid(outputs[2]), targets[:, :, ::2, ::2, ::2])
            + criterion(torch.sigmoid(outputs[3]), targets)) / 4.
    return loss, outputs


# TODO: check dtypes of img and gt in all stages!
def train(train_indices, val_indices, origin_data,data_num, fold=0, wdh=0):
    print((train_indices, val_indices, fold))
    # params
    #model_name = train_params['model_name']
    model_name = 'unet_' + origin_data + '_Scratch'
    model_name = model_name + 'f' + str(fold) + '_wdh' + str(wdh) if wdh is not 0 else model_name + 'f' + str(fold)
    #log_dir = train_params['log_dir']
    log_dir = 'logs/scratch' + str(data_num) + '/' + origin_data + '/'
    num_epochs = train_params['epochs']
    num_repetitions = train_params['rep_per_epoch']

    writer = SummaryWriter(log_dir=log_dir + model_name)

    augmentation_params = {'patch_size': 64,
                           'fg_rate': 0.5,
                           'flip_horizontal': 0.5,
                           'flip_vertical': 0.5,
                           'flip_depth': 0.5}
    ps2 = int(augmentation_params['patch_size'] / 2)
    if origin_data =='lc':
        train_set = LC1Rater_Set(train_indices, augmentation_params, data_file=train_params['data_dir'], validation=False)

        val_set = LC1Rater_Set(val_indices, augmentation_params, data_file=train_params['data_dir'], validation=True)
    else:
        train_set = SNSet(train_indices, augmentation_params, data_file=train_params['data_dir'],
                                 validation=False)

        val_set = SNSet(val_indices, augmentation_params, data_file=train_params['data_dir'], validation=True)



    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    print('data loaded')

    if train_params['network'] == 'unet':
        net = UNet3D(in_channels=1, n_classes=2, depth=4, wf=5, padding=True, batch_norm=True)
    #elif train_params['network'] == 'attention-unet':
        #net = AttentionUnet(in_channels=1, out_channels=2, is_leaky=True)
    #elif train_params['network'] == 'UNetPlusPlus':
     #   net = UnetPlusPlus.NestedUNet()
    #elif train_params['network'] == 'attentionUnet':
    #    net = AttU_Net(img_ch=1, output_ch=2)
    #elif train_params['network'] == 'resnet':
     #   net = ResnetGenerator(input_nc=1, output_nc=2, ngf=64, norm_layer=get_norm_layer('batch'), use_dropout=False,
      #                        n_blocks=9, padding_type='zero')
    net.cuda()
    #net = torch.nn.DataParallel(net).cuda()
    #summary(net, input_size=(1, augmentation_params['patch_size'],
     #                        augmentation_params['patch_size'],
      #                       augmentation_params['patch_size']))

    if train_params['loss'] == 'dsc':
        forward = forward_default
        criterion = dice_loss
    #elif train_params['loss'] == 'loss_PlusPlus':
        #forward = forward_default
        #criterion = BCEDiceLoss
    elif train_params['loss'] == 'tversky':
        forward = forward_default
        criterion = tversky_loss
    elif train_params['loss'] == 'focal_tversky':
        forward = forward_attention_unet
        criterion = focal_tversky_loss
    elif train_params['loss'] == 'ce':
        forward = forward_default
        criterion = two_class_balanced_cross_entropy

    optimizer = optim.Adam(net.parameters())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(net))

    print(net)
    #print(net.parameters)

    #for parameter in net.parameters():
    #    print(len(parameter))


    best_val_loss = 999999999
    num_mini_batches = len(train_loader)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for r in range(num_repetitions):
            net.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                print(i)
                # get the inputs
                inputs, targets = data
                #print(inputs.size())
                #print(targets.size())
                inputs = inputs.cuda()
                targets = targets.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                try:
                    with torch.autograd.detect_anomaly():
                        '''
                        if train_params['network'] == 'UNetPlusPlus':
                            outputs = net(inputs)
                            loss = 0
                            for output in outputs:
                                loss += criterion(output, targets)
                            loss /= len(outputs)
                        else:
                        '''
                        # forward
                        loss, outputs = forward(net, inputs, targets, criterion)

                        # backward + optimize
                        loss.backward()
                        #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                        optimizer.step()
                except Exception as e:
                    print(e)
                    torch.save(outputs, 'outputs.pt')
                    torch.save(inputs, 'inputs.pt')
                    torch.save(targets, 'targets.pt')
                    print(loss)
                    return

                running_loss += loss.item()
                epoch_loss += loss.item()

            # print statistics
            # writer.add_scalar('loss/train', running_loss / num_mini_batches, (epoch * num_repetitions) + r)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, r + 1, running_loss / num_mini_batches))

        # calculate training loss
        writer.add_scalar('loss/train', epoch_loss / num_mini_batches / num_repetitions, epoch)

        # calculate validation loss
        net.eval()
        mean_val_loss = 0
        with torch.no_grad():
            for r in range(num_repetitions):
                for j, val_data in enumerate(val_loader):
                    val_inputs, val_targets = val_data
                    val_inputs = val_inputs.cuda()
                    val_targets = val_targets.cuda()

                    if train_params['network'] == 'UNetPlusPlus':
                        val_outputs = net(val_inputs)
                        val_loss = 0
                        for output in val_outputs:
                            val_loss += criterion(output, val_targets)
                        val_loss /= len(val_outputs)
                        #iou = iou_score(outputs[-1], val_targets)
                    else:
                        # val_loss = criterion(val_outputs, val_targets)
                        val_loss, val_outputs = forward(net, val_inputs, val_targets, criterion)
                    mean_val_loss += val_loss.item()
            mean_val_loss /= num_repetitions
            writer.add_scalar('loss/val', mean_val_loss, epoch)

        # save model if it has a better validation loss than all before
        if mean_val_loss < best_val_loss:
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, log_dir + model_name + '/checkpoint_best')
            best_val_loss = mean_val_loss
        # save latest model
        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, log_dir + model_name + '/checkpoint_latest')

        # write images
        '''
        if train_params['network'] == 'UNetPlusPlus':
            writer.add_image('train/input', vutils.make_grid(inputs[0, 0, ..., ps2], normalize=True, scale_each=True),
                             epoch)
            writer.add_image('train/output', vutils.make_grid(outputs[3][0, 0, ..., ps2], normalize=True, scale_each=True),
                             epoch)
            writer.add_image('train/target', vutils.make_grid(targets[0, 0, ..., ps2], normalize=True, scale_each=True),
                             epoch)
            writer.add_image('val/input', vutils.make_grid(val_inputs[0, 0, ..., ps2], normalize=True, scale_each=True),
                             epoch)
            writer.add_image('val/output',
                             vutils.make_grid(val_outputs[3][0, 0, ..., ps2], normalize=True, scale_each=True),
                             epoch)
            writer.add_image('val/target',
                             vutils.make_grid(val_targets[0, 0, ..., ps2], normalize=True, scale_each=True),
                             epoch)

        else:
        '''
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
    #'model_name': 'unet_LC_Scratch',
    'network': 'unet',
    'loss': 'dsc',
    'epochs': 250,
   # 'log_dir': 'logs/scratch22/lc/',
    'rep_per_epoch': 10,
    'train_repetitions': 1,
    'num_kfolds': 5,
    'data_dir': '../data/SN-LC-3R-data.h5',
    'num_samples': 81
}

# reproducibility stuff
random.seed(train_params['rnd_seed'])
np.random.seed(train_params['rnd_seed'])
torch.manual_seed(train_params['rnd_seed'])
torch.cuda.manual_seed(train_params['rnd_seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    origin = ['lc','sn']
    dataset_num = [22,42,62]
    num_wdh = train_params['train_repetitions']
    indices_total = range(train_params['num_samples'])
    for org in origin:
        for num in dataset_num:
            indices = sample(indices_total, num)
            print(indices)

            kf = KFold(n_splits=train_params['num_kfolds'], shuffle=True, random_state=train_params['rnd_seed'])
            train_val_s, test_s = train_test_split(indices, test_size=0.27, shuffle=True,
                                                   random_state=train_params['rnd_seed'])
            print(('test', test_s))
            for w in range(train_params['train_repetitions']):
                for counter, (train_indices, val_indices) in enumerate(kf.split(train_val_s)):
                    train_set = [train_val_s[i] for i in train_indices]
                    val_set = [train_val_s[i] for i in val_indices]
                    train(train_set, val_set, origin_data=org,data_num=num, fold=counter, wdh=num_wdh)


'''

    #kf = KFold(n_splits=train_params['num_kfolds'], shuffle=True, random_state=131294)
    for w in range(num_wdh):
        #for counter, (train_indices, val_indices) in enumerate(kf.split(indices)):

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


first_fold = list(kf.split(indices))[0]
    train_indices = first_fold[0]
    val_indices = first_fold[1]
    train(train_indices, val_indices, 0)'''
