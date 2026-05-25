import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.optim as optim
import torchvision.utils as vutils
import time
import random

from sklearn.model_selection import KFold, train_test_split
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary

from datasets.lc import LCSet
from networks.unet import UNet3D
#from networks.attention_unet import AttentionUnet
from networks.ResNet import ResnetGenerator, get_norm_layer
from loss_functions import dice_loss, focal_tversky_loss

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
def train(train_indices, val_indices, fold=0, fold2=0, wdh=0):
    print((train_indices, val_indices, fold, fold2))
    # params
    model_name = train_params['new_model_name']
    model_name = model_name + 'f' + str(fold) + '_f2-' + str(fold2) + '_wdh' + str(wdh) if wdh is not 0 else model_name + 'f' + str(fold) + '_f2-' + str(fold2)
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

    train_set = LCSet(train_indices, augmentation_params, data_dir=train_params['tune_data_dir'], validation=False)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_set = LCSet(val_indices, augmentation_params, data_dir=train_params['tune_data_dir'], validation=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    print('data loaded')

    if train_params['network'] == 'unet':
        net = UNet3D(in_channels=1, n_classes=2, depth=4, wf=5, padding=True, batch_norm=True)
    #elif train_params['network'] == 'attention-unet':
        #net = AttentionUnet(in_channels=1, out_channels=2, is_leaky=True)
    elif train_params['network'] == 'resnet':
        net = ResnetGenerator(input_nc=1, output_nc=2, ngf=64, norm_layer=get_norm_layer('batch'), use_dropout=False,
                              n_blocks=9, padding_type='zero')
    net.cuda()
    checkpoint = torch.load(train_params['start_model'])
    # checkpoint = torch.load(model_dir + model_name + '/checkpoint_' + model_type)
    net.load_state_dict(checkpoint['model_state_dict'])

    summary(net, input_size=(1, augmentation_params['patch_size'],
                             augmentation_params['patch_size'],
                             augmentation_params['patch_size']))

    if train_params['loss'] == 'dsc':
        forward = forward_default
        criterion = dice_loss
    elif train_params['loss'] == 'focal_tversky':
        forward = forward_attention_unet
        criterion = focal_tversky_loss
    optimizer = optim.Adam(net.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
                # print(inputs.size())
                # print(targets.size())
                inputs = inputs.cuda()
                targets = targets.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                try:
                    with torch.autograd.detect_anomaly():
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
    'start_model': '/pool/max/train_logs/lc/LC_03_f0/checkpoint_latest',
    'new_model_name': 'LC_03_AD-MCI-tuned_chilled_',
    'network': 'unet',
    'loss': 'dsc',
    'epochs': 250,
    'log_dir': 'logs/lc/',
    'rep_per_epoch': 10,
    'train_repetitions': 2,
    'num_kfolds': 2,
    'tune_data_dir': 'data/lc_ad-mci_processed/'
}


if __name__ == '__main__':
    num_wdh = train_params['train_repetitions']
    num_samples = 20
    indices = range(num_samples)
    print(indices)

    with open('subsets_ad-mci.csv', 'w') as f:
        f.write('w,f1,f2,train,val,test\n')
        for w in range(num_wdh):
            kf = KFold(n_splits=train_params['num_kfolds'], shuffle=True, random_state=131294 + w)
            for counter, (train_val_indices, test_indices) in enumerate(kf.split(indices)):
                train_val_indices = [indices[i] for i in train_val_indices]
                test_indices = [indices[i] for i in test_indices]
                for counter2, (train_indices, val_indices) in enumerate(kf.split(train_val_indices)):
                    train_indices = [train_val_indices[i] for i in train_indices]
                    val_indices = [train_val_indices[i] for i in val_indices]
                    f.write(str(w) + ',' + str(counter) + ',' + str(counter2) + ',' + str(train_indices) + ',' + str(val_indices) + ',' + str(test_indices) + '\n')
                    train(train_indices, val_indices, fold=counter, fold2=counter2, wdh=w)
