import torch
import glob


for file in list(glob.glob('logs/**/checkpoint_latest', recursive=True)):

    checkpoint = torch.load(file)
    if checkpoint['epoch'] == 250:
        pass
    elif checkpoint['epoch'] == 1000:
        pass
    else:
        print(checkpoint['epoch'])
        print(file)

#'logs/unet10/LcToSn/decay/down1/f1_wdh1/checkpoint_latest'