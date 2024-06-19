# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:15:35 2019
Uses a trained model to return the probability matrix of an entire image
@author: SCSC
"""

import torch
import time
from torch import nn
from MyLoader import OrigDataset as XDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
from MedMamba import VSSM as medmamba
from utils import inference
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    # Define the network
    model = medmamba(num_classes=2)
    model.load_state_dict(torch.load('checkpoint_32.pth')['state_dict'])
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    # Define the dataset
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    test_dset = XDataset('test-32.lib', transform=trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=128, shuffle=False,
        pin_memory=False
    )

    # Start inference
    start_time = time.time()
    test_dset.setmode(1)
    loss, probs = inference(0, test_loader, model, criterion)
    print(time.time() - start_time)
    torch.save(probs, 'probs_32.pth')

if __name__ == '__main__':
    main()
