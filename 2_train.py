# -*- coding: utf-8 -*-
"""
Main training script for a model using the MedMamba architecture.
Requires inference, training, and various utility functions.
"""

import torch
import time
from random import shuffle as sf
from torch import nn, optim
import numpy as np
from MyLoader import OrigDataset as XDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import resnet18
from MedMamba import VSSM as medmamba
from utils import inference, train, group_argtopk, writecsv, group_max, calc_err
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Batch size of 128 is optimal based on testing
def main():
    best_acc = 0
    pk = 2  # Number of positive examples to select
    nk = 5  # Number of negative examples to select
    n_epoch = 25  # Number of epochs
    test_every = 1  # Frequency of testing (every n epochs)

    # Define the device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = medmamba(num_classes=2)
    model.cuda()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    cudnn.benchmark = True

    # Define data transformations and datasets
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    train_dset = XDataset('train-32.lib', transform=trans)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=128, shuffle=False, pin_memory=False)
    test_dset = XDataset('test-32.lib', transform=trans)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=128, shuffle=False, pin_memory=False)

    # Initialize log files
    with open('Training_32.csv', 'w') as f:
        f.write('time,epoch,loss,error\n')
        f.write('%d,0,0,0\n' % time.time())

    with open('Testing_32.csv', 'w') as f:
        f.write('time,epoch,loss,error\n')
        f.write('%d,0,0,0\n' % time.time())

    # Start the training process
    for epoch in range(n_epoch):
        # Phase 1: Inference on the entire training set
        train_dset.setmode(1)
        _, probs = inference(epoch, train_loader, model, criterion)
        probs1 = probs[:train_dset.plen]  # Probs from positive examples
        probs0 = probs[train_dset.plen:]  # Probs from negative examples

        # Phase 2: Select top-k examples
        topk1 = np.array(group_argtopk(np.array(train_dset.slideIDX[:train_dset.plen]), probs1, pk))
        topk0 = np.array(group_argtopk(np.array(train_dset.slideIDX[train_dset.plen:]), probs0, nk)) + train_dset.plen
        topk = np.append(topk1, topk0).tolist()
        sf(topk)

        # Phase 3: Prepare training data
        train_dset.maketraindata(topk)
        train_dset.setmode(2)

        # Phase 4: Train the model and save results
        loss, err = train(train_loader, model, criterion, optimizer)
        writecsv([time.time(), epoch + 1, loss, err], 'Training.csv')
        print(f'Training epoch={epoch + 1}, loss={loss:.5f}, error={err:.5f}')

        # Phase 5: Validate the model
        if (epoch + 1) % test_every == 0:
            test_dset.setmode(1)
            loss, probs = inference(epoch, test_loader, model, criterion)
            maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))  # Max probability for each slice
            pred = [1 if x >= 0.5 else 0 for x in maxs]
            err = calc_err(pred, test_dset.targets)
            writecsv([time.time(), epoch + 1, loss, err], 'Testing.csv')
            print(f'Testing epoch={epoch + 1}, loss={loss:.5f}, error={err:.5f}')

            # Save the best model
            if 1 - err >= best_acc:
                best_acc = 1 - err
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }, 'checkpoint_128.pth')

if __name__ == '__main__':
    main()
