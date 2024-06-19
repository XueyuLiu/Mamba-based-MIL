# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:25:28 2019

@author: SCSC
"""
import csv, torch, sys
import torch.nn.functional as F
import numpy as np
from MedMamba import VSSM as medmamba
def writecsv(wlist, dst):
    wlist = list(map(str,wlist))
    with open(dst,'a', newline='') as fw:
        csv_writer = csv.writer(fw)
        csv_writer.writerow(wlist)

def inference(run, loader, model, criterion):
    model.eval()
    running_loss = 0.
    probs = []
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            sys.stdout.write('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]\r'.format(run+1, 100, i+1, len(loader)))
            sys.stdout.flush()
            target = target.cuda()
            input = input.cuda()
            output = model(input)
            #criterion=multiClassHingeLoss()
            #print(criterion)
            loss = criterion(output, target)
            #loss=torch.mean(torch.clamp(1 - output.t() * target, min=0))
            #print(criterion)

            running_loss += loss.item()*input.size(0)
            output = F.softmax(output, dim=1)
            probs.extend(output.detach()[:,1].cpu().numpy().tolist()) #输出的第一列（预测值为正例则输出1，反例输出0）放到probs里
    print('')
    probs = np.array(probs)
    loss = running_loss/len(loader.dataset)
    return loss, probs

def train(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    probs = []
    real = []
    for i, (input, target) in enumerate(loader):
        real.extend(target.numpy().tolist())
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        pred = F.softmax(output, dim=1)
        probs.extend(pred.detach()[:,1].cpu().numpy().tolist())
        #criterion=torch.nn.HingeEmbeddingLoss(margin=1.0,  reduction='mean')
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    probs = [1 if x >= 0.5 else 0 for x in probs]
    err = calc_err(probs, real)
    loss = running_loss/len(loader.dataset)
    print(loss)
    return loss, err

def calc_err(probs,real):
    probs = np.array(probs)
    real = np.array(real)
    assert len(probs) == len(real)
    neq = np.not_equal(probs, real)
    err = float(neq.sum())/probs.shape[0]
#    fpr = float(np.logical_and(probs==1,neq).sum())/(real==0).sum()  #FP占所有负例的比例
#    fnr = float(np.logical_and(probs==0,neq).sum())/(real==1).sum()  #FN占所有正例的比例
    return err
#    return err, fpr, fnr


def group_argtopk(groups, data,k=1):  #groups为所有瓦片对应的切片序号组成的array，data为这些瓦片的预测值
    order = np.lexsort((data, groups))  #首先按照groups的元素排序，如果出现相同大小的情况，则再按照data排序，由小到大。
    groups = groups[order]  #把order对应的groups元素取出来
    data = data[order]  #把排好序的data取出来
    index = np.empty(len(groups), 'bool')
    index[-k:] = True  #最后K个设为True
    index[:-k] = groups[k:] != groups[:-k]  #将groups错开k位，这样就能保证每个slides返回概率最大的k个tiles
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]  #由Numpy的填充性质可得，可以保证被抽到的groups都有该groups最大的data(概率)，同时最大的概率对应的group一定会存在于out中。
    return out