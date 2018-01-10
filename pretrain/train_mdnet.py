import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

from data_prov import *
from model import *
from options import *
from tensorboardX import SummaryWriter
from FocalLoss import *

img_home = '/data1/tracking'
data_path = 'data/vot-otb.pkl'


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train_mdnet():
    ## Init dataset ##
    # with open(data_path, 'rb') as fp:
    #     data = pickle.load(fp)
    # to unpickle .pkl in python3 which is pickles in python2, should do like this
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
    # and to load values in this dict, should use data[key.encode('utf-8')]

    K = len(data)
    dataset = [None] * K
    # for k, (seqname, seq) in enumerate(data.iteritems()):
    seqnames = []
    for k,  seqname in enumerate(data):
        img_list=data[seqname]['images']
        gt = data[seqname]['gt']
        img_dir = os.path.join(img_home, seqname)
        dataset[k] = RegionDataset(img_dir, img_list, gt, opts)
        seqnames.append(seqname)

    use_summary = opts['use_summary']
    use_gpu = opts['use_gpu']
    # prepare for tensorboardX
    if use_summary:
        summary = SummaryWriter(comment='CrossEntropyLoss')
        summary_different_model = SummaryWriter('runs/comparsion_between_model/CrossEntropyLoss_MDNet')

    ## Init model ##
    model = MDNet(opts['init_model_path'], K)
    if use_gpu:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    ## Init criterion and optimizer ##
    # criterion = BinaryLoss()
    # posLoss = FocalLoss(class_num=2,size_average=False,alpha=torch.ones(2,1)*0.25)
    # negLoss = FocalLoss(class_num=2, size_average=False,alpha=torch.ones(2,1)*0.25)
    posLoss = nn.CrossEntropyLoss(size_average=False)
    negLoss = nn.CrossEntropyLoss(size_average=False)
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'])

    best_prec = 0.
    prec_per = dict()
    for s in seqnames:
        prec_per[s]=0

    for i in range(opts['n_cycles']):
        print("==== Start Cycle %d ====" % (i))
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        total_loss = 0
        for j, k in enumerate(k_list):
            tic = time.time()
            pos_regions, neg_regions = dataset[k].next()

            pos_target = np.ones(pos_regions.shape[0], dtype=int)
            neg_target = np.zeros(neg_regions.shape[0], dtype=int)
            pos_target = torch.from_numpy(pos_target)
            neg_target = torch.from_numpy(neg_target)
            pos_target = Variable(pos_target)
            neg_target = Variable(neg_target)

            pos_regions = Variable(pos_regions)
            neg_regions = Variable(neg_regions)

            if use_gpu:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
                pos_target = pos_target.cuda()
                neg_target = neg_target.cuda()

            pos_score = model(pos_regions, k)
            neg_score = model(neg_regions, k)

            pos_loss = posLoss(pos_score,pos_target)
            neg_loss = negLoss(neg_score,neg_target)

            loss = pos_loss + neg_loss
            total_loss += loss.clone().cpu().data[0]
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time() - tic
            print("Cycle %d, [%d/%d] (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                  (i, j, K, k, loss.data[0], prec[k], toc))

        cur_prec = prec.mean()
        if use_summary:
            summary.add_scalar('total_loss',total_loss/K,i)
            for index, _ in enumerate(seqnames):
                prec_per[seqnames[index]]=prec[index]
            summary.add_scalars('precision_per_seq',prec_per,i)
            summary.add_scalar('mean_precision',cur_prec,i)
            summary_different_model.add_scalar('total_loss',total_loss/K,i)
            summary_different_model.add_scalar('mean_precision',cur_prec,i)
        print("Mean Precision: %.3f" % (cur_prec))
        if cur_prec > best_prec:
            best_prec = cur_prec
            if use_gpu:
                model = model.cpu()
            states = {'shared_layers': model.layers.state_dict()}
            print("Save model to %s" % opts['model_path'])
            torch.save(states, opts['model_path'])
            if use_gpu:
                model = model.cuda()

    if use_summary:
        summary.close()
        summary_different_model.close()


if __name__ == "__main__":
    train_mdnet()
