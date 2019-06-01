import platform
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


# img_home = '/data1/tracking'
usr_home = os.path.expanduser('~')
OS = platform.system()
if OS == 'Windows':
    # usr_home = 'C:/Users/smush/'
    img_home = os.path.join(usr_home, 'downloads/VOT')
elif OS == 'Linux':
    # usr_home = '~/'
    img_home = os.path.join(usr_home, 'MDNet-data/VOT')
else:
    sys.exit("aa! errors!")

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


def train_fcnet():
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
    K = len(data)  # K is the number of sequences (58)
    dataset = [None] * K

    seqnames = []
    for k, seqname in enumerate(data):
        img_list = data[seqname]['images']
        gt = data[seqname]['gt']  # gt is a ndarray of rectangles
        img_dir = os.path.join(img_home, seqname)
        dataset[k] = RegionDataset(img_dir, img_list, gt, opts)  ############# ????????????????????????

        seqnames.append(seqname)

    use_gpu = opts['use_gpu']

    # Init model #
    bb_fc_model_path = None  ##################### TBD ##########################
    bb_fc_model = FCRegressor(bb_fc_model_path)
    if use_gpu:
        bb_fc_model = bb_fc_model.cuda()

    best_prec = 0
    for i in range(opts['n_cycles']):
        print('')
        print("==== Start Cycle %d ====" % (i))
        k_list = np.random.permutation(K)  # reorder training sequences each epoch
        curr_epoch_prec_per_seq = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()

            pos_regions, neg_regions = dataset[k].next()
            pos_target = np.ones(pos_regions.shape[0], dtype=int)

            pos_regions = Variable(pos_regions)
            if use_gpu:
                pos_regions = pos_regions.cuda()

            #################################
            cv_feats_BB = forward_samples(bb_fc_model, image, np.array([cv_BB]))
            cv_feats_full_frame = forward_samples(bb_fc_model, image, np.array([[0, 0, image.size[0], image.size[1]]]))

            cv_BB_std = np.array(cv_BB)
            img_size_std = opts['img_size']
            cv_BB_std[0] = cv_BB[0] * img_size_std / image.size[0]
            cv_BB_std[2] = cv_BB[2] * img_size_std / image.size[0]
            cv_BB_std[1] = cv_BB[1] * img_size_std / image.size[1]
            cv_BB_std[3] = cv_BB[3] * img_size_std / image.size[1]

            bb_fc_input = torch.cat((cv_feats_BB, cv_feats_full_frame, torch.Tensor(np.array([cv_BB_std]))), dim=1)

            with torch.no_grad():
                bb_fc_input = bb_fc_input.to(device=device)
                cv_BB_refined_std = bb_fc_model(bb_fc_input)

            bb_fc_input = bb_fc_input.to(device='cpu')  # is this needed ??????????????????????
            cv_BB_refined_std = cv_BB_refined_std.to(device='cpu')  # is this needed ??????????????????????
            #################################

            pos_score = bb_fc_model(pos_regions, k)

            pos_target = torch.from_numpy(pos_target)
            pos_target = Variable(pos_target)
            if use_gpu:
                pos_target = pos_target.cuda()

            loss = posLoss(pos_score, pos_target)

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])  ######## ????????????????
            optimizer.step()
            curr_epoch_prec_per_seq[k] = evaluator(pos_score)

            toc = time.time() - tic
            if loss.dim() == 0:
                print("Cycle %d, [%d/%d] (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                      (i, j, K, k, loss.data, curr_epoch_prec_per_seq[k], toc))
            else:
                print("Cycle %d, [%d/%d] (seq %2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                      (i, j, K, k, loss.data[0], curr_epoch_prec_per_seq[k], toc))

        cur_prec = curr_epoch_prec_per_seq.mean()
        print("Mean Precision: %.3f" % (cur_prec))

        if cur_prec > best_prec:
            best_prec = cur_prec
            if use_gpu:
                model = model.cpu()
            states = {'FCRegressor_layers': model.layers.state_dict()}
            print("Save model to %s" % bb_fc_model_path)
            torch.save(states, bb_fc_model_path)
            if use_gpu:
                model = model.cuda()


def train_mdnet():
    # Init dataset #
    # with open(data_path, 'rb') as fp:
    #     data = pickle.load(fp)
    # to unpickle .pkl in python3 which is pickles in python2, should do like this
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)
    # and to load values in this dict, should use data[key.encode('utf-8')]

    ###############################################
    # data is a dictionary with elements like this:
    # 'vot2016/octopus': {
    #        'images': ['00000001.jpg', '00000002.jpg', ... , '00000291.jpg'],
    #        'gt': array([
    #           [628.85, 264.86, 149.53, 113.96],
    #           [630.68, 267.29, 149.28, 111.17],
    #            ...,
    #            [634.41, 399.8 , 410.39, 231.23]
    #        ])
    # }
    # list = []
    # for k, seqname in enumerate(data):
    #     list.append(seqname)
    #     print(seqname)
    ###############################################

    K = len(data)  # K is the number of sequences (58)
    dataset = [None] * K
    # for k, (seqname, seq) in enumerate(data.iteritems()):
    seqnames = []
    for k,  seqname in enumerate(data):
        img_list = data[seqname]['images']
        gt = data[seqname]['gt']  # gt is a ndarray of rectangles
        img_dir = os.path.join(img_home, seqname)
        dataset[k] = RegionDataset(img_dir, img_list, gt, opts)
        seqnames.append(seqname)

    use_summary = opts['use_summary']
    use_gpu = opts['use_gpu']
    # prepare for tensorboardX
    if use_summary:
        summary = SummaryWriter(comment='CrossEntropyLoss')
        summary_different_model = SummaryWriter('runs/comparsion_between_model/CrossEntropyLoss_MDNet')

    # Init model #
    model = MDNet(opts['init_model_path'], K)
    if use_gpu:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer #
    # criterion = BinaryLoss()
    # posLoss = FocalLoss(class_num=2,size_average=False,alpha=torch.ones(2,1)*0.25)
    # negLoss = FocalLoss(class_num=2, size_average=False,alpha=torch.ones(2,1)*0.25)

    ################################################
    posLoss = nn.CrossEntropyLoss(size_average=False)
    negLoss = nn.CrossEntropyLoss(size_average=False)
    # seems to work, only warning. also reduction default seems to be 'mean' so I don't want to touch this
    # posLoss = nn.CrossEntropyLoss(reduction='sum')
    # negLoss = nn.CrossEntropyLoss(reduction='sum')
    ################################################

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

            ########################################
            pos_target = pos_target.type(torch.long)
            neg_target = neg_target.type(torch.long)
            ########################################

            pos_loss = posLoss(pos_score, pos_target)
            neg_loss = negLoss(neg_score, neg_target)

            loss = pos_loss + neg_loss
            ##########################################
            if loss.clone().cpu().dim() == 0:
                total_loss += loss.clone().cpu().data
            else:
                total_loss += loss.clone().cpu().data[0]
            ##########################################
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time() - tic

            ##########################################
            if loss.dim() == 0:
                print("Cycle %d, [%d/%d] (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                      (i, j, K, k, loss.data, prec[k], toc))
            else:
                print("Cycle %d, [%d/%d] (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                      (i, j, K, k, loss.data[0], prec[k], toc))
            ##########################################


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


train_mdnet = False
train_fcnet = True
if __name__ == "__main__":

    if train_mdnet():
        train_mdnet()

    if train_fcnet:
        train_fcnet()
