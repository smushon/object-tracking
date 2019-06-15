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

from tracking_utils import *
import options
device = options.training_device
opts = options.pretrain_opts
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


def train_fcnet(md_model_path):

    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)


    K = len(data)  # K is the number of sequences (58)
    dataset = [None] * K

    for k, seqname in enumerate(data):
        img_list = data[seqname]['images']
        gt = data[seqname]['gt']  # gt is a ndarray of rectangles
        img_dir = os.path.join(img_home, seqname)
        dataset[k] = PosRegionDataset(img_dir, img_list, gt, opts)
        # dataset[k] = FCDataset(img_dir, img_list, gt, opts)

    use_gpu = opts['use_gpu']

    # Init bb refeinement model #
    bb_fc_model_path = '../models/decider.pth'
    bb_fc_model = FCRegressor(bb_fc_model_path)
    bb_fc_model.train()
    if use_gpu:
        bb_fc_model = bb_fc_model.cuda()

    # Init MDNet model #
    md_model = MDNet(md_model_path)
    if opts['use_gpu']:
        md_model = md_model.to(device)

    # RefinementLoss = nn.CrossEntropyLoss(size_average=False)
    RefinementLoss = nn.MSELoss(reduce=False)
    optimizer = optim.Adam(bb_fc_model.parameters(), lr=0.001)

    best_prec = 0
    for i in range(opts['n_cycles']):
        print('')
        print("==== Start Cycle %d ====" % (i))
        k_list = np.random.permutation(K)  # reorder training sequences each epoch
        curr_epoch_prec_per_seq = np.zeros(K)

        # we iterate over sequences
        # from each sequence we will extract the next batch of frames to train in this epoch
        # think of this as BFS
        for j, k in enumerate(k_list):
            tic = time.time()

            pos_regions, pos_bbs, num_example_list, image_list, gt_bbox_list = dataset[k].next()
            idx = 0
            sum_ious = 0
            num_ious = 0

            # iterting over frames in current batch
            for num_examples, frame, gt_bb in zip(num_example_list, image_list, gt_bbox_list):  # replace with batch ????????????
                num_ious += num_examples

                # frame = Variable(frame)
                # gt_bb = Variable(gt_bb)

                # iterating over examples extracted for this frame
                for region, bb in zip(pos_regions[idx:idx+num_examples], pos_bbs[idx:idx+num_examples]):
                    # with torch.no_grad():
                    feats_bb = forward_samples(md_model, frame, np.array([bb]))
                    feats_frame = forward_samples(md_model, frame, np.array([[0, 0, frame.size[0], frame.size[1]]]))

                    bb_std = np.array(bb)
                    img_size_std = opts['img_size']
                    bb_std[0] = bb[0] * img_size_std / frame.size[0]
                    bb_std[2] = bb[2] * img_size_std / frame.size[0]
                    bb_std[1] = bb[1] * img_size_std / frame.size[1]
                    bb_std[3] = bb[3] * img_size_std / frame.size[1]

                    net_input = torch.cat((feats_bb, feats_frame, torch.Tensor(np.array([bb_std]))), dim=1)
                    if use_gpu:
                        net_input = net_input.to(device=device)

                    bb_refined_std = bb_fc_model(net_input)

                    # bb_refined_std = bb_refined_std.numpy()
                    #                     # bb_refined = bb_refined_std
                    #                     # bb_refined[0] = bb_refined_std[0] * frame.size[0] / img_size_std
                    #                     # bb_refined[2] = bb_refined_std[2] * frame.size[0] / img_size_std
                    #                     # bb_refined[1] = bb_refined_std[1] * frame.size[1] / img_size_std
                    #                     # bb_refined[3] = bb_refined_std[3] * frame.size[1] / img_size_std

                    # gt_bb_std = np.array(gt_bb)
                    gt_bb_std = gt_bb
                    gt_bb_std[0] = gt_bb[0] * img_size_std / frame.size[0]
                    gt_bb_std[2] = gt_bb[2] * img_size_std / frame.size[0]
                    gt_bb_std[1] = gt_bb[1] * img_size_std / frame.size[1]
                    gt_bb_std[3] = gt_bb[3] * img_size_std / frame.size[1]
                    gt_bb_std = torch.from_numpy(gt_bb_std).float()

                    # iou_score = overlap_ratio(bb_refined, gt_bb)[0]
                    iou_score = torch_overlap_ratio(bb_refined_std, gt_bb_std)
                    sum_ious += iou_score.item()

                    iou_target = torch.ones_like(iou_score)
                    if use_gpu:
                        iou_target = iou_target.to(device)

                    loss = RefinementLoss(iou_score, iou_target)

                    bb_fc_model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(bb_fc_model.parameters(), opts['grad_clip'])  ######## ????????????????

                    # no need for 'with torch.no_grad():' since we use optimizer from torch.optim
                    optimizer.step()

                idx += num_examples

            curr_epoch_prec_per_seq[k] = sum_ious / num_ious
            toc = time.time() - tic  # time it took to train over current sequqnce

            if loss.dim() == 0:
                print("Cycle %d, [%d/%d] (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                      (i, j, K, k, loss.data, curr_epoch_prec_per_seq[k], toc))
            else:
                print("Cycle %d, [%d/%d] (seq %2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                      (i, j, K, k, loss.data[0], curr_epoch_prec_per_seq[k], toc))

        cur_prec = curr_epoch_prec_per_seq.mean()  # precision of this epoch
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
        model = model.to(device)
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer #
    # criterion = BinaryLoss
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


run_train_mdnet = False
run_train_fcnet = True
if __name__ == "__main__":

    if run_train_mdnet:
        train_mdnet()

    # md_model_path = pretrain_opts['init_model_path']
    md_model_path = pretrain_opts['model_path']

    if run_train_fcnet:
        train_fcnet(md_model_path)
