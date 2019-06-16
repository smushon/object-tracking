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

import sys
sys.path.append("..")
from tracking.tracking_utils import *

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


def train_regnet(md_model_path, force_init_regnet=False, lr=0.001, translate_mode=True, direct_loss=True):

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

    # use_gpu = opts['use_gpu']
    img_size_std = opts['img_size']

    # Init bb refeinement model #
    regnet_model_path = '../models/decider.pth'
    if force_init_regnet:
        regnet_model = RegNet(translate_mode=translate_mode)
        best_prec = 0
    else:
        regnet_model = RegNet(translate_mode=translate_mode, model_path=regnet_model_path)
        best_prec = 0  # should load from saved state !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # for param in regnet_model.parameters():
    #     print(param.data)
    regnet_model.train()
    if torch.cuda.is_available():
        regnet_model = regnet_model.cuda()

    # Init MDNet model #
    md_model = MDNet(md_model_path)
    if torch.cuda.is_available():
        md_model = md_model.cuda()

    RefinementLoss = nn.MSELoss()
    # RefinementLoss = nn.L1Loss()
    # optimizer = optim.Adam(regnet_model.parameters(), lr=0.001)
    optimizer = optim.Adam(regnet_model.parameters(), lr=lr)
    regnet_model.zero_grad()

    # cycle <> epoch, because each cycle we cycle through all sequences, but work only on a batch of frames from each
    for i in range(opts['n_cycles']):
        print('')
        print("==== Start Cycle %d ====" % (i))
        k_list = np.random.permutation(K)  # reorder training sequences each epoch
        curr_epoch_prec_per_seq = np.zeros(K)

        # we iterate over sequences
        # from each sequence we will extract the next batch of frames to train in this epoch
        # think of this as BFS training flow while DFS would have been to train over each sequence fully before moving
        # on to the next sequence
        for j, k in enumerate(k_list):
            tic = time.time()

            pos_regions, pos_bbs, num_example_list, image_list, gt_bbox_list = dataset[k].next()
            idx = 0
            sum_ious = 0
            num_ious = 0
            total_loss = 0
            num_loss_steps = 0

            # we want SGD so we reset grads before each new batch
            # if pretrain_opts['large_memory_gpu'] or not torch.cuda.is_available():
            #     regnet_model.zero_grad()

            # iterting over frames in current batch
            for num_examples, frame, gt_bb in zip(num_example_list, image_list, gt_bbox_list):  # replace with batch ????????????
                num_ious += num_examples

                # frame = Variable(frame)
                # gt_bb = Variable(gt_bb)

                # resizing (scaling) GT-BB to fit standard 107x107 frame
                gt_bb_std = gt_bb
                gt_bb_std[0] = gt_bb[0] * img_size_std / frame.size[0]
                gt_bb_std[2] = gt_bb[2] * img_size_std / frame.size[0]
                gt_bb_std[1] = gt_bb[1] * img_size_std / frame.size[1]
                gt_bb_std[3] = gt_bb[3] * img_size_std / frame.size[1]
                if torch.cuda.is_available():
                    gt_bb_std_as_tensor = torch.from_numpy(gt_bb_std).float().cuda()
                else:
                    gt_bb_std_as_tensor = torch.from_numpy(gt_bb_std).float()

                # iterating over examples extracted for this frame
                for region, bb in zip(pos_regions[idx:idx+num_examples], pos_bbs[idx:idx+num_examples]):
                    # with torch.no_grad():
                    feats_bb = forward_samples(md_model, frame, np.array([bb]), is_cuda=torch.cuda.is_available())
                    feats_frame = forward_samples(md_model, frame, np.array([[0, 0, frame.size[0], frame.size[1]]]), is_cuda=torch.cuda.is_available())

                    # bb_std = np.array(bb)
                    bb_std = bb
                    bb_std[0] = bb[0] * img_size_std / frame.size[0]
                    bb_std[2] = bb[2] * img_size_std / frame.size[0]
                    bb_std[1] = bb[1] * img_size_std / frame.size[1]
                    bb_std[3] = bb[3] * img_size_std / frame.size[1]

                    if torch.cuda.is_available():
                        bb_std_as_tensor = torch.Tensor(np.array([bb_std])).cuda()
                    else:
                        bb_std_as_tensor = torch.Tensor(np.array([bb_std]))

                    net_input = torch.cat((feats_bb, feats_frame, bb_std_as_tensor), dim=1)
                    # if torch.cuda.is_available():
                    #     # net_input = net_input.to(device=device)
                    #     net_input = net_input.cuda()

                    bb_refined_std = regnet_model(net_input)

                    if translate_mode:
                        bb_refined_std += gt_bb_std_as_tensor

                    # iou_score = overlap_ratio(bb_refined, gt_bb)[0]
                    iou_score = torch_overlap_ratio(bb_refined_std, gt_bb_std_as_tensor)
                    sum_ious += iou_score.item()

                    if direct_loss:
                        # loss = RefinementLoss(bb_refined_std, gt_bb_std_as_tensor)
                        loss = RefinementLoss(10*(bb_refined_std - gt_bb_std_as_tensor)/img_size_std, torch.zeros_like(gt_bb_std_as_tensor))
                    else:
                        # ----- iou loss -----------

                        # # iou_score = overlap_ratio(bb_refined, gt_bb)[0]
                        # iou_score = torch_overlap_ratio(bb_refined_std, gt_bb_std_as_tensor)
                        # sum_ious += iou_score.item()

                        iou_target = torch.ones_like(iou_score)
                        if torch.cuda.is_available():
                            # iou_target = iou_target.to(device)
                            iou_target = iou_target.cuda()

                        iou_loss = RefinementLoss(iou_score, iou_target)

                        # ----- distance loss -----------

                        bb_refined_std_center = bb_refined_std[:2] + bb_refined_std[2:] / 2
                        gt_bb_std_center_as_tensor = gt_bb_std_as_tensor[:2] + gt_bb_std_as_tensor[2:] / 2
                        result_distance = torch.dist(bb_refined_std_center, gt_bb_std_center_as_tensor)
                        result_distance_norm = result_distance / img_size_std

                        distance_target = torch.zeros_like(result_distance_norm)
                        if torch.cuda.is_available():
                            distance_target = distance_target.cuda()

                        distance_loss = RefinementLoss(result_distance_norm, distance_target)

                        # ----- size loss -----------

                        bb_refined_std_size = bb_refined_std[2]*bb_refined_std[3]
                        gt_bb_std_size = gt_bb_std_as_tensor[2]*gt_bb_std_as_tensor[3]
                        size_relative = bb_refined_std_size / gt_bb_std_size
                        # size_relative = torch.log(size_relative)

                        size_target = torch.ones_like(result_distance_norm)
                        if torch.cuda.is_available():
                            size_target = size_target.cuda()

                        size_loss = RefinementLoss(size_relative, size_target)

                        # ----- combined loss -----------

                        if translate_mode:
                            loss = (distance_loss * 5) + (size_loss / 4) #+ iou_loss
                        else:
                            loss = distance_loss + size_loss

                    if pretrain_opts['large_memory_gpu'] or not torch.cuda.is_available():
                        total_loss += loss
                    else:
                        # we don't normalize loss because number of back-props before optim.step() isnt deterministic
                        loss.backward()
                        total_loss += loss.clone().cpu().data

                    num_loss_steps += 1

                idx += num_examples

            # we want SGD so we update grads only after batch has ended
            if not num_loss_steps == num_ious:
                raise Exception('sanity failed: loss_steps = %d, num_iou = %d' % (num_loss_steps, num_ious))
            total_loss = total_loss / num_ious
            if pretrain_opts['large_memory_gpu'] or not torch.cuda.is_available():
                total_loss.backward()
            # torch.nn.utils.clip_grad_norm(regnet_model.parameters(), opts['grad_clip'])  # ??????????????
            optimizer.step()  # no need for 'with torch.no_grad():' since we use optimizer from torch.optim
            regnet_model.zero_grad()

            curr_epoch_prec_per_seq[k] = sum_ious / num_ious
            toc = time.time() - tic  # time it took to train over current sequqnce

            # displaying stats for current sequence
            # reminder: we only processed a batch of frames from current sequence, not all of it
            if pretrain_opts['large_memory_gpu'] or not torch.cuda.is_available():
                if total_loss.dim() == 0:
                    total_loss = total_loss.data
                else:
                    total_loss = total_loss.data[0]  # what ???????????????????????????????????????????
            print("Cycle %d, [iter %d/%d] (seq %2d), Loss %.3f, IoU %.3f, Time %.3f" % \
                      (i, j, K, k, total_loss, curr_epoch_prec_per_seq[k], toc))

        cur_prec = curr_epoch_prec_per_seq.mean()  # precision of this epoch
        print("Mean IoU: %.3f" % (cur_prec))

        if cur_prec > best_prec:
            best_prec = cur_prec
            if torch.cuda.is_available():
                regnet_model = regnet_model.cpu()
            states = {'RegNet_layers': regnet_model.layers.state_dict()}
            print("Save regnet_model to %s" % regnet_model_path)
            torch.save(states, regnet_model_path)
            if torch.cuda.is_available():
                regnet_model = regnet_model.cuda()


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


force_train_mdnet = False
force_train_regnet = True
force_init_regnet = False
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # ----- mdnet related -----
    parser.add_argument('-md', '--train_mdnet', action='store_true')  # defaut: don't traing mdnet
    # ----- regnet related -----
    parser.add_argument('-rg', '--train_regnet', action='store_true')  # defaut: don't traing regnet
    parser.add_argument('-tm', '--trained_mdnet', action='store_true')  # default: use original mdnet weights
    parser.add_argument('-ir', '--init_regnet', action='store_false')  # default: init regnet (else - cont saved state)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('-ot', '--translate_mode', action='store_false')  # default: output as coord shift
    parser.add_argument('-il', '--indirect_loss', action='store_true')  # default: direct (mse) loss
    args = parser.parse_args()

    # ------------

    if args.train_mdnet or force_train_mdnet:
        print('trainig mdnet')
        train_mdnet()

    # ------------

    # we need md_model for extracting features from frames
    # these features are used as inputs to the regressor network
    # we selct either the original MDNet paper model, or one that we trained
    if args.trained_mdnet:
        md_model_path = pretrain_opts['model_path']
    else:
        md_model_path = pretrain_opts['init_model_path']
    if args.train_regnet or force_train_regnet:
        print('trainig regnet, learning_rate=%.4f' % (args.learning_rate))
        if args.translate_mode:
            print('regnet output translation')
        else:
            print('regnet outputs coordinates')
        if args.indirect_loss:
            print('loss: combined')
        else:
            print('loss: mse')
        train_regnet(md_model_path, force_init_regnet=force_init_regnet or args.init_regnet, lr=args.learning_rate, translate_mode=args.translate_mode, direct_loss=not args.indirect_loss)
