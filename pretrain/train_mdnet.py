import platform
import os
import sys
import pickle
import time
import datetime
# import msvcrt

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
import matplotlib.pyplot as plt
import copy
from shapely.geometry import Polygon

# this is output of the script prepro_data.py
data_path_vot = 'data/mdnet_vot.pkl'
data_path_vot_quadrilateral = 'data/vot_quadrilateral.pkl'
data_path_otb = 'data/otb.pkl'
data_path_default = data_path_vot

quadrilateral = True

# img_home = '/data1/tracking'
usr_home = os.path.expanduser('~')
OS = platform.system()
if OS == 'Windows':
    # usr_home = 'C:/Users/smush/'
    img_home_default = os.path.join(usr_home, 'downloads', data_path_default)
elif OS == 'Linux':
    # usr_home = '~/'
    img_home_default = os.path.join(usr_home, 'MDNet-data/' + data_path_default)
else:
    sys.exit("aa! errors!")

import sys
sys.path.append("..")
from tracking.tracking_utils import *
from pathlib import Path, PureWindowsPath

import options
device = options.training_device
opts = options.pretrain_opts
from tracking.options import tracking_opts


# so basically it sets the learning rate for the fc layers to be 10 times that of conv layers
# note that each layer can be given a different lr
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


train_regnet_opts_defaults = {
    'init_regnet': False,
    'lr': 0.0001,
    'translate_mode': True,
    'direct_loss': True,
    # 'fewer_sequences': 0,
    'training_indices': [0,0],
    'dont_save': False,
    'saved_state': None,
    'first_cycle': 0,
    'fixed_learning_rate': False,
    'limit_frame_per_seq': False,
    'regnet_model_path': '../models/regnet.pth',
    'pre_generate': True,
    'blackout': False,
    'data_path': data_path_default,
    'img_home': img_home_default
}



validate_regnet_opts_defaults = {
    'translate_mode': True,
    # 'fewer_sequences': False,
    'saved_state': None,
    'limit_frame_per_seq': False,
    'validation_data_indices': [0,0],
    'validation_data_path': data_path_default,
    'validation_sub_sample': False,
    'validation_sub_sample_oddness': False,
    'regnet_model_path': '../models/regnet.pth',
    'blackout': False,
    'img_home': img_home_default
}


# num_fewer_seq = 2 # 30 # 10
num_frames_per_seq = 2 # 16
sub_sample = True

# choosing True will accelerate about 15%
# but will cause much larger variance in training curve
# if we pre-generate the samples then it doesn't matter anyway...
# generate_std = False

# pre_generate = True


def validate_regnet(md_model_path, validate_regnet_opts=validate_regnet_opts_defaults):
    generate_std = False
    # use_gpu = opts['use_gpu']
    img_size_std = opts['img_size']

    translate_mode = validate_regnet_opts['translate_mode']
    # fewer_sequences = validate_regnet_opts['fewer_sequences']
    saved_state = validate_regnet_opts['saved_state']
    limit_frame_per_seq = validate_regnet_opts['limit_frame_per_seq']
    validation_data_indices = validate_regnet_opts['validation_data_indices']
    validation_data_path = validate_regnet_opts['validation_data_path']
    validation_sub_sample = validate_regnet_opts['validation_sub_sample']
    validation_sub_sample_oddness = validate_regnet_opts['validation_sub_sample_oddness']
    regnet_model_path = validate_regnet_opts['regnet_model_path']
    blackout = validate_regnet_opts['blackout']
    img_home = validate_regnet_opts['img_home']

    # ------------

    # regnet_model_path = '../models/regnet.pth'
    if saved_state is not None:
        state = saved_state
    else:
        state = torch.load(regnet_model_path)
    print('model saved layers:')
    model_str = ''
    for key, value in state['RegNet_layers'].items():
        if key.split('.')[-1] == 'weight':
            print('  ' + key.split('.')[0] + ': ' + str(value.shape[1]) + ' --> ' + str(value.shape[0]))
            model_str += str(value.shape[1])
    model_str += str(value.shape[0])
    regnet_model = RegNet(translate_mode=translate_mode, state=state)
    # regnet_model.load_state_dict(state['RegNet_layers'])  # done via RegNet() init function instead
    regnet_model.eval()
    if torch.cuda.is_available():
        regnet_model = regnet_model.cuda()
    if 'best_prec' in state.keys():
        best_prec = state['best_prec']
        # print("starting precision = %.5f" % best_prec)
    else:
        best_prec = 0.0
    if 'best_std' in state.keys():
        best_std = state['best_std']
    else:
        best_std = 0.0
    if 'translate_mode' in state.keys():
        translate_mode = state['translate_mode']  # we overide train_regnet input according to saved state

    print("model precision (avg:std) = (%.5f:%.5f)" % (best_prec, best_std))

    # -------------

    # Init MDNet model #
    md_model = MDNet(md_model_path)
    if torch.cuda.is_available():
        md_model = md_model.cuda()

    # -------------

    if quadrilateral:  # this is a bad hack, should be temporary........... !!!!!!!!!!!!
        with open(data_path_vot_quadrilateral, 'rb') as fp:
            data = pickle.load(fp)
    else:
        with open(validation_data_path, 'rb') as fp:
            data = pickle.load(fp)


    # [0] -> starting index, [1] -> number of sequences
    validation_data_indices[0] = max(validation_data_indices[0], 0)
    validation_data_indices[0] = min(len(data)-1, validation_data_indices[0])

    K = len(data) - validation_data_indices[0]
    if validation_data_indices[1] > 0:
        K = min(validation_data_indices[1], K)
    # if fewer_sequences:
    #     K = min(num_fewer_seq, K)

    dataset = [None] * K
    seqnames = [None] * K
    frame_features = [None] * K
    num_frames = [None] * K
    for j, seqname in enumerate(data):
        if j < validation_data_indices[0]:
            continue
        seqname.replace("\\", '/')
        print('preparing sequence %d - ' % j + seqname)

        k = j - validation_data_indices[0]
        seqnames[k] = seqname
        img_list = data[seqname]['images']
        gt = data[seqname]['gt']  # gt is a ndarray of rectangles
        if quadrilateral:
            gt_origin = data[seqname]['gt_origin']
        if validation_sub_sample:  # thin down their number - complementary of training indices !
            if validation_sub_sample_oddness:
                img_list = img_list[1::2]
                gt = gt[1::2]
                if quadrilateral:
                    gt_origin = gt_origin[1::2]
            else:
                img_list = img_list[0::2]
                gt = gt[0::2]
                if quadrilateral:
                    gt_origin = gt_origin[0::2]
        if limit_frame_per_seq:
            num_frames[k] = min(len(img_list), num_frames_per_seq)
            img_list = img_list[:num_frames[k]]
            gt = gt[:num_frames[k]]
            if quadrilateral:
                gt_origin = gt_origin[:num_frames[k]]
        else:
            num_frames[k] = len(img_list)
        if OS == 'Windows':
            img_dir = os.path.join(img_home, str(PureWindowsPath(seqname)))
        else:
            img_dir = os.path.join(img_home, seqname)
        # every sequence gets the frames order randomly permutated
        # this also happens during training every time frame are exhausted

        dataset[k] = PosRegionDataset(img_dir, img_list, gt, opts, torch.cuda.is_available(),
                                      generate_std=False, pre_generate=False, seq_regions_filename='', blackout=blackout)

        img_path_list = np.array([os.path.join(img_dir, img) for img in img_list])
        frame_features[k] = {}
        for img_path in img_path_list:
            image = Image.open(img_path).convert('RGB')
            curr_frame_features = forward_samples(md_model, image, np.array([[0, 0, image.size[0], image.size[1]]]), is_cuda=torch.cuda.is_available())
            frame_features[k][img_path] = curr_frame_features

        if k + 1 == K:
            break

    print('loaded %d sequences' % K)
    print('frames per sequence, avg: %.4g, med: %.4g' % (np.mean(num_frames), np.median(num_frames)))

    # --------------

    for i in range(0, 100):
        print('')
        print("==== Start Cycle %d ====" % i)

        k_list = np.arange(K)

        prec_per_seq = np.zeros((K,3))
        iou_per_seq = np.zeros((K,3))
        toc = np.zeros(K)
        for k in k_list:
            tic = time.time()

            pos_regions, pos_bbs, num_example_list, frame_numbers = dataset[k].next()

            # -----------------

            num_ious = sum(num_example_list)
            image_path_list = dataset[k].img_list[frame_numbers]
            image_size = dataset[k].image_size
            if quadrilateral:
                gt_bbox_std_as_tensor = dataset[k].gt_origin_std_as_tensor[frame_numbers]
            else:
                gt_bbox_std_as_tensor = dataset[k].gt_std_as_tensor[frame_numbers]

            feats_bbs = forward_regions(md_model, pos_regions, is_cuda=torch.cuda.is_available())
            # using forward_samples on pos_bbs called crop image with valid==False ...

            for iterator, (num_examples, frame_path) in enumerate(zip(num_example_list, image_path_list)):
                feats_frame = frame_features[k][frame_path]
                if iterator == 0:
                    expanded_feats_frames = feats_frame.repeat(num_examples, 1)
                    expanded_gt_bbox_std_as_tensor = gt_bbox_std_as_tensor[iterator].repeat(num_examples,1)
                else:
                    expanded_feats_frames = torch.cat((expanded_feats_frames, feats_frame.repeat(num_examples, 1)))
                    expanded_gt_bbox_std_as_tensor = torch.cat((expanded_gt_bbox_std_as_tensor, gt_bbox_std_as_tensor[iterator].repeat(num_examples, 1)))

            pos_bbs_std = pos_bbs
            # assuming all frames in given sequence have the same size
            pos_bbs_std[:,0] = pos_bbs[:,0] * img_size_std / image_size[0]
            pos_bbs_std[:,2] = pos_bbs[:,2] * img_size_std / image_size[0]
            pos_bbs_std[:,1] = pos_bbs[:,1] * img_size_std / image_size[1]
            pos_bbs_std[:,3] = pos_bbs[:,3] * img_size_std / image_size[1]

            if torch.cuda.is_available():
                pos_bbs_std_as_tensor = torch.Tensor(pos_bbs_std).cuda()
            else:
                pos_bbs_std_as_tensor = torch.Tensor(pos_bbs_std)

            net_input = torch.cat((feats_bbs, expanded_feats_frames, pos_bbs_std_as_tensor), dim=1)

            bb_refined_std = regnet_model(net_input)
            if translate_mode:
                if quadrilateral:
                    pos_bbs_quadrilateral_std_as_tensor = pos_bbs_std_as_tensor.repeat(1, 2)
                    pos_bbs_quadrilateral_std_as_tensor[:, 0] = pos_bbs_std_as_tensor[:, 0]
                    pos_bbs_quadrilateral_std_as_tensor[:, 1] = pos_bbs_std_as_tensor[:, 1] + pos_bbs_std_as_tensor[:, 3]
                    pos_bbs_quadrilateral_std_as_tensor[:, 2] = pos_bbs_std_as_tensor[:, 0]
                    pos_bbs_quadrilateral_std_as_tensor[:, 3] = pos_bbs_std_as_tensor[:, 1]
                    pos_bbs_quadrilateral_std_as_tensor[:, 4] = pos_bbs_std_as_tensor[:, 0] + pos_bbs_std_as_tensor[:, 2]
                    pos_bbs_quadrilateral_std_as_tensor[:, 5] = pos_bbs_std_as_tensor[:, 1]
                    pos_bbs_quadrilateral_std_as_tensor[:, 6] = pos_bbs_std_as_tensor[:, 0] + pos_bbs_std_as_tensor[:, 2]
                    pos_bbs_quadrilateral_std_as_tensor[:, 7] = pos_bbs_std_as_tensor[:, 1] + pos_bbs_std_as_tensor[:, 3]
                    bb_refined_std += pos_bbs_quadrilateral_std_as_tensor
                else:
                    bb_refined_std += pos_bbs_std_as_tensor  # cute bug added expanded_gt_bbox_std_as_tensor...

            if quadrilateral:
                for iter in range(bb_refined_std.shape[0]):
                    bb_refined_std_pol = Polygon(bb_refined_std[iter,:].reshape(-1,2)).convex_hull
                    pos_bbs_std_as_tensor_pol = Polygon(pos_bbs_quadrilateral_std_as_tensor[iter,:].reshape(-1,2)).convex_hull
                    expanded_gt_bbox_std_as_tensor_pol = Polygon(expanded_gt_bbox_std_as_tensor[iter,:].reshape(-1,2)).convex_hull
                    if iter == 0:
                        iou_scores = torch.as_tensor([bb_refined_std_pol.intersection(expanded_gt_bbox_std_as_tensor_pol).area / bb_refined_std_pol.union(expanded_gt_bbox_std_as_tensor_pol).area])
                        sample_ious = torch.as_tensor([pos_bbs_std_as_tensor_pol.intersection(expanded_gt_bbox_std_as_tensor_pol).area / pos_bbs_std_as_tensor_pol.union(expanded_gt_bbox_std_as_tensor_pol).area])
                    else:
                        iou_scores = torch.cat((iou_scores, torch.as_tensor([bb_refined_std_pol.intersection(expanded_gt_bbox_std_as_tensor_pol).area / bb_refined_std_pol.union(expanded_gt_bbox_std_as_tensor_pol).area])))
                        sample_ious = torch.cat((sample_ious, torch.as_tensor([pos_bbs_std_as_tensor_pol.intersection(expanded_gt_bbox_std_as_tensor_pol).area / pos_bbs_std_as_tensor_pol.union(expanded_gt_bbox_std_as_tensor_pol).area])))
            else:
                iou_scores = torch_overlap_ratio(bb_refined_std, expanded_gt_bbox_std_as_tensor)
                sample_ious = torch_overlap_ratio(pos_bbs_std_as_tensor, expanded_gt_bbox_std_as_tensor)

            if k == 0:
                cycle_iou_results = iou_scores.clone().cpu().data
                cycle_iou_samples = sample_ious.clone().cpu().data
            else:
                cycle_iou_results = torch.cat((cycle_iou_results, iou_scores.clone().cpu().data))
                cycle_iou_samples = torch.cat((cycle_iou_samples, sample_ious.clone().cpu().data))
            sum_ious = iou_scores.sum().item()
            sum_sample_iou = sample_ious.sum().item()

            # -----------------

            if num_ious == 0:
                print("skipped")
                # problematic - this will skew cycle avergae precision calculation
                continue

            n_prev = iou_per_seq[k,0]
            # before regnet
            iou_per_seq[k, 1] = (iou_per_seq[k, 1] * n_prev + sum_sample_iou) / (n_prev + num_ious)
            iou_per_seq[k, 2] = np.sqrt(((iou_per_seq[k, 2] ** 2) * (n_prev - 1) + (sample_ious.std().item() ** 2) * (num_ious - 1)) / (n_prev + num_ious - 1)) # before regnet
            iou_per_seq[k, 0] = iou_per_seq[k, 0] + num_ious
            # after regnet
            prec_per_seq[k, 1] = (prec_per_seq[k, 1] * n_prev + sum_ious) / (n_prev + num_ious)
            prec_per_seq[k, 2] = np.sqrt(((prec_per_seq[k, 2] ** 2) * (n_prev - 1) + (iou_scores.std().item() ** 2) * (num_ious - 1)) / (n_prev + num_ious - 1))  # before regnet
            prec_per_seq[k, 0] = prec_per_seq[k, 0] + num_ious

            toc[k] = time.time() - tic  # time it took to train over current sequqnce

            print("seq %3d/%3d - %-20s), IoU %.5f:%.5f --> %.5f:%.5f, Time %.3f" % \
                  (k, K - 1, seqnames[k], iou_per_seq[k, 1], iou_per_seq[k, 2], prec_per_seq[k, 1], prec_per_seq[k, 2], toc[k]))

        print("total IoU (avg:std): (%.5f:%.5f) --> (%.5f:%.5f)" % (cycle_iou_samples.mean(), cycle_iou_samples.std(),
                                                                   cycle_iou_results.mean(), cycle_iou_results.std()))
        print('median time per sequence: %.3f' % np.median(toc))
        print('expected time per cycle: %.3f' % (np.median(toc)*K))

    print("best training precision (avg:std): (%.5f:%.5f)" % (best_prec, best_std))
    return

# def train_regnet(md_model_path, init_regnet=False, lr=0.0001, translate_mode=True, direct_loss=True, fewer_sequences=False, dont_save=False, saved_state=None, first_cycle=0, fixed_learning_rate=False):
def train_regnet(md_model_path, train_regnet_opts=train_regnet_opts_defaults):
    generate_std = False
    # use_gpu = opts['use_gpu']
    img_size_std = opts['img_size']
    tic_save = time.time()

    init_regnet = train_regnet_opts['init_regnet']
    lr = train_regnet_opts['lr']
    translate_mode = train_regnet_opts['translate_mode']
    direct_loss = train_regnet_opts['direct_loss']
    # fewer_sequences = train_regnet_opts['fewer_sequences']
    dont_save = train_regnet_opts['dont_save']
    saved_state = train_regnet_opts['saved_state']
    first_cycle = train_regnet_opts['first_cycle']
    fixed_learning_rate = train_regnet_opts['fixed_learning_rate']
    limit_frame_per_seq = train_regnet_opts['limit_frame_per_seq']
    regnet_model_path = train_regnet_opts['regnet_model_path']
    pre_generate = train_regnet_opts['pre_generate']
    blackout = train_regnet_opts['blackout']
    training_indices = train_regnet_opts['training_indices']
    data_path = train_regnet_opts['data_path']
    img_home = train_regnet_opts['img_home']

    if pre_generate:
        generate_std = False  # because no need to save much time
        if blackout:
            seq_regions_dir = "seq_regions_blackout"
        else:
            seq_regions_dir = "seq_regions"
        os.makedirs(seq_regions_dir, exist_ok=True)

    # ------------

    # Init bb refeinement model #
    # regnet_model_path = '../models/regnet.pth'
    optimizer_state_loaded = False
    if init_regnet or not os.path.isfile(regnet_model_path):
        regnet_model = RegNet(translate_mode=translate_mode, state=None)
        regnet_model.train()
        if torch.cuda.is_available():
            regnet_model = regnet_model.cuda()
        optimizer = optim.Adam(regnet_model.parameters(), lr=lr)
        best_prec = 0.0
        best_std = 0.0
        last_training_cycle_idx = -1
        loss_graphs = np.array([])
        log_messages = []
        lr_history = np.empty((2, 0))
    else:  # load regnet from saved state
        if saved_state is not None:
            state = saved_state
        else:
            state = torch.load(regnet_model_path)
        # print('model saved layers:')
        # for key, value in state['RegNet_layers'].items():
        #     if key.split('.')[-1] == 'weight':
        #         print('  ' + key.split('.')[0] + ': ' + str(value.shape[1]) + ' --> ' + str(value.shape[0]))
        regnet_model = RegNet(translate_mode=translate_mode, state=state)
        # regnet_model.load_state_dict(state['RegNet_layers'])  # done via RegNet() init function instead
        regnet_model.train()
        if torch.cuda.is_available():
            regnet_model = regnet_model.cuda()
        optimizer = optim.Adam(regnet_model.parameters(), lr=lr)
        if 'best_prec' in state.keys():
            best_prec = state['best_prec']
            # print("starting precision = %.5f" % best_prec)
        else:
            best_prec = 0.0
        if 'best_std' in state.keys():
            best_std = state['best_std']
        else:
            best_std = 0.0
        if 'translate_mode' in state.keys():
            if translate_mode is not state['translate_mode']:
                print('WARNING: translate mode mismatch - saved state vs. argument')
            translate_mode = state['translate_mode']  # we overide train_regnet input according to saved state
        if 'last_training_cycle_idx' in state.keys():
            last_training_cycle_idx = state['last_training_cycle_idx']
        else:
            last_training_cycle_idx = 49  # historical...
        if 'loss_graphs' in state.keys():
            loss_graphs = state['loss_graphs']
        else:
            loss_graphs = np.array([])
        if 'optimizer' in state.keys():
            optimizer.load_state_dict(state['optimizer'])
            optimizer_state_loaded = True
        if 'log_messages' in state.keys():
            log_messages = state['log_messages']
        else:
            log_messages = []
        if 'lr_history' in state.keys():
            lr_history = state['lr_history']
        else:
            lr_history = np.empty((2, 0))
        if 'blackout' in state.keys():
            if blackout is not state['blackout']:
                print('WARNING: blackout mode mismatch - saved state vs. argument')
            blackout = state['blackout']  # we overide train_regnet input according to saved state

    print('model architecture:')
    model_str = ''
    for key, value in regnet_model.layers.state_dict().items():
        if key.split('.')[-1] == 'weight':
            print('  ' + key.split('.')[0] + ': ' + str(value.shape[1]) + ' --> ' + str(value.shape[0]))
            model_str += str(value.shape[1]) + '-'
    model_str += str(value.shape[0])

    log_str = "starting precision (avg:std) = (%.5f:%.5f)" % (best_prec, best_std)
    print(log_str)
    datetime_str = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log_messages.append(datetime_str + ' - ' + log_str)

    # for param in regnet_model.parameters():
    #     print(param.data)

    cur_lr = lr
    # lr_marks = [[0.0, 0.93], [0.9, 0.96], [0.94, 1.0]]
    lr_marks = [[0.0, 0.5], [0.45, 1]]
    for idx in range(len(lr_marks)):
        if (best_prec >= lr_marks[idx][0]) and (best_prec <= lr_marks[idx][1]):
            lr_idx = idx
            highest_lr_idx = idx
            break
        else:
            if (not fixed_learning_rate) and (not optimizer_state_loaded):
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / (2 * (idx + 1))
                    cur_lr = cur_lr / (2 * (idx + 1))
    lr_new_history = np.array([[cur_lr], [0]])

    # -------------

    # Init MDNet model #
    md_model = MDNet(md_model_path)
    if torch.cuda.is_available():
        md_model = md_model.cuda()

    # -------------

    if quadrilateral:  # this is a bad hack, should be temporary........... !!!!!!!!!!!!
        with open(data_path_vot_quadrilateral, 'rb') as fp:
            data = pickle.load(fp)
    else:
        with open(data_path, 'rb') as fp:
            data = pickle.load(fp)

    # K is the number of sequences (58)
    # [0] -> starting index, [1] -> number of sequences
    training_indices[0] = max(training_indices[0], 0)
    training_indices[0] = min(len(data)-1, training_indices[0])
    K = len(data) - training_indices[0]
    if training_indices[1] > 0:
        K = min(training_indices[1], K)
    # if fewer_sequences > 0:
    #     K = min(len(data), fewer_sequences)
    # else:
    #     K = len(data)

    dataset = [None] * K
    seqnames = [None] * K
    frame_features = [None] * K
    if pre_generate:
        all_feats_bb = [None] * K
    num_frames = [None] * K
    for j, seqname in enumerate(data):
        if j < training_indices[0]:
            continue
        seqname.replace("\\", '/')
        print('preparing sequence %d - ' % j + seqname)
        k = j - training_indices[0]

        seqnames[k] = seqname
        img_list = data[seqname]['images']
        img_list = [st.replace('\\', '/') for st in img_list]
        gt = data[seqname]['gt']  # gt is a ndarray of rectangles
        if quadrilateral:
            gt_origin = data[seqname]['gt_origin']
        if sub_sample:  # thin down their number
            img_list = img_list[0::2]
            gt = gt[0::2]
            if quadrilateral:
                gt_origin = gt_origin[0::2]
        if limit_frame_per_seq:
            num_frames[k] = min(len(img_list), num_frames_per_seq)
            img_list = img_list[:num_frames[k]]
            gt = gt[:num_frames[k]]
            if quadrilateral:
                gt_origin = gt_origin[:num_frames[k]]
        else:
            num_frames[k] = len(img_list)
        if OS == 'Windows':
            img_dir = os.path.join(img_home, str(PureWindowsPath(seqname)))
        else:
            img_dir = os.path.join(img_home, seqname)
        # every sequence gets the frames order randomly permutated
        # this also happens during training every time frame are exhausted

        if pre_generate:
            os.makedirs(os.path.join(seq_regions_dir, *(seqname.split('/')[:-1])), exist_ok=True)
            seq_regions_filename = os.path.join(seq_regions_dir, *seqname.split('/')) + '.pth'
        else:
            seq_regions_filename = ''
        # if OS == 'Windows':
        #     seq_regions_filename = os.path.join(seq_regions_dir, "\\".join(seqname.split('/'))) + '.pth'
        # else:
        #     seq_regions_filename = os.path.join(seq_regions_dir, "/".join(seqname.split('/'))) + '.pth'
        dataset[k] = PosRegionDataset(img_dir, img_list, gt, opts, torch.cuda.is_available(),
                                      generate_std=generate_std, pre_generate=pre_generate,
                                      seq_regions_filename=seq_regions_filename, blackout=blackout, gt_origin=gt_origin)
        # dataset[k] = FCDataset(img_dir, img_list, gt, opts)

        img_path_list = np.array([os.path.join(img_dir, img) for img in img_list])
        frame_features[k] = {}
        for img_path in img_path_list:
            image = Image.open(img_path).convert('RGB')
            curr_frame_features = forward_samples(md_model, image, np.array([[0, 0, image.size[0], image.size[1]]]), is_cuda=torch.cuda.is_available())
            frame_features[k][img_path] = curr_frame_features

        if pre_generate:

            # using forward_samples on pos_bbs called crop image with valid==False ...
            # if large_memory_gpu_for_pre_gen:
            if dataset[k].pos_regions_path is '':
                pos_regions = dataset[k].pos_regions
            else:
                saved_regions = torch.load(dataset[k].pos_regions_path)
                pos_regions = saved_regions['pos_regions']
            for off_start in range(0,len(pos_regions),32):
                off_end = min(off_start+32,len(pos_regions))
                all_feats_bb_k_batch = forward_regions(md_model, pos_regions[off_start:off_end], is_cuda=True)
                if off_start==0:
                    all_feats_bb_k = all_feats_bb_k_batch.cpu()
                else:
                    all_feats_bb_k = torch.cat((all_feats_bb_k,all_feats_bb_k_batch.cpu()),dim=0)
            all_feats_bb[k] = all_feats_bb_k
            # else:
            #     # we can loop over chunks of the regions to till use GPU but in quants
            #     md_model = md_model.cpu()
            #     all_feats_bb[k] = forward_regions(md_model, dataset[k].pos_regions, is_cuda=False)
            #     if torch.cuda.is_available():
            #         md_model = md_model.cuda()

            # draft of code alternative to training-time expansion loops
            # gt_bbox_std_as_tensor = dataset[k].gt_std_as_tensor
            # for iterator, frame_path in enumerate(img_path_list):
            #     num_examples = dataset[k].batch_pos
            #     feats_frame = frame_features[k][frame_path]
            #     if iterator == 0:
            #         all_feats_frame = feats_frame.repeat(num_examples, 1)
            #         expanded_gt_bbox_std_as_tensor = gt_bbox_std_as_tensor[iterator].repeat(num_examples, 1)
            #     else:
            #         all_feats_frame = torch.cat((all_feats_frame, feats_frame.repeat(num_examples, 1)))
            #         expanded_gt_bbox_std_as_tensor = torch.cat((expanded_gt_bbox_std_as_tensor, gt_bbox_std_as_tensor[iterator].repeat(num_examples, 1)))

        if k + 1 == K:
            break

    print('loaded %d sequences' % K)
    print('frames per sequence, avg: %.4g, med: %.4g' % (np.mean(num_frames), np.median(num_frames)))

    # --------------

    RefinementLoss = nn.MSELoss(reduction='mean')
    # RefinementLoss = nn.L1Loss()
    # optimizer = optim.Adam(regnet_model.parameters(), lr=lr)

    regnet_model.zero_grad()

    log_str = "starting cycle %d" % first_cycle
    print(log_str)
    datetime_str = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log_messages.append(datetime_str + ' - ' + log_str)

    zig = False
    if pre_generate:
        batch_offset = 0
        batch_indices = np.random.permutation(opts['batch_pos'])
        samples_per_frame = opts['batch_pos'] // opts['batch_frames']
    precision_history = np.zeros((K + 1, opts['n_cycles'] - first_cycle))
    avg_fig_num = 1
    top_fig_num = 2
    bottom_fig_num = 3
    med_fig_num = 4
    # cycle <> epoch, because each cycle we cycle through all sequences, but work only on a batch of frames from each
    best_cycle = first_cycle - 1
    for i in range(first_cycle, opts['n_cycles']):
        print('')
        cyc_num = (i-first_cycle) + (last_training_cycle_idx+1)
        print("==== Start Cycle %d (%d) ====" % (i, cyc_num))
        print('learning rate: %.5g' % cur_lr)
        # lr_new_history[:,-1] = np.array([cur_lr, i])
        lr_new_history[1, -1] += 1

        # ---------------

        k_list = np.random.permutation(K)  # reorder training sequences each cycle

        if pre_generate:
            next_batch_offset = min(batch_offset + samples_per_frame, opts['batch_pos'])
            idx_offsets = batch_indices[batch_offset:next_batch_offset]
            if len(idx_offsets) < samples_per_frame:
                batch_indices = np.random.permutation(opts['batch_pos'])
                next_batch_offset = samples_per_frame - len(idx_offsets)
                idx_offsets = np.concatenate((idx_offsets, batch_indices[:next_batch_offset]))
            batch_offset = next_batch_offset

        # ---------------

        curr_cycle_prec_per_seq = np.zeros(K)
        curr_cycle_seq_sample_iou = np.zeros(K)
        toc = np.zeros(K)
        # we iterate over sequences
        # from each sequence we will extract the next batch of frames to train in this epoch
        # think of this as BFS training flow while DFS would have been to train over each sequence fully before moving
        # on to the next sequence
        for j, k in enumerate(k_list):
            tic = time.time()

            if pre_generate:
                frame_numbers = dataset[k].next()
                # samples_per_frame = dataset[k].batch_pos // dataset[k].batch_frames
                num_example_list = [samples_per_frame] * len(frame_numbers)
            else:
                pos_regions, pos_bbs, num_example_list, frame_numbers = dataset[k].next()

            # -----------------

            num_ious = sum(num_example_list)
            image_path_list = dataset[k].img_list[frame_numbers]
            image_size = dataset[k].image_size
            if quadrilateral:
                gt_bbox_std_as_tensor = dataset[k].gt_origin_std_as_tensor[frame_numbers]
            else:
                gt_bbox_std_as_tensor = dataset[k].gt_std_as_tensor[frame_numbers]

            if pre_generate:
                for iterator, frame_number in enumerate(frame_numbers):

                    idx_repeated_start = frame_number * opts['batch_pos']
                    idx_repeated = idx_repeated_start + idx_offsets

                    if iterator == 0:
                        feats_bbs = all_feats_bb[k][idx_repeated]
                        pos_bbs_std_as_tensor = dataset[k].pos_bbs_std_as_tensor[idx_repeated]
                    else:
                        feats_bbs = torch.cat((feats_bbs, all_feats_bb[k][idx_repeated]))
                        pos_bbs_std_as_tensor = torch.cat((pos_bbs_std_as_tensor, dataset[k].pos_bbs_std_as_tensor[idx_repeated]))
                if torch.cuda.is_available():
                    feats_bbs = feats_bbs.cuda()
            else:
                feats_bbs = forward_regions(md_model, pos_regions, is_cuda=torch.cuda.is_available())
                # using forward_samples on pos_bbs called crop image with valid==False ...

            for iterator, (num_examples, frame_path) in enumerate(zip(num_example_list, image_path_list)):
                feats_frame = frame_features[k][frame_path]
                if iterator == 0:
                    expanded_feats_frames = feats_frame.repeat(num_examples, 1)
                    expanded_gt_bbox_std_as_tensor = gt_bbox_std_as_tensor[iterator].repeat(num_examples,1)
                else:
                    expanded_feats_frames = torch.cat((expanded_feats_frames, feats_frame.repeat(num_examples, 1)))
                    expanded_gt_bbox_std_as_tensor = torch.cat((expanded_gt_bbox_std_as_tensor, gt_bbox_std_as_tensor[iterator].repeat(num_examples, 1)))

            if not pre_generate:
                pos_bbs_std = pos_bbs
                if not generate_std:
                    # assuming all frames in given sequence have the same size
                    pos_bbs_std[:,0] = pos_bbs[:,0] * img_size_std / image_size[0]
                    pos_bbs_std[:,2] = pos_bbs[:,2] * img_size_std / image_size[0]
                    pos_bbs_std[:,1] = pos_bbs[:,1] * img_size_std / image_size[1]
                    pos_bbs_std[:,3] = pos_bbs[:,3] * img_size_std / image_size[1]

                if torch.cuda.is_available():
                    pos_bbs_std_as_tensor = torch.Tensor(pos_bbs_std).cuda()
                else:
                    pos_bbs_std_as_tensor = torch.Tensor(pos_bbs_std)

            net_input = torch.cat((feats_bbs, expanded_feats_frames, pos_bbs_std_as_tensor), dim=1)

            bb_refined_std = regnet_model(net_input)
            if translate_mode:
                if quadrilateral:
                    pos_bbs_quadrilateral_std_as_tensor = pos_bbs_std_as_tensor.repeat(1, 2)
                    pos_bbs_quadrilateral_std_as_tensor[:, 0] = pos_bbs_std_as_tensor[:, 0]
                    pos_bbs_quadrilateral_std_as_tensor[:, 1] = pos_bbs_std_as_tensor[:, 1] + pos_bbs_std_as_tensor[:, 3]
                    pos_bbs_quadrilateral_std_as_tensor[:, 2] = pos_bbs_std_as_tensor[:, 0]
                    pos_bbs_quadrilateral_std_as_tensor[:, 3] = pos_bbs_std_as_tensor[:, 1]
                    pos_bbs_quadrilateral_std_as_tensor[:, 4] = pos_bbs_std_as_tensor[:, 0] + pos_bbs_std_as_tensor[:, 2]
                    pos_bbs_quadrilateral_std_as_tensor[:, 5] = pos_bbs_std_as_tensor[:, 1]
                    pos_bbs_quadrilateral_std_as_tensor[:, 6] = pos_bbs_std_as_tensor[:, 0] + pos_bbs_std_as_tensor[:, 2]
                    pos_bbs_quadrilateral_std_as_tensor[:, 7] = pos_bbs_std_as_tensor[:, 1] + pos_bbs_std_as_tensor[:, 3]
                    bb_refined_std += pos_bbs_quadrilateral_std_as_tensor
                else:
                    bb_refined_std += pos_bbs_std_as_tensor  # cute bug added expanded_gt_bbox_std_as_tensor...

            if quadrilateral:
                for iter in range(bb_refined_std.shape[0]):
                    bb_refined_std_pol = Polygon(bb_refined_std[iter,:].reshape(-1,2)).convex_hull
                    # if len(bb_refined_std_pol.exterior.coords.xy[0]) < 5:
                    #     # this is a shitty hack
                    #     # I would prefer finding which point is "inside" the rectangle
                    #     # then find closest point on one of its line
                    #     # lin = pol.exterior.coords
                    #     # list(lin.coords)
                    #     # https://stackoverflow.com/questions/33311616/find-coordinate-of-the-closest-point-on-polygon-in-shapely
                    #     bb_refined_std_pol = bb_refined_std_pol.minimum_rotated_rectangle
                    # bb_refined_std[iter,:] = torch.as_tensor(bb_refined_std_pol.exterior.coords.xy).transpose(0,1).reshape(1,10)[0,:8]
                    pos_bbs_std_as_tensor_pol = Polygon(pos_bbs_quadrilateral_std_as_tensor[iter,:].reshape(-1,2)).convex_hull
                    expanded_gt_bbox_std_as_tensor_pol = Polygon(expanded_gt_bbox_std_as_tensor[iter,:].reshape(-1,2)).convex_hull
                    # if len(expanded_gt_bbox_std_as_tensor_pol.exterior.coords.xy[0]) < 5:
                    #     expanded_gt_bbox_std_as_tensor_pol = expanded_gt_bbox_std_as_tensor_pol.minimum_rotated_rectangle
                    # expanded_gt_bbox_std_as_tensor[iter, :] = torch.as_tensor(expanded_gt_bbox_std_as_tensor_pol.exterior.coords.xy).transpose(0,1).reshape(1, 10)[0, :8]
                    if iter == 0:
                        iou_scores = torch.as_tensor([bb_refined_std_pol.intersection(expanded_gt_bbox_std_as_tensor_pol).area / bb_refined_std_pol.union(expanded_gt_bbox_std_as_tensor_pol).area])
                        sample_ious = torch.as_tensor([pos_bbs_std_as_tensor_pol.intersection(expanded_gt_bbox_std_as_tensor_pol).area / pos_bbs_std_as_tensor_pol.union(expanded_gt_bbox_std_as_tensor_pol).area])
                    else:
                        iou_scores = torch.cat((iou_scores, torch.as_tensor([bb_refined_std_pol.intersection(expanded_gt_bbox_std_as_tensor_pol).area / bb_refined_std_pol.union(expanded_gt_bbox_std_as_tensor_pol).area])))
                        sample_ious = torch.cat((sample_ious, torch.as_tensor([pos_bbs_std_as_tensor_pol.intersection(expanded_gt_bbox_std_as_tensor_pol).area / pos_bbs_std_as_tensor_pol.union(expanded_gt_bbox_std_as_tensor_pol).area])))
            else:
                iou_scores = torch_overlap_ratio(bb_refined_std, expanded_gt_bbox_std_as_tensor)
                sample_ious = torch_overlap_ratio(pos_bbs_std_as_tensor, expanded_gt_bbox_std_as_tensor)

            if j==0:
                cycle_iou_results = iou_scores.clone().cpu().data
                cycle_iou_samples = sample_ious.clone().cpu().data
            else:
                cycle_iou_results = torch.cat((cycle_iou_results, iou_scores.clone().cpu().data))
                cycle_iou_samples = torch.cat((cycle_iou_samples, sample_ious.clone().cpu().data))
            sum_ious = iou_scores.sum().item()
            sum_sample_iou = sample_ious.sum().item()

            if direct_loss:
                # loss = RefinementLoss(bb_refined_std, gt_bb_std_as_tensor)
                loss = RefinementLoss(10 * (bb_refined_std - expanded_gt_bbox_std_as_tensor) / img_size_std, torch.zeros_like(expanded_gt_bbox_std_as_tensor))
            else:
                raise Exception('need to indirect loss work with batch')

            # -----------------

            # idx = 0
            # sum_ious = 0
            # sum_sample_iou = 0
            # num_ious = 0
            # total_loss = 0
            # num_loss_steps = 0

            # we want SGD so we reset grads before each new batch
            # if pretrain_opts['large_memory_gpu'] or not torch.cuda.is_available():
            #     regnet_model.zero_grad()

            # # iterting over frames in current batch
            # for num_examples, frame, gt_bb in zip(num_example_list, image_list, gt_bbox_list):
            #     num_ious += num_examples
            #
            #     # frame = Variable(frame)
            #     # gt_bb = Variable(gt_bb)
            #
            #     # resizing (scaling) GT-BB to fit standard 107x107 frame
            #     gt_bb_std = gt_bb
            #     gt_bb_std[0] = gt_bb[0] * img_size_std / frame.size[0]
            #     gt_bb_std[2] = gt_bb[2] * img_size_std / frame.size[0]
            #     gt_bb_std[1] = gt_bb[1] * img_size_std / frame.size[1]
            #     gt_bb_std[3] = gt_bb[3] * img_size_std / frame.size[1]
            #     if torch.cuda.is_available():
            #         gt_bb_std_as_tensor = torch.from_numpy(gt_bb_std).float().cuda()
            #     else:
            #         gt_bb_std_as_tensor = torch.from_numpy(gt_bb_std).float()
            #
            #     # iterating over examples extracted for this frame
            #     for region, bb in zip(pos_regions[idx:idx+num_examples], pos_bbs[idx:idx+num_examples]):
            #         # with torch.no_grad():
            #         feats_bb = forward_samples(md_model, frame, np.array([bb]), is_cuda=torch.cuda.is_available())
            #         feats_frame = forward_samples(md_model, frame, np.array([[0, 0, frame.size[0], frame.size[1]]]), is_cuda=torch.cuda.is_available())
            #
            #         # bb_std = np.array(bb)
            #         bb_std = bb
            #         bb_std[0] = bb[0] * img_size_std / frame.size[0]
            #         bb_std[2] = bb[2] * img_size_std / frame.size[0]
            #         bb_std[1] = bb[1] * img_size_std / frame.size[1]
            #         bb_std[3] = bb[3] * img_size_std / frame.size[1]
            #
            #         if torch.cuda.is_available():
            #             bb_std_as_tensor = torch.Tensor(np.array([bb_std])).cuda()
            #         else:
            #             bb_std_as_tensor = torch.Tensor(np.array([bb_std]))
            #
            #         net_input = torch.cat((feats_bb, feats_frame, bb_std_as_tensor), dim=1)
            #         # if torch.cuda.is_available():
            #         #     # net_input = net_input.to(device=device)
            #         #     net_input = net_input.cuda()
            #
            #         bb_refined_std = regnet_model(net_input)
            #         bb_refined_std = bb_refined_std[0,:]
            #         if translate_mode:
            #             bb_refined_std += bb_std_as_tensor[0,:]
            #
            #         # iou_score = overlap_ratio(bb_refined, gt_bb)[0]
            #         iou_score = torch_overlap_ratio(bb_refined_std, gt_bb_std_as_tensor)
            #         sum_ious += iou_score.item()
            #         sample_iou = torch_overlap_ratio(bb_std_as_tensor[0,:], gt_bb_std_as_tensor)
            #         # print("sample iou: %.5f" % sample_iou.item())
            #         sum_sample_iou += sample_iou.item()
            #
            #         if direct_loss:
            #             # loss = RefinementLoss(bb_refined_std, gt_bb_std_as_tensor)
            #             loss = RefinementLoss(10*(bb_refined_std - gt_bb_std_as_tensor)/img_size_std, torch.zeros_like(gt_bb_std_as_tensor))
            #         else:
            #             # ----- iou loss -----------
            #
            #             # # iou_score = overlap_ratio(bb_refined, gt_bb)[0]
            #             # iou_score = torch_overlap_ratio(bb_refined_std, gt_bb_std_as_tensor)
            #             # sum_ious += iou_score.item()
            #
            #             iou_target = torch.ones_like(iou_score)
            #             if torch.cuda.is_available():
            #                 # iou_target = iou_target.to(device)
            #                 iou_target = iou_target.cuda()
            #
            #             iou_loss = RefinementLoss(iou_score, iou_target)
            #
            #             # ----- distance loss -----------
            #
            #             bb_refined_std_center = bb_refined_std[:2] + bb_refined_std[2:] / 2
            #             gt_bb_std_center_as_tensor = gt_bb_std_as_tensor[:2] + gt_bb_std_as_tensor[2:] / 2
            #             result_distance = torch.dist(bb_refined_std_center, gt_bb_std_center_as_tensor)
            #             result_distance_norm = result_distance / img_size_std
            #
            #             distance_target = torch.zeros_like(result_distance_norm)
            #             if torch.cuda.is_available():
            #                 distance_target = distance_target.cuda()
            #
            #             distance_loss = RefinementLoss(result_distance_norm, distance_target)
            #
            #             # ----- size loss -----------
            #
            #             bb_refined_std_size = bb_refined_std[2]*bb_refined_std[3]
            #             gt_bb_std_size = gt_bb_std_as_tensor[2]*gt_bb_std_as_tensor[3]
            #             size_relative = bb_refined_std_size / gt_bb_std_size
            #             # size_relative = torch.log(size_relative)
            #
            #             size_target = torch.ones_like(result_distance_norm)
            #             if torch.cuda.is_available():
            #                 size_target = size_target.cuda()
            #
            #             size_loss = RefinementLoss(size_relative, size_target)
            #
            #             # ----- combined loss -----------
            #
            #             if translate_mode:
            #                 loss = (distance_loss * 5) + (size_loss / 4) #+ iou_loss
            #             else:
            #                 loss = distance_loss + size_loss
            #
            #         if pretrain_opts['large_memory_gpu'] or not torch.cuda.is_available():
            #             total_loss += loss
            #         else:
            #             # we don't normalize loss because number of back-props before optim.step() isnt deterministic
            #             loss.backward()
            #             total_loss += loss.clone().cpu().data
            #
            #         num_loss_steps += 1
            #
            #     idx += num_examples

            # -----------------

            if num_ious == 0:
                print("skipped")
                # problematic - this will skew cycle avergae precision calculation
                continue

            # we want SGD so we update grads only after batch has ended
            # if not num_loss_steps == num_ious:
            #     raise Exception('sanity failed: loss_steps = %d, num_iou = %d' % (num_loss_steps, num_ious))

            # total_loss = total_loss / num_ious
            total_loss = loss  # now working with batches
            if pretrain_opts['large_memory_gpu'] or not torch.cuda.is_available():
                total_loss.backward()
            else:
                raise Exception('did not backward. indirect loss scenario not covered yet.')
            # torch.nn.utils.clip_grad_norm(regnet_model.parameters(), opts['grad_clip'])  # ??????????????
            optimizer.step()  # no need for 'with torch.no_grad():' since we use optimizer from torch.optim
            regnet_model.zero_grad()

            curr_cycle_seq_sample_iou[k] = sum_sample_iou / num_ious  # before regnet
            curr_cycle_prec_per_seq[k] = sum_ious / num_ious  # after regnet
            precision_history[k, i - first_cycle] = sum_ious / num_ious
            toc[k] = time.time() - tic  # time it took to train over current sequqnce

            # displaying stats for current sequence
            # reminder: we only processed a batch of frames from current sequence, not all of it
            if pretrain_opts['large_memory_gpu'] or not torch.cuda.is_available():
                if total_loss.dim() == 0:
                    total_loss = total_loss.data
                else:
                    total_loss = total_loss.data[0]  # what ???????????????????????????????????????????

            # print("Cycle %d (%d), [iter %d/%d] (seq %2d - %-20s), Loss %.5f, IoU %.5f --> %.5f, Time %.3f" % \
            #           (i, cyc_num, j, K-1, k, seqnames[k], total_loss, curr_cycle_seq_sample_iou[k], curr_cycle_prec_per_seq[k], toc[k]))
            print("iter %d/%d (seq %2d - %-20s), Loss %.5f, IoU %.5f --> %.5f, Time %.3f" % \
                  (j, K - 1, k, seqnames[k], total_loss, curr_cycle_seq_sample_iou[k], curr_cycle_prec_per_seq[k], toc[k]))

        # regnet refinement statistics, current cycle
        cur_regnet_prec = curr_cycle_prec_per_seq.mean()
        # cur_regnet_std = curr_cycle_prec_per_seq.std()
        cur_regnet_std = cycle_iou_results.std()

        if cur_regnet_prec < 0.00001:
            log_str = "low precision, restarting"
            print(log_str)
            datetime_str = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            log_messages.append(datetime_str + ' - ' + 'Cycle %d (%d)' % (i, cyc_num) + ' - ' + log_str)

            # this is legacy code
            # motivation: regnet used to refine IoU assuming minimal intersection.
            # return will allow reshuffle of sequences on next call, hopefully jump start the learning curve
            # this was used before realized proper (small) weight initialization and lr are instrumental
            return False, saved_state, i
        precision_history[K, i - first_cycle] = curr_cycle_prec_per_seq.mean()
        if i<2000:  # we stop updating argmin_prec after a while
            argmin_prec = precision_history[:,:i+1].mean(axis=1).argmin()
        # precision_history[K + 1, i - first_cycle] = curr_cycle_prec_per_seq[np.argsort(curr_cycle_prec_per_seq)[-(K-5):]].mean()

        fig = plt.figure(avg_fig_num)
        plt.clf()
        fig.suptitle('curr/best prec: %f/%f -- curr lr: %f -- model %s' % (cur_regnet_prec, best_prec, cur_lr, model_str))  # , fontsize=16)
        plt.plot(np.arange(last_training_cycle_idx + 1 - len(loss_graphs), last_training_cycle_idx + 1), loss_graphs)  # saved graphs
        plt.plot(np.arange(i - first_cycle + 1) + last_training_cycle_idx + 1, precision_history[K, :i - first_cycle + 1])  # average all iou
        # plt.plot(np.arange(i - first_cycle + 1) + last_training_cycle_idx + 1, precision_history[K + 1, :i - first_cycle + 1])  # average top ious

        # plt.plot(np.arange(i - first_cycle + 1) + last_training_cycle_idx + 1, precision_history[argmin_prec, :i - first_cycle + 1])  # sampled iou
        if not fixed_learning_rate:
            for idx in range(len(lr_marks)):  # plot lr change grid
                x = lr_marks[idx][1] * np.ones(i - first_cycle + 1 + len(loss_graphs))
                plt.plot(np.arange(last_training_cycle_idx + 1 - len(loss_graphs), i - first_cycle + 1 + last_training_cycle_idx + 1), x, 'r--')
                x = lr_marks[idx][0] * np.ones(i - first_cycle + 1 + len(loss_graphs))
                plt.plot(np.arange(last_training_cycle_idx + 1 - len(loss_graphs), i - first_cycle + 1 + last_training_cycle_idx + 1), x, 'b--')
                # x = lr_marks[idx][0] * np.ones(i + 1)
                # plt.plot(x, 'b--')
        plt.ylabel('average sequence precision (IoU)')
        plt.xlabel('cycle')

        plt.pause(.01)
        plt.draw()
        plt.show(block=False)

        print("Curr IoU (avg:std): (%.5f:%.5f) --> (%.5f:%.5f)" % (cycle_iou_samples.mean(), cycle_iou_samples.std(),
                                                                   cur_regnet_prec, cur_regnet_std))
        # print("Curr IoU (avg:std): (%.5f:%.5f) --> (%.5f:%.5f)" % (curr_cycle_seq_sample_iou.mean(), curr_cycle_seq_sample_iou.std(),
        #                                                            cur_regnet_prec, cur_regnet_std))
        print("Best IoU (avg:std): (%.5f:%.5f)" % (best_prec, best_std))
        print('median time per sequence: %.3f' % np.median(toc))
        print('expected time per cycle: %.3f' % (np.median(toc)*K))

        if cur_regnet_prec > lr_marks[lr_idx][1]:
            if not fixed_learning_rate:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/(2*(lr_idx+1))
                    cur_lr = cur_lr/(2*(lr_idx+1))
                    lr_new_history = np.concatenate((lr_new_history, np.array([[cur_lr], [0]])), axis=1)
            lr_idx = min(lr_idx + 1, len(lr_marks) - 1)  # redundant, lr_marks[-1][1] must be 1.0
            highest_lr_idx = lr_idx
        if cur_regnet_prec < lr_marks[lr_idx][0]:
            if not fixed_learning_rate:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*(2*(lr_idx+1))
                    cur_lr = cur_lr * (2 * (lr_idx + 1))
                    lr_new_history = np.concatenate((lr_new_history, np.array([[cur_lr], [0]])), axis=1)
            lr_idx = max(lr_idx - 1, 0)  # redundant, lr_marks[0][0] must be 0.0
            if lr_idx < highest_lr_idx - 1:  # identify a large fall in precision
                log_str = "too large drop in precision, restarting from last best spot"
                print(log_str)
                datetime_str = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                log_messages.append(datetime_str + ' - ' + 'Cycle %d (%d)' % (i, cyc_num) + ' - ' + log_str)

                return False, saved_state, best_cycle+1

        if cur_regnet_prec > best_prec:
            best_prec = cur_regnet_prec
            best_std = cur_regnet_std
            best_cycle = i
            if torch.cuda.is_available():
                regnet_model = regnet_model.cpu()
            saved_state = {
                'RegNet_layers': regnet_model.layers.state_dict(),
                'best_prec': best_prec,
                'best_std': best_std,
                'translate_mode': translate_mode,
                'last_training_cycle_idx': cyc_num,
                'loss_graphs': np.concatenate((loss_graphs,precision_history[K, :i - first_cycle + 1])),
                'optimizer': optimizer.state_dict(),
                'log_messages': log_messages,
                'lr_history': np.concatenate((lr_history, lr_new_history), axis=1),
                'blackout': blackout
            }

            if not dont_save:
                log_str = "Save regnet_model to %s" % regnet_model_path
                print(log_str)
                # datetime_str = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                # log_messages.append(datetime_str + ' - ' + log_str)

                # don't grind the HDD with many writes
                if time.time() - tic_save > 100:
                    # (mostly) mitigate corruption if ctrl+c while saving file
                    if zig:
                        torch.save(saved_state, regnet_model_path)
                    else:  # za
                        torch.save(saved_state, regnet_model_path + '1')
                    zig = not zig
                    tic_save = time.time()

            if torch.cuda.is_available():
                regnet_model = regnet_model.cuda()


    log_str = "best precision (avg:std): (%.5f:%.5f)" % (best_prec, best_std)
    print(log_str)
    datetime_str = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log_messages.append(datetime_str + ' - ' + log_str)

    return True, saved_state, i


def train_mdnet(data_path=data_path_default,img_home=img_home_default):
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
force_train_regnet = False
force_init_regnet = False
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # ----- mdnet related -----
    parser.add_argument('-md', '--train_mdnet', action='store_true')  # defaut: don't traing mdnet
    # ----- regnet training -----
    parser.add_argument('-rg', '--train_regnet', action='store_true')  # defaut: don't traing regnet
    parser.add_argument('-tm', '--trained_mdnet', action='store_true')  # default: use original mdnet weights (for feature extraction)
    parser.add_argument('-ir', '--init_regnet', action='store_false')  # default: init regnet (else - cont saved state)
    parser.add_argument('-lr', '--learning_rate', default=0.000001, type=float, help='learning rate')  # starting lr
    parser.add_argument('-flr', '--fixed_learning_rate', action='store_true')  # default: lr changes depending on training precision
    parser.add_argument('-il', '--indirect_loss', action='store_true')  # default: direct (mse) loss
    # parser.add_argument('-fs', '--fewer_sequences', default=0, type=int)  # default: train on all dataset
    parser.add_argument('-ti', '--training_indices', nargs=2, type=int, default=[0, 30])  # start index, number of sequences
    parser.add_argument('-ds', '--dont_save', action='store_true')  # default: save to file best precision model
    parser.add_argument('-pg', '--pre_generate', action='store_true')    # default: continuously generate frames on-the-fly
    parser.add_argument('-bl', '--blackout', action='store_true')  # default: crop and resize sample before feature extraction
    # ----- regnet validation -----
    parser.add_argument('-vrg', '--validate_regnet', action='store_true')  # defaut: don't validate regnet
    parser.add_argument('-vsb', '--validation_sub_sample', action='store_true')  # defaut: don't sub_sample
    parser.add_argument('-vsbo', '--validation_sub_sample_oddness', action='store_true')  # defaut: 0 first index
    parser.add_argument('-vi', '--validation_data_indices', nargs=2, type=int, default=[30, 15])  # start index, number of sequences
    # ----- regnet both -----
    parser.add_argument('-ot', '--translate_mode', action='store_false')  # default: output as coord shift
    parser.add_argument('-lf', '--limit_frame_per_seq', action='store_true')
    parser.add_argument('-mp', '--regnet_model_path', default='../models/regnet.pth', type=str)
    parser.add_argument('-d', '--dataset', default='VOT')

    args = parser.parse_args()

    # ------------

    dataset = args.dataset
    if dataset == 'VOT':
        data_path = data_path_vot
    elif dataset == 'OTB':
        data_path = data_path_otb
    else:
        raise Exception('unknown dataset..................')

    if OS == 'Windows':
        # usr_home = 'C:/Users/smush/'
        img_home = os.path.join(usr_home, 'downloads', dataset)
    elif OS == 'Linux':
        # usr_home = '~/'
        img_home = os.path.join(usr_home, 'MDNet-data/' + dataset)
    else:
        sys.exit("aa! errors!")

    # ------------

    if args.train_mdnet or force_train_mdnet:
        print('training mdnet')
        train_mdnet(data_path=data_path, img_home=img_home)

    # ------------

    # we need md_model for extracting features from frames
    # these features are used as inputs to the regressor network
    # we selct either the original MDNet paper model, or one that we trained
    # if args.trained_mdnet:
    #     md_model_path = pretrain_opts['model_path']
    # else:
    #     md_model_path = pretrain_opts['init_model_path']
    if args.trained_mdnet:
        md_model_path = tracking_opts['new_model_path']
    else:
        md_model_path = tracking_opts['model_path']
    if args.train_regnet or force_train_regnet:
        print('training regnet:')
        print('  learning_rate = %.5g' % (args.learning_rate))
        if args.translate_mode:
            print('  regnet output translation')
        else:
            print('  regnet outputs coordinates')
        if args.indirect_loss:
            print('  loss: combined')
        else:
            print('  loss: mse')

        # sometimes learning can't pick up, get stuck in 0 at start
        # can also happen during learning when precision suddenly crashes
        completed = False
        init_regnet = force_init_regnet or args.init_regnet
        saved_state = None
        first_cycle = 0
        train_regnet_opts = {
            'init_regnet': init_regnet,
            'lr': args.learning_rate,
            'translate_mode': args.translate_mode,
            'direct_loss': not args.indirect_loss,
            # 'fewer_sequences': args.fewer_sequences,
            'training_indices': args.training_indices,
            'dont_save': args.dont_save,
            'saved_state': saved_state,
            'first_cycle': first_cycle,
            'fixed_learning_rate': args.fixed_learning_rate,
            'limit_frame_per_seq': args.limit_frame_per_seq,
            'regnet_model_path': args.regnet_model_path,
            'pre_generate': args.pre_generate,
            'blackout': args.blackout,
            'data_path': data_path,
            'img_home': img_home
        }
        while not completed:
            # completed, saved_state, first_cycle = train_regnet(md_model_path, init_regnet=init_regnet, lr=args.learning_rate, translate_mode=args.translate_mode, direct_loss=not args.indirect_loss, fewer_sequences=args.fewer_sequences, dont_save=args.dont_save, saved_state=saved_state, first_cycle=first_cycle, fixed_learning_rate=args.fixed_learning_rate)
            completed, saved_state, first_cycle = train_regnet(md_model_path, train_regnet_opts)
            train_regnet_opts['saved_state'] = saved_state
            train_regnet_opts['first_cycle'] = first_cycle
            if saved_state is not None:
                train_regnet_opts['init_regnet']=False  # if we're recovering, then do this from last best saved state.

    if args.validate_regnet:
        validate_regnet_opts = {
            'translate_mode': args.translate_mode,
            # 'fewer_sequences': args.fewer_sequences,
            'saved_state': None,
            'limit_frame_per_seq': args.limit_frame_per_seq,
            # 'validation_data_indices': [30,15], # vot-[0, 30],[30,15],[45,13] otb-[25,24]
            'validation_data_indices': args.validation_data_indices,
            'validation_data_path': data_path,
            'validation_sub_sample': args.validation_sub_sample,
            'validation_sub_sample_oddness': args.validation_sub_sample_oddness,
            'regnet_model_path': args.regnet_model_path,
            'blackout': args.blackout,
            'img_home': img_home
        }
        validate_regnet(md_model_path, validate_regnet_opts)

