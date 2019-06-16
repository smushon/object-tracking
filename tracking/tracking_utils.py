from torch.autograd import Variable
import torch.optim as optim

import sys
sys.path.append("..")
from modules.utils import *

import torch.nn as nn
import numpy as np

import tracking.options as options
opts = options.tracking_opts


def forward_samples(model, image, samples, out_layer='conv3', is_cuda=opts['use_gpu']):
    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        # if regions.requires_grad:
        #     print('requires grad')
        regions = Variable(regions)
        if is_cuda:
        # if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)
    return feats


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


# tensor
class MyIoULoss2(nn.Module):
    def __init__(self):
        super(MyIoULoss2, self).__init__()

    def forward(self, pos_scores, neg_scores):

        pos_ious_loss = -0.25 * torch.pow(1 - pos_scores, 2) * pos_scores.log()
        neg_ious_loss = -0.5 * torch.pow(1 - neg_scores, 2) * neg_scores.log()

        loss = pos_ious_loss.sum() + neg_ious_loss.sum()
        return loss


# numpy array
class MyIoULoss(nn.Module):
    def __init__(self):
        super(MyIoULoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        # loss = np.mean(scores - target)

        pos_ious_loss = -0.25 * np.power(1 - pos_scores, 2) * np.log(pos_scores)
        neg_ious_loss = -0.5 * np.power(1 - neg_scores, 2) * np.log(neg_scores)
        loss = np.sum(pos_ious_loss) + np.sum(neg_ious_loss)

        return loss


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4',
          iou_loss=MyIoULoss(), pos_ious=[], neg_ious=[], loss_index=1):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while len(pos_idx) < batch_pos * maxiter:
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while len(neg_idx) < batch_neg_cand * maxiter:
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        #####################
        # print(type(pos_ious))
        # print(pos_ious[0])
        # print(type(pos_cur_idx.cpu()))
        # print(pos_cur_idx.cpu()[0])
        # print(pos_ious[pos_cur_idx.cpu()][0])
        # print('============')
        batch_pos_ious = pos_ious[pos_cur_idx.cpu()]
        batch_neg_ious = neg_ious[neg_cur_idx.cpu()]
        #####################

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()

        #####################
        # hard negative mining
        batch_neg_ious = batch_neg_ious[batch_neg_ious.argsort()[-batch_neg:]]
        #####################

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        score = torch.cat((pos_score,neg_score),dim=0)

        pos_target = np.ones(pos_score.shape[0], dtype=int)
        neg_target = np.zeros(neg_score.shape[0], dtype=int)
        target = np.hstack((pos_target,neg_target))
        target = torch.from_numpy(target)
        target = Variable(target)

        if opts['use_gpu']:
            target = target.cuda()

        # optimize

        loss = criterion(score, target)

        #############
        pos_ious_loss = -0.25 * np.power(1 - batch_pos_ious, 2) * np.log(batch_pos_ious)
        neg_ious_loss = -0.5 * np.power(1 - batch_neg_ious, 2) * np.log(batch_neg_ious)
        ious_loss = np.sum(pos_ious_loss) + np.sum(neg_ious_loss)
        # batch_pos_ious_tensor = torch.from_numpy(batch_pos_ious)
        # batch_neg_ious_tensor = torch.from_numpy(batch_neg_ious)
        # ious_loss = iou_loss(batch_pos_ious_tensor, batch_neg_ious_tensor)

        # print(ious_loss)
        # print(loss)

        if loss_index==2:
            loss = 0.5*(loss+ious_loss)
        # loss = ious_loss
        #############

        model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()

        # print "Iter %d, Loss %.4f" % (iter, loss.data[0])

