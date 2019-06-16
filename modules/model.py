import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

from options import *


def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child._parameters.items():
        # for k,p in child._parameters.iteritems():
            if p is None: continue
            
            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k
            
            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


#####################################
class FlattenLayer(torch.nn.Module):
    def __init__(self, *args):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.)

        y = m.in_features
        # m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data.fill_(0)


        # torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)


class RegNet(torch.nn.Module):
    # def __init__(self, *args):
    def __init__(self, translate_mode=True, model_path=None, image_size=107):
        super(RegNet, self).__init__()

        input_layer_size = 2 * 4608 + 4  # images features, crop features, BB coordinates
        hidden_layer_size = 50  # 512

        self.translate_mode = translate_mode
        self.image_size = image_size
        self.layers = nn.Sequential(OrderedDict([
            ('flatten',  FlattenLayer()),
            ('fc1', nn.Sequential(nn.Linear(input_layer_size, hidden_layer_size),
                                  nn.LeakyReLU())),
            ('fc2', nn.Sequential(nn.Linear(hidden_layer_size, 4)))
        ]))

        if (model_path is None) or (not os.path.isfile(model_path)):
            self.layers.apply(init_weights)
            # nn.init.kaiming_normal_(self.model.fc1.weight)
            # nn.init.constant_(self.model.fc1.bias, 0.)
            # nn.init.kaiming_normal_(self.model.fc2.weight)
            # nn.init.constant_(self.model.fc2.bias, 0.)
        else:
            if os.path.splitext(model_path)[1] == '.pth':
                states = torch.load(model_path)
                self.layers.load_state_dict(states['RegNet_layers'])
            else:
                raise RuntimeError("unused model format: %s" % (model_path))

    # x is a BB, output if a refined BB
    def forward(self, x):
        if self.translate_mode:
            input_bb = x.data[0,-4:].clone()  # assuming single frame in batch.... I need to generalize this !!!!!!!!!
        for name, module in self.layers.named_children():
            x = module(x)
        # x = (x1, y1, width, height)

        x = x[0]  # assuming single frame in batch.... I need to generalize this !!!!!!!!!

        if self.translate_mode:
            x += input_bb

        # ----- crop ------
        # we assume object is in frame and just require fine-tuning, so x should also be in frame
        # we won't return error if x is out of frame.
        min_bb_size = 2

        # x1,y1 must be within frame boundries
        x[0] = min(self.image_size-1-min_bb_size, x[0])
        x[0] = max(0, x[0])
        x[1] = min(self.image_size-1-min_bb_size, x[1])
        x[1] = max(0, x[1])

        # height/width can't be negative
        x[2] = max(0+min_bb_size, x[2])
        x[3] = max(0+min_bb_size, x[3])

        # height/width can't be too large
        # i.e. x2,y2 can't extend beyond frame edges
        if x[0] + x[2] > self.image_size:
            x[2] = self.image_size - x[0]
            if x[2] < min_bb_size:
                x[0] -= (min_bb_size - x[2])
        if x[1] + x[3] > self.image_size:
            x[3] = self.image_size - x[1]
            if x[3] < min_bb_size:
                x[1] -= (min_bb_size - x[3])

        if self.translate_mode:
            x -= input_bb

        return x

#####################################


#####################################
class IoUPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        scores = None
        return scores


class IoUPredModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.do = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        scores = self.fc(self.do(x))
        return scores
#####################################


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU())),
                ('fc4',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()))]))
        
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), 
                                                     nn.Linear(512, 2)) for _ in range(K)])

        # print('loading trained model')
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
 
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        #
        # forward model from in_layer to out_layer

        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.view(x.size(0),-1)
                if name == out_layer:
                    return x
        
        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x)
    
    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)
    
    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]
        
        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])

    

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
 
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]
        
        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):

        # returns how many (percentage) of the pos scores are in the top len(pos_scores) amongest all scores
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)  # concatenate pos- and neg- scores
        topk = torch.topk(scores, pos_score.size(0))[1]  # indices of topk |pos_scores| pos- and neg- scores
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)

        #######################
        if prec.dim() == 0:
            return prec.data
        else:
            return prec.data[0]
        #######################

