from collections import OrderedDict

import torch

tracking_opts = OrderedDict()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    tracking_device = torch.device('cuda:0')
    total_mem = torch.cuda.get_device_properties(tracking_device).total_memory
    if total_mem > 2500000000:
        tracking_opts['use_gpu'] = True
        # print('tracking using gpu')
    else:
        tracking_opts['use_gpu'] = False
        # print('tracking using cpu')
        tracking_device = 'cpu'
else:
    tracking_opts['use_gpu'] = False
    # print('tracking using cpu')
    tracking_device = 'cpu'

# tracking_opts['use_gpu'] = True  ###################3 hack for debug #######################
# tracking_device = torch.device('cuda:0') ###################3 hack for debug #######################

tracking_opts['model_path'] = '../models/mdnet_vot-otb.pth'
tracking_opts['new_model_path'] = '../models/mdnet_vot-otb_new.pth'
tracking_opts['img_size'] = 107
tracking_opts['padding'] = 16

tracking_opts['batch_pos'] = 32
tracking_opts['batch_neg'] = 96
tracking_opts['batch_neg_cand'] = 1024
tracking_opts['batch_test'] = 256

tracking_opts['n_samples'] = 256
tracking_opts['trans_f'] = 0.6
tracking_opts['scale_f'] = 1.05
tracking_opts['trans_f_expand'] = 1.5

tracking_opts['n_bbreg'] = 1000
tracking_opts['overlap_bbreg'] = [0.6, 1]
tracking_opts['scale_bbreg'] = [1, 2]

tracking_opts['lr_init'] = 0.0001
tracking_opts['maxiter_init'] = 30
tracking_opts['n_pos_init'] = 500
tracking_opts['n_neg_init'] = 5000
tracking_opts['overlap_pos_init'] = [0.7, 1]
tracking_opts['overlap_neg_init'] = [0, 0.5]

tracking_opts['lr_update'] = 0.0002
tracking_opts['maxiter_update'] = 15
tracking_opts['n_pos_update'] = 50
tracking_opts['n_neg_update'] = 200
tracking_opts['overlap_pos_update'] = [0.7, 1]
tracking_opts['overlap_neg_update'] = [0, 0.3]

tracking_opts['success_thr'] = 0
tracking_opts['n_frames_short'] = 20
tracking_opts['n_frames_long'] = 100
tracking_opts['long_interval'] = 10

tracking_opts['w_decay'] = 0.0005
tracking_opts['momentum'] = 0.9
tracking_opts['grad_clip'] = 10
tracking_opts['lr_mult'] = {'fc6':10}
tracking_opts['ft_layers'] = ['fc']
