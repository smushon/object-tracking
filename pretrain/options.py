from collections import OrderedDict

import torch
device = torch.device('cuda:0')
total_mem = torch.cuda.get_device_properties(device).total_memory

opts = OrderedDict()

if total_mem > 2500000000:
    opts['use_gpu'] = True
    print('using gpu for training')
else:
    opts['use_gpu'] = False
    print('using cpu for training')
opts['use_summary'] = True

# import platform
# import sys
# OS = platform.system()
# if OS == 'Windows':
#     # the git directory is syncing with my OneDrive so I don't want to copy the .mat file (~400 MB)
#     opts['init_model_path'] = 'C:/Users/smush/OneDrive/Documents/GitHub/MDNet-py3.5/models/imagenet-vgg-m.mat'
# elif OS == 'Linux':
#     opts['init_model_path'] = '../models/imagenet-vgg-m.mat'
# else:
#     sys.exit("aa! errors!")
opts['init_model_path'] = '../models/imagenet-vgg-m.mat'

opts['model_path'] = '../models/mdnet_vot-otb_new.pth'
opts['summary_path'] = '../models/summary'

opts['batch_frames'] = 8
opts['batch_pos'] = 32
opts['batch_neg'] = 96

opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

opts['img_size'] = 107
opts['padding'] = 16

opts['lr'] = 0.0001
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['ft_layers'] = ['conv','fc']
opts['lr_mult'] = {'fc':10}
opts['n_cycles'] = 50
