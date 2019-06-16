from collections import OrderedDict

import torch
training_device = torch.device('cuda:0')
total_mem = torch.cuda.get_device_properties(training_device).total_memory

pretrain_opts = OrderedDict()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    training_device = torch.device('cuda:0')
    total_mem = torch.cuda.get_device_properties(training_device).total_memory
    pretrain_opts['large_memory_gpu'] = True  # regnet won't backward() on every sample
    if total_mem > 2500000000:  # 2.5 GB
        pretrain_opts['use_gpu'] = True
        # pretrain_opts['large_memory_gpu'] = True
        print('mdnet training using gpu')
    else:
        pretrain_opts['use_gpu'] = False
        # pretrain_opts['large_memory_gpu'] = False
        print('mdnet training using cpu')
        training_device = 'cpu'
else:
    pretrain_opts['use_gpu'] = False
    pretrain_opts['large_memory_gpu'] = False
    print('mdnet training using cpu')
    training_device = 'cpu'


pretrain_opts['use_summary'] = True

# import platform
# import sys
# OS = platform.system()
# if OS == 'Windows':
#     # the git directory is syncing with my OneDrive so I don't want to copy the .mat file (~400 MB)
#     pretrain_opts['init_model_path'] = 'C:/Users/smush/OneDrive/Documents/GitHub/MDNet-py3.5/models/imagenet-vgg-m.mat'
# elif OS == 'Linux':
#     pretrain_opts['init_model_path'] = '../models/imagenet-vgg-m.mat'
# else:
#     sys.exit("aa! errors!")
pretrain_opts['init_model_path'] = '../models/imagenet-vgg-m.mat'

pretrain_opts['model_path'] = '../models/mdnet_vot-otb_new.pth'
pretrain_opts['summary_path'] = '../models/summary'

pretrain_opts['batch_frames'] = 8
pretrain_opts['batch_pos'] = 32
pretrain_opts['batch_neg'] = 96

pretrain_opts['overlap_pos'] = [0.7, 1]
pretrain_opts['overlap_neg'] = [0, 0.5]

pretrain_opts['img_size'] = 107
pretrain_opts['padding'] = 16

pretrain_opts['lr'] = 0.0001
pretrain_opts['w_decay'] = 0.0005
pretrain_opts['momentum'] = 0.9
pretrain_opts['grad_clip'] = 10
pretrain_opts['ft_layers'] = ['conv','fc']
pretrain_opts['lr_mult'] = {'fc':10}
pretrain_opts['n_cycles'] = 50
