import os
import numpy as np
import pickle
from collections import OrderedDict

import platform
import sys
from pathlib import PureWindowsPath

dataset = 'VOT'  # 'OTB'
seqlist_else_walk = False
quadrilateral = True

if seqlist_else_walk:
    if dataset == 'VOT':
        if quadrilateral:
            seqlist_path = 'data/vot_quadrilateral.txt'
        else:
            seqlist_path = 'data/mdnet_vot.txt'
    elif dataset == 'OTB':
        seqlist_path = None
        raise Exception('no seqlist for OTB')
    else:
        raise Exception('unknown dataset')

usr_home = os.path.expanduser('~')
OS = platform.system()
if OS == 'Windows':
    seq_home = os.path.join(usr_home, 'downloads')
elif OS == 'Linux':
    seq_home = os.path.join(usr_home, 'MDNet-data')
else:
    sys.exit("aa! errors!")

seq_home = os.path.join(seq_home, dataset)

if dataset == 'OTB':
    gt_txt_file_name = 'groundtruth_rect.txt'
    if not seqlist_else_walk:
        output_path = 'data/otb.pkl'
elif dataset == 'VOT':
    gt_txt_file_name = 'groundtruth.txt'
    if not seqlist_else_walk:
        output_path = 'data/vot.pkl'
else:
    raise Exception('unknown dataset')

if seqlist_else_walk:
    if seqlist_path is None:
        raise Exception('no seqlist file for ' + dataset)
    else:
        output_path = seqlist_path[:seqlist_path.rfind('.txt')] + '.pkl'

if seqlist_else_walk:
    with open(seqlist_path, 'r') as fp:
        seq_list = fp.read().splitlines()
else:
    # seq_list = os.listdir(seq_home)

    if dataset == 'VOT':
        top_seq_list = next(os.walk(seq_home))[1]
        seq_list = []
        for folder in top_seq_list:
            sub_folders = next(os.walk(os.path.join(seq_home, folder)))[1]
            for sub_folder in sub_folders:
                seq_list.append(os.path.join(folder, sub_folder))
    else:
        seq_list = next(os.walk(seq_home))[1]

data = {}
for i, seq in enumerate(seq_list):
    if OS == 'Windows':
        seq = str(PureWindowsPath(seq))
    if dataset == 'OTB':
        img_list = sorted([os.path.join('img', p) for p in os.listdir(os.path.join(seq_home, seq, 'img')) if os.path.splitext(p)[1] == '.jpg'])
    else:
        img_list = sorted([p for p in os.listdir(os.path.join(seq_home, seq)) if os.path.splitext(p)[1] == '.jpg'])

    # I renamed the file, no longer needed
    # if (dataset=='OTB') and ((seq == 'Human4') or (seq == 'Skating2')):
    #     gt_txt_file_name = 'groundtruth_rect.2.txt'

    try:
        gt = np.loadtxt(os.path.join(seq_home, seq, gt_txt_file_name), delimiter=',')
    except:
        gt = np.loadtxt(os.path.join(seq_home, seq, gt_txt_file_name))

    # assert len(img_list) == len(gt), "Lengths do not match for " + seq

    if gt.shape[1] == 4:
        print('non-rotated? ' + seq)
        if quadrilateral:
            print('skipping ' + seq + ' | TBD: transform [x,y,w,h] into [x1,y1,x2,y2,x3,y3,x4,y4]')
            continue

    if gt.shape[1] == 5:
        print('rotated? ' + seq)
        gt = gt[:, :-1]  # this is bullshit, assuming "fifth coordinate" is rotation angle...
        if quadrilateral:
            print('skipping ' + seq + ' | TBD: transform [x,y,w,h,rot_angle?] into [x1,y1,x2,y2,x3,y3,x4,y4]')
            continue
        else:
            print('skipping ' + seq + ' | TBD: transform [x,y,w,h,rot_angle?] into [x1,y1,w,h]')
            continue

    if gt.shape[1] == 8:
        print('generic? ' + seq)
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt_rect = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    data[seq] = {'images': img_list, 'gt': gt_rect, 'gt_origin': gt}  # data is a dictionary {e.g. 'vot2013/cup'} of dictionaries

    # if dataset == 'OTB':
    #     gt_txt_file_name = 'groundtruth_rect.txt'

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)

