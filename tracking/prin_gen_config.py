import os
import json
import numpy as np


def prin_gen_config(args):
    if args.seq != '':
        # generate config from a sequence name

        # seq_home = '/data1/tracking/princeton/validation'
        # seq_home = '../dataset/OTB'
        seq_home = args.seq_home
        save_home = '../result_fig'
        result_home = '../result'

        seq_name = args.seq
        # img_dir = os.path.join(seq_home, seq_name, 'rgb')
        img_dir = os.path.join(seq_home, seq_name, 'img')
        # img_dir = seq_home + '/' + seq_name + '/rgb'
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')
        # gt_path = seq_home + '/' + seq_name + '/groundtruth_rect.txt'

        # print('loading images from: ', img_dir)

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]

        gt = np.loadtxt(gt_path, delimiter=',')
        if gt.shape[1] == 5:
            gt = gt[:, :-1]
        init_bbox = gt[0]

        savefig_dir = os.path.join(save_home, seq_name)
        result_dir = os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # result_path = os.path.join(result_dir, 'result.json')
        result_path = result_dir

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None

    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path
