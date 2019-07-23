import os
import json
import numpy as np


def prin_gen_config(args, sub_folder='', ro=False, benchmark_dataset='OTB', quadrilateral=False):
    if args.seq != '':
        # generate config from a sequence name

        seq_home = args.seq_home
        save_home = '../result_fig'
        result_home = '../result'
        if sub_folder is not '':
            result_home = os.path.join(result_home, sub_folder)

        seq_name = args.seq
        if benchmark_dataset=='OTB':
            img_dir = os.path.join(seq_home, seq_name, 'img')
            gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')
        elif benchmark_dataset=='VOT':
            img_dir = os.path.join(seq_home, seq_name)
            gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')
        else:
            raise Exception('unknown dataset')

        # print('loading images from: ', img_dir)

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]

        try:
            gt = np.loadtxt(gt_path, delimiter=',')
        except:
            gt = np.loadtxt(gt_path)  # delimeter is white spaces
        gt_origin = None

        if (gt.shape[1] == 4) and quadrilateral:
            gt = None
            print('TBD - transform [x,y,w,h] into [x1,y1,x2,y2,x3,y3,x4,y4]')
        elif (gt.shape[1] == 5) and not quadrilateral:
            print('WTF ?? gt.shape[1]=5 ???')
            gt = gt[:, :-1]
        elif (gt.shape[1] == 5) and quadrilateral:
            print('WTF ?? gt.shape[1]=5 ???')
            gt = None

        elif gt.shape[1] == 8:
            if quadrilateral:
                gt_origin = gt.copy()
            x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

        init_bbox = gt[0]

        savefig_dir = os.path.join(save_home, seq_name)
        result_dir = os.path.join(result_home, seq_name)
        if not ro:
            # if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
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

    if args.savefig and not ro:
        # if not os.path.exists(savefig_dir):
        os.makedirs(savefig_dir, exist_ok=True)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, (not args.dont_display), result_path, gt_origin
