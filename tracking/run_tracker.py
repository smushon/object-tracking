import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

# needed for following usage:
#  cd tracking
#  python run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
sys.path.insert(0,'../modules')

from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *
from prin_gen_config import *
from FocalLoss import *
from tracking_utils import *

import itertools
from cycler import cycler
#from pynvml import *

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)


# --------------- opencv options --------------------
# Illumination Variation - the illumination in the target region is significantly changed.
OTB_IL = ['Basketball', 'Box', 'Car1', 'Car2', 'Car24', 'Car4', 'CarDark', 'Coke', 'Crowds', 'David', 'Doll', 'FaceOcc2', 'Fish', 'Human2', 'Human4', 'Human7', 'Human8', 'Human9', 'Ironman', 'KiteSurf', 'Lemming', 'Liquor', 'Man', 'Matrix', 'Mhyang', 'MotorRolling', 'Shaking', 'Singer1', 'Singer2', 'Skating1', 'Skiing', 'Soccer', 'Sylvester', 'Tiger1', 'Tiger2', 'Trans', 'Trellis', 'Woman']

# Scale Variation – the ratio of the bounding boxes of the first frame and the current frame is out of the range ts, ts > 1 (ts=2).
OTB_SV = ['Biker', 'BlurBody', 'BlurCar2', 'BlurOwl', 'Board', 'Box', 'Boy', 'Car1', 'Car24', 'Car4', 'CarScale', 'ClifBar', 'Couple', 'Crossing', 'Dancer', 'David', 'Diving', 'Dog', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FleetFace', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2', 'Gym', 'Human2', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Ironman', 'Jump', 'Lemming', 'Liquor', 'Matrix', 'MotorRolling', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer1', 'Skater', 'Skater2', 'Skating1', 'Skating2', 'Skiing', 'Soccer', 'Surfer', 'Toy', 'Trans', 'Trellis', 'Twinnings', 'Vase', 'Walking', 'Walking2', 'Woman']

# Occlusion – the target is partially or fully occluded.
OTB_OCC = ['Basketball', 'Biker', 'Bird2', 'Bolt', 'Box', 'CarScale', 'ClifBar', 'Coke', 'Coupon', 'David', 'David3', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Football', 'Freeman4', 'Girl', 'Girl2', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Ironman', 'Jogging', 'Jump', 'KiteSurf', 'Lemming', 'Liquor', 'Matrix', 'Panda', 'RedTeam', 'Rubik', 'Singer1', 'Skating1', 'Skating2', 'Soccer', 'Subway', 'Suv', 'Tiger1', 'Tiger2', 'Trans', 'Walking', 'Walking2', 'Woman']

# Deformation – non-rigid object deformation.
OTB_DEF = ['Basketball', 'Bird1', 'Bird2', 'BlurBody', 'Bolt', 'Bolt2', 'Couple', 'Crossing', 'Crowds', 'Dancer', 'Dancer2', 'David', 'David3', 'Diving', 'Dog', 'Dudek', 'FleetFace', 'Girl2', 'Gym', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Jogging', 'Jump', 'Mhyang', 'Panda', 'Singer2', 'Skater', 'Skater2', 'Skating1', 'Skating2', 'Skiing', 'Subway', 'Tiger1', 'Tiger2', 'Trans', 'Walking', 'Woman']

# Motion Blur – the target region is blurred due to the motion of target or camera.
OTB_MB = ['Biker', 'BlurBody', 'BlurCar1', 'BlurCar2', 'BlurCar3', 'BlurCar4', 'BlurFace', 'BlurOwl', 'Board', 'Box', 'Boy', 'ClifBar', 'David', 'Deer', 'DragonBaby', 'FleetFace', 'Girl2', 'Human2', 'Human7', 'Human9', 'Ironman', 'Jump', 'Jumping', 'Liquor', 'MotorRolling', 'Soccer', 'Tiger1', 'Tiger2', 'Woman']

# Fast Motion – the motion of the ground truth is larger than tm pixels (tm=20).
OTB_FM = ['Biker', 'Bird1', 'Bird2', 'BlurBody', 'BlurCar1', 'BlurCar2', 'BlurCar3', 'BlurCar4', 'BlurFace', 'BlurOwl', 'Board', 'Boy', 'CarScale', 'ClifBar', 'Coke', 'Couple', 'Deer', 'DragonBaby', 'Dudek', 'FleetFace', 'Human6', 'Human7', 'Human9', 'Ironman', 'Jumping', 'Lemming', 'Liquor', 'Matrix', 'MotorRolling', 'Skater2', 'Skating2', 'Soccer', 'Surfer', 'Tiger1', 'Tiger2', 'Toy', 'Vase', 'Woman']

# In-Plane Rotation – the target rotates in the image plane.
OTB_IPR = ['Bird2', 'BlurBody', 'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Boy', 'CarScale', 'ClifBar', 'Coke', 'Dancer', 'David', 'David2', 'Deer', 'Diving', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc2', 'FleetFace', 'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Gym', 'Ironman', 'Jump', 'KiteSurf', 'Matrix', 'MotorRolling', 'MountainBike', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer2', 'Skater', 'Skater2', 'Skiing', 'Soccer', 'Surfer', 'Suv', 'Sylvester', 'Tiger1', 'Tiger2', 'Toy', 'Trellis', 'Vase']

# Out-of-Plane Rotation – the target rotates out of the image plane.
OTB_OPR = ['Basketball', 'Biker', 'Bird2', 'Board', 'Bolt', 'Box', 'Boy', 'CarScale', 'Coke', 'Couple', 'Dancer', 'David', 'David2', 'David3', 'Dog', 'Dog1', 'Doll', 'DragonBaby', 'Dudek', 'FaceOcc2', 'FleetFace', 'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2', 'Gym', 'Human2', 'Human3', 'Human6', 'Ironman', 'Jogging', 'Jump', 'KiteSurf', 'Lemming', 'Liquor', 'Matrix', 'Mhyang', 'MountainBike', 'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer1', 'Singer2', 'Skater', 'Skater2', 'Skating1', 'Skating2', 'Skiing', 'Soccer', 'Surfer', 'Sylvester', 'Tiger1', 'Tiger2', 'Toy', 'Trellis', 'Twinnings', 'Woman']

# Out-of-View – some portion of the target leaves the view.
OTB_OV = ['Biker', 'Bird1', 'Board', 'Box', 'ClifBar', 'DragonBaby', 'Dudek', 'Human6', 'Ironman', 'Lemming', 'Liquor', 'Panda', 'Suv', 'Tiger2']

# Background Clutters – the background near the target has the similar color or texture as the target.
OTB_BC = ['Basketball', 'Board', 'Bolt2', 'Box', 'Car1', 'Car2', 'Car24', 'CarDark', 'ClifBar', 'Couple', 'Coupon', 'Crossing', 'Crowds', 'David3', 'Deer', 'Dudek', 'Football', 'Football1', 'Human3', 'Ironman', 'Liquor', 'Matrix', 'Mhyang', 'MotorRolling', 'MountainBike', 'Shaking', 'Singer2', 'Skating1', 'Soccer', 'Subway', 'Trellis']

# Low Resolution – the number of pixels inside the ground-truth bounding box is less than tr (tr =400).
OTB_LR = ['Biker', 'Car1', 'Freeman3', 'Freeman4', 'Panda', 'RedTeam', 'Skiing', 'Surfer', 'Walking']

OTB_attributes_dict = {'IL': OTB_IL, 'SV': OTB_SV, 'OCC':OTB_OCC, 'DEF':OTB_DEF, 'MB':OTB_MB, 'FM':OTB_FM,
                       'IPR':OTB_IPR, 'OPR':OTB_OPR, 'OV':OTB_OV, 'BC':OTB_BC, 'LR':OTB_LR}

OTB_select_attributes_strings = ['IL', 'SV', 'OCC', 'FM', 'BC', 'LR']
# ---------------------------------------------------



###########################################
import platform
import statistics
# seq_home = '../dataset/'
usr_home = os.path.expanduser('~')
OS = platform.system()
if OS == 'Windows':
    # usr_home = 'C:/Users/smush/'
    seq_home = os.path.join(usr_home, 'downloads/')
elif OS == 'Linux':
    # usr_home = '~/'
    seq_home = os.path.join(usr_home, 'MDNet-data/')
else:
    sys.exit("aa! errors!")

# benchmark_dataset = 'VOT/vot2016'
benchmark_dataset = 'OTB'
seq_home = os.path.join(seq_home, benchmark_dataset)

my_sequence_list = ['DragonBaby', 'Car4', 'Woman']
show_average_over_sequences = True
show_per_sequence = True
###########################################


###########################################
# benchmarking
losses_strings = {1: 'original-focal', 2: 'average-with-iou'}
models_strings = {1: 'original-git', 2: 'new-learnt'}
models_paths = {1: opts['model_path'], 2: opts['new_model_path']}

# tracking: speed-ups
if opts['use_gpu']:
    load_features_from_file = False
    avg_iters_per_sequence = 3  # should be 15 per the VOT challenge
    fewer_images = False  # default. can be overriden by command argument
    loss_indices_for_tracking = [1, 2]
    models_indices_for_tracking = [1, 2]
else:  # minimalist - just see the code works
    load_features_from_file = True
    avg_iters_per_sequence = 1
    fewer_images = True
    loss_indices_for_tracking = [1]
    models_indices_for_tracking = [1, 2]

# load_features_from_file = True  ############# hack for debug ###################33

sequence_len_limit = 10  # limit number of frame taken from each sequence
save_features_to_file = False
detailed_printing = False

if load_features_from_file:
    save_features_to_file = False


init_after_loss = False  # True - VOT metrics, False - OTB metrics
display_VOT_benchmark = True
display_OTB_benchmark = True
###########################################


# --------------- opencv options --------------------
import cv2
OPENCV_OBJECT_TRACKERS = {"csrt": cv2.TrackerCSRT_create,
                          "kcf": cv2.TrackerKCF_create,
                          "boosting": cv2.TrackerBoosting_create,
                          "mil": cv2.TrackerMIL_create,
                          "tld": cv2.TrackerTLD_create,
                          "medianflow": cv2.TrackerMedianFlow_create,
                          "mosse": cv2.TrackerMOSSE_create}
OPENCV_TRACKERS_COLORS = {"csrt": 'yellow',
                          "kcf": 'blue',
                          "boosting": 'violet',
                          "mil": 'magenta',
                          "tld": 'black',
                          "medianflow": 'orange',
                          "mosse": 'cyan'}

tracker_strings_selected = ['kcf', 'mil'] #, 'medianflow']

# # trackers = []
# trackers_dict = {}
# for tracker_String in tracker_strings_selected:
#     # trackers.append(OPENCV_OBJECT_TRACKERS[tracker_String]())
#     trackers_dict.update( {tracker_String : OPENCV_OBJECT_TRACKERS[tracker_String]()})

bb_fc_model_path = '../models/regnet.pth'
if not os.path.isfile(bb_fc_model_path):
    raise Exception('no saved RegNet state')
regnet_state = torch.load(bb_fc_model_path)
if 'translate_mode' in regnet_state.keys():
    translate_mode = regnet_state['translate_mode']  # we overide train_regnet input according to saved state
else:
    translate_mode = True

use_opencv = False
perform_refinement = True  # True - use RegNet/BBregressor to refine BB, False - use opencv/mdnet tracker output as-is
use_regnet = True
use_lin_reg = True
update_non_refined_on_fail = True  # True - will use opencv tracker output, False - will use last successful refinement
# ---------------------------------------------------


##################
import options
device = options.tracking_device
opts = options.tracking_opts
##################


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False, loss_index=1, model_path=opts['model_path'], seq_name=None):

    # num_images include frame 0
    if fewer_images:
        num_images = min(sequence_len_limit, len(img_list))
    else:
        num_images = len(img_list)

    # Init bbox per GT of frame 0
    if init_bbox is None:
        raise Exception('No init BBox for tracker')
    target_bbox = np.array(init_bbox)
    result = np.zeros((num_images, 4))
    result_bb = np.zeros((num_images, 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    #################
    if use_regnet:
        result_regnet_bb = np.zeros((num_images, 4))
        result_regnet_bb[0] = target_bbox
    #################

    #################
    # init metrics we return at end of run_mdnet
    if gt is not None:
        num_gts = np.minimum(gt.shape[0], num_images)
        gt_centers = gt[:num_gts, :2] + gt[:num_gts, 2:] / 2
        result_centers = np.zeros_like(gt[:num_gts, :2])
        result_centers[0] = gt_centers[0]
        result_ious = np.zeros(num_gts, dtype='float64')
        result_ious[0] = 1.

        result_regnet_centers = np.zeros_like(gt[:num_gts, :2])
        result_regnet_centers[0] = gt_centers[0]
        result_regnet_ious = np.zeros(num_gts, dtype='float64')
        result_regnet_ious[0] = 1.
    #################

    # Init model
    model = MDNet(model_path)
    if opts['use_gpu']:
        model = model.to(device)
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion
    # criterion = BinaryLoss()
    criterion = FocalLoss(class_num=2, alpha=torch.ones(2, 1)*0.25, size_average=False)
    iou_loss = MyIoULoss()
    iou_loss2 = MyIoULoss2()

    # Init Optimizers
    # e.g. SGD with a list of 6 parameter groups
    # e.g. parameter group:
    #     dampening: 0
    #     lr: 0.0001  <--  learning rate, changing
    #     momentum: 0.9
    #     nesterov: False
    #     weight_decay: 0.0005
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    # --------
    print('    initializing...')
    tic = time.time()

    # SampleGenerator -- returns a list of random BBs (translate, scale) around a given BB
    # gen_samples -- will attempt to call SampleGenerator under additional constaints (until give up)
    #       ratio constraint -- IoU between generated and original BB in specified range
    #       scale constraint -- relative size change of BB (compared with original BB) is in specified range
    # forward_samples -- crops the image per the list of BBs (i.e. samples) and forwards each crop
    #       for each crop, returns the output out_layer='conv3' (each output is an entire feature layer)
    #       for online training, we later specify e.g. 'fc6' instead of the default 'conv3'
    # BBRegressor -- inference
    #       input: BB and its features (i.e. forward_samples)
    #       operation: use linear regression to predict ground truth BB
    #       output: corrections to BB coordinates (i.e. fine tune) towards predicted GT-BB
    # BBRegressor -- training
    #       input: BB and its features (i.e. forward_samples) + GT-BB (e.g. marked target on first frame)
    # pos_examples -- defined by having large IoU with target_bbox, e.g. in range [0.7,1]
    # neg_examples -- defined by having small IoU with target_bbox, e.g. in range [0,0.3]
    # target_bbox -- during init time, this is the GT BB of the object (frame 0)
    # train -- updates the weights of the common FC layers (fc4-fc5) and the new FC layer (fc6)
    #       features of negative examples should output scores --> 0
    #       features of positive examples should output scores --> 1
    #       hard mining - training iterations choose the worst negative examples, i.e. those with highest scores[1]
    #       loss function is FocalLoss(class_num=2, alpha=torch.ones(2, 1)*0.25, size_average=False)

    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    ######################
    if use_opencv:
        # init opencv trackers and refined_bb
        # bbox = (current_frame_x1, current_frame_y1, current_frame_width, current_frame_height)

        # (re-) instantiate selected trackers before init
        trackers_dict = {}
        for tracker_String in tracker_strings_selected:
            trackers_dict.update({tracker_String: OPENCV_OBJECT_TRACKERS[tracker_String]()})

        # refined_bb_dict will be used to hold the latest best guess for BB for each tracker
        refined_bb_dict = {}
        cv_output_dict = {}
        for (string, tracker) in trackers_dict.items():
            refined_bb_dict.update({string: target_bbox})
            cv_output_dict.update({string: [True,target_bbox]})
            # tracker.clear()

            # init instantiated trackers
            if not tracker.init(np.array(image), tuple(target_bbox)):
                raise Exception('error init tracker: ', string)
            # else:
            #     print('      success init tracker: ', string)

    # use_regnet - i.e. we can use alongside BBRegressor regardless of opencv
    # perform_refinement - i.e. we can take opencv/mdnet trackers output as-is
    if use_regnet:
        bb_fc_model = RegNet(translate_mode=translate_mode, state=regnet_state)
        if 'best_prec' in regnet_state.keys():
            best_prec = regnet_state['best_prec']
            print("    regnet loaded with precision = %.4f" % best_prec)
        if opts['use_gpu']:
            bb_fc_model = bb_fc_model.to(device)
        bb_fc_model.eval()
    ######################

    # Train bbox regressor
    if perform_refinement and use_lin_reg:
        if detailed_printing:
            print('       training BB regressor...')
        bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                     target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
        bbreg_feats = forward_samples(model, image, bbreg_examples)
        bbreg = BBRegressor(image.size)  # image_size is e.g. (640, 360)
        bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        if detailed_printing:
            print('       finished training BB regressor.')

    # Draw pos/neg samples
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init']),
        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                    target_bbox, opts['n_neg_init'] // 2, opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    ######################
    # hacks to speed-up execution for debugging on expense of accuracy
    fw_samples = False
    if load_features_from_file:
        if os.path.isfile('../features/' + seq_name + '_pos_feats.pt') and os.path.isfile('../features/' + seq_name + '_neg_feats.pt'):
            pos_feats = torch.load('../features/' + seq_name + '_pos_feats.pt')
            neg_feats = torch.load('../features/' + seq_name + '_neg_feats.pt')
        else:
            fw_samples = True
    if fw_samples or (not load_features_from_file):
        # Extract pos/neg features
        if detailed_printing:
            print('       extracting features from BB samples...')
        if fewer_images:  # shorter run in general, less accurate
            pos_feats = forward_samples(model, image, pos_examples[:50])
            neg_feats = forward_samples(model, image, neg_examples[:500])
        else:
            pos_feats = forward_samples(model, image, pos_examples)
            neg_feats = forward_samples(model, image, neg_examples)
        if save_features_to_file or fw_samples:
            torch.save(pos_feats, '../features/' + seq_name + '_pos_feats.pt')
            torch.save(neg_feats, '../features/' + seq_name + '_neg_feats.pt')
        if detailed_printing:
            print('       finished extracting features from BB samples.')
    ######################
    feat_dim = pos_feats.size(-1)

    ######################
    # Extract pos/neg IoUs
    if loss_index == 2:
        pos_ious = overlap_ratio(pos_examples, target_bbox)
        neg_ious = overlap_ratio(neg_examples, target_bbox)
    ######################

    # Initial training
    if detailed_printing:
        print('       first training pass on FC layers...')
    if loss_index == 2:
        train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], \
              iou_loss=iou_loss2, pos_ious=pos_ious, neg_ious=neg_ious, loss_index=loss_index)
    else:
        train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], \
              loss_index=loss_index)
    if detailed_printing:
        print('       finished first training pass on FC layers.')

    # Init sample generators
    # sample_generator - for tracking
    # pos_generator, neg_generator - for online re-training
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]

    ######################
    # Init pos/neg ious for update
    if loss_index == 2:
        pos_ious_all = [pos_ious[:opts['n_pos_update']]]
        neg_ious_all = [neg_ious[:opts['n_neg_update']]]
    ######################

    spf_total = time.time() - tic
    if detailed_printing:
        print('    initialization done, Time: %.3f' % (spf_total))

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(1,frameon=False, figsize=figsize, dpi=dpi/0.8)
        fig.clf()
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        ax = plt.axes()
        im = ax.imshow(image, aspect='auto')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        if gt is not None:
            gt_rect = patches.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3], linewidth=2, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_artist(gt_rect)
            plt.plot([], label='gt', color='#00ff00')

        target_rect = patches.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3], linewidth=2, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_artist(target_rect)
        plt.plot([], label='target', color='#ff0000')

        ######################
        if perform_refinement and use_lin_reg:
            linreg_rect = patches.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3], linewidth=2, edgecolor="#0f000f", zorder=1, fill=False)
            ax.add_artist(linreg_rect)
            plt.plot([], label='linreg', color='#0f000f')

        if use_regnet:
            regnet_rect = patches.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3], linewidth=2, edgecolor="#0000ff", zorder=1, fill=False)
            ax.add_artist(regnet_rect)
            if perform_refinement:
                plt.plot([], label='regnet_ref', color='#0000ff')
            else:
                plt.plot([], label='regnet_trk', color='#0000ff')
        ######################

        ######################
        if use_opencv:
            refined_patch_dict = {}
            for tracker_String in tracker_strings_selected:
                print('adding patch for ', tracker_String)
                refined_patch_dict.update({tracker_String:
                        plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                        linewidth=3, edgecolor=OPENCV_TRACKERS_COLORS[tracker_String], zorder=1, fill=False)})
                ax.add_patch(refined_patch_dict[tracker_String])
        ######################

        if display:
            plt.pause(.01)
            plt.legend()
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    print('    main loop...')
    num_short_updates = 0
    spf_total = 0  # I don't want to take into account initialization
    for i in range(1, num_images):

        # given frame[i],
        # we take BB estimation ('target_bbox') we made in frame[i-1]
        # we generate BB samples around it and forward through CNN+FC+new_head('fc6')
        # the new 'target_bbox' is the average (per coordinate) of top 5 samples (based on 'fc6' scores[1])
        # success is defined if the mean score[1] of those top 5 BBs passes some threshold
        # note: score[0] is the background classification, score[1] is the target classification
        # note: the paper says new 'target_bbox' is the single sample with highest "positive score" (aka scores[1])
        # note: both scores can be negative. later the focal loss computes softmax to squeeze their values to (0,1)
        #
        # if "success",
        # we keep this new 'target_bbox' and also calculate a new GT-BB prediction as follows:
        # regression fine-tunes those top 5 (i.e. highest score[1]) BB samples
        # prediction of GT-BB is taken as average (per coordinate) of regressions of those top 5 BBs
        # this fine-tuned (and final) prediction is called 'bbreg_bbox'
        #
        # if "success", we also generate new positive and negative BB samples around the new 'target_bbox'
        # we forward them and record their output features ('conv3')
        # lets call them 'positive features' and 'negative features' for ease
        # the recorded history of positive samples is longer (100 frames) then of negative samples (20 frames)
        # we will use both stacks of recorded features for the long- and short-term updates
        #
        # updates:
        # long-term - if success happenned on a modulo 10 iteration, we perform update using all available
        #       positive and negative features recorded
        # short-term - if not success, we perform similar update routine using all available negative features
        #       but limit the number of positive features

        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')


        try_again = True
        while try_again:
            # Estimate target bbox
            if sample_generator.get_trans_f() == opts['trans_f_expand']:
                samples = gen_samples(sample_generator, target_bbox, 2*opts['n_samples'])
            else:
                samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
            # for sample in samples:
            #     print("iou: %.5f" % overlap_ratio(target_bbox, sample))
            sample_scores = forward_samples(model, image, samples, out_layer='fc6')
            top_scores, top_idx = sample_scores[:, 1].topk(5)
            top_idx = top_idx.cpu().numpy()
            target_score = top_scores.mean()
            target_bbox = samples[top_idx].mean(axis=0)

            if sample_generator.get_trans_f() == opts['trans_f_expand']:
                try_again = False

            # target_score = 1 ################# hack for debug #################
            success = target_score > opts['success_thr']
            # Expand search area at failure
            if success:
                sample_generator.set_trans_f(opts['trans_f'])
                try_again = False
            else:
                sample_generator.set_trans_f(opts['trans_f_expand'])


        ##############################################################
        if use_opencv:
            for (cv_string, cv_tracker) in trackers_dict.items():
                cv_success, cv_BB = cv_tracker.update(np.array(image))
                cv_output_dict.update({cv_string: [cv_success, cv_BB]})
        ##############################################################


        ###########################################
        if gt is not None:
            if i < gt.shape[0]:
                # identify tracking failure and abort when in VOT mode
                if use_opencv:
                    IoU = overlap_ratio(target_bbox, gt[i])[0]
                    # TBD - instead calculate IoU based on opencv traker(s) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                else:
                    IoU = overlap_ratio(target_bbox, gt[i])[0]
                if (IoU == 0) and init_after_loss:
                    print('    * lost track in frame %d since init*' % (i))
                    result_distances = scipy.spatial.distance.cdist(result_centers[:i], gt_centers[:i], metric='euclidean').diagonal()
                    result_regnet_distances = scipy.spatial.distance.cdist(result_regnet_centers[:i], gt_centers[:i], metric='euclidean').diagonal()
                    num_images_tracked = i - 1  # we don't count frame 0 and current frame (lost track)

                    # display failed frame
                    if display:
                        im.set_data(image)
                        if gt is not None:
                            if i < gt.shape[0]:
                                gt_rect.set_xy(gt[i, :2])
                                gt_rect.set_width(gt[i, 2])
                                gt_rect.set_height(gt[i, 3])
                            else:
                                gt_rect.set_xy(np.array([np.nan, np.nan]))
                                gt_rect.set_width(np.nan)
                                gt_rect.set_height(np.nan)

                        target_rect.set_xy(result[i, :2])
                        target_rect.set_width(result[i, 2])
                        target_rect.set_height(result[i, 3])

                        #########################################
                        # draw raw (unrefined) cv trackers output
                        if use_opencv:
                            for tracker_String in tracker_strings_selected:
                                refined_patch_dict[tracker_String].set_xy(cv_output_dict[tracker_String][1][:2])
                                refined_patch_dict[tracker_String].set_width(cv_output_dict[tracker_String][1][2])
                                refined_patch_dict[tracker_String].set_height(cv_output_dict[tracker_String][1][3])
                        #########################################

                        plt.pause(2)  # pause longer to observe failure
                        plt.draw()

                    return result[:i], result_bb[:i], num_images_tracked, spf_total, result_distances, result_ious[:i], result_regnet_distances, result_regnet_ious[:i], True
        ########################################



        ##############################################################
        if use_opencv:
            for (cv_string, cv_tracker) in trackers_dict.items():
                cv_output = cv_output_dict[cv_string]
                cv_success, cv_BB = cv_output[0], cv_output[1]

                if cv_success:  # opencv tracker will not return a BB otherwise
                    if perform_refinement and use_regnet:

                        # prepare input for refinement network
                        cv_feats_BB = forward_samples(model, image, np.array([cv_BB]))
                        cv_feats_full_frame = forward_samples(model, image, np.array([[0, 0, image.size[0], image.size[1]]]))
                        cv_BB_std = np.array(cv_BB)
                        img_size_std = opts['img_size']
                        cv_BB_std[0] = cv_BB[0] * img_size_std / image.size[0]
                        cv_BB_std[2] = cv_BB[2] * img_size_std / image.size[0]
                        cv_BB_std[1] = cv_BB[1] * img_size_std / image.size[1]
                        cv_BB_std[3] = cv_BB[3] * img_size_std / image.size[1]

                        # with torch.no_grad():
                        bb_fc_input = torch.cat((cv_feats_BB, cv_feats_full_frame, torch.Tensor(np.array([cv_BB_std]))), dim=1)

                        if opts['use_gpu']:
                            bb_fc_input = bb_fc_input.to(device=device)

                        # perform refinement
                        with torch.no_grad():
                            cv_BB_refined_std = bb_fc_model(bb_fc_input)
                        if translate_mode:
                            cv_BB_refined_std += bb_fc_input[0,-4:]

                        # cv_BB_refined_std = cv_BB_refined_std.detach().numpy()
                        cv_BB_refined_std = cv_BB_refined_std.numpy()

                        # refined BB fail check
                        # if cv_BB_refined_std[2] < 2 or cv_BB_refined_std[3] < 2:  # BB too small
                        #     print('      refinement for opencv model ', cv_string, ' failed at frame ', i, ' after init')
                        #     if update_non_refined_on_fail:
                        #         refined_bb_dict.update({cv_string: np.array(cv_BB)})

                        # re-scale refined BB back to frame proportions
                        cv_BB_refined = cv_BB_refined_std
                        cv_BB_refined[0] = cv_BB_refined_std[0] * image.size[0] / img_size_std
                        cv_BB_refined[2] = cv_BB_refined_std[2] * image.size[0] / img_size_std
                        cv_BB_refined[1] = cv_BB_refined_std[1] * image.size[1] / img_size_std
                        cv_BB_refined[3] = cv_BB_refined_std[3] * image.size[1] / img_size_std

                        # keep refined BB for this tracker
                        refined_bb_dict.update({cv_string: cv_BB_refined})

                        # use refined BB to re-init the opencv tracker
                        if not cv_tracker.init(np.array(image), tuple(cv_BB_refined)):
                            raise Exception('error re-init tracke per refinementr: ', cv_string)

                # nothing to do if opencv tracker fails to track
                # we can only hope to use re-init procedure from VOT benchmark to re-init tracker that lost track
                # else:
                #     print('opencv model ', cv_string, ' failed at frame ', i, ' after init')
        ##############################################################

        ###################################################
        if use_regnet:
            # prepare input for refinement network
            if (success or init_after_loss) and perform_refinement:
                if use_opencv:
                    bb_to_refine = samples[top_idx]  # regnet will refine opencv - TBD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                else:
                    bb_to_refine = samples[top_idx]  # regnet will refine mdnet best samples
            else:
                bb_to_refine = np.array([result_regnet_bb[i - 1]])  # regnet will refine its previous output
            res_regnet_feats_BB = forward_samples(model, image, bb_to_refine)
            feats_full_frame = forward_samples(model, image, np.array([[0, 0, image.size[0], image.size[1]]]))

            # result_regnet_bb_std = np.array(result_regnet_bb[i - 1])
            result_regnet_bb_std = bb_to_refine
            img_size_std = opts['img_size']
            result_regnet_bb_std[:, 0] = bb_to_refine[:, 0] * img_size_std / image.size[0]
            result_regnet_bb_std[:, 2] = bb_to_refine[:, 2] * img_size_std / image.size[0]
            result_regnet_bb_std[:, 1] = bb_to_refine[:, 1] * img_size_std / image.size[1]
            result_regnet_bb_std[:, 3] = bb_to_refine[:, 3] * img_size_std / image.size[1]

            # result_regnet_bb_std_as_tensor = torch.Tensor(np.array([result_regnet_bb_std]))
            result_regnet_bb_std_as_tensor = torch.Tensor(result_regnet_bb_std)
            if opts['use_gpu']:
                result_regnet_bb_std_as_tensor = result_regnet_bb_std_as_tensor.cuda()

            bb_fc_input = torch.cat((res_regnet_feats_BB, feats_full_frame.expand(res_regnet_feats_BB.shape), result_regnet_bb_std_as_tensor), dim=1)

            if opts['use_gpu']:
                bb_fc_input = bb_fc_input.to(device=device)

            # perform refinement
            with torch.no_grad():
                result_regnet_bb_refined_std = bb_fc_model(bb_fc_input)
            if translate_mode:
                result_regnet_bb_refined_std += bb_fc_input[:, -4:]

            # averaging the refinement results
            result_regnet_bb_refined_std = result_regnet_bb_refined_std.mean(dim=0)

            # cv_BB_refined_std = cv_BB_refined_std.detach().numpy()
            # result_regnet_bb_refined_std = result_regnet_bb_refined_std.numpy()

            # re-scale refined BB back to frame proportions
            result_regnet_bb_refined = result_regnet_bb_refined_std
            result_regnet_bb_refined[0] = result_regnet_bb_refined_std[0] * image.size[0] / img_size_std
            result_regnet_bb_refined[2] = result_regnet_bb_refined_std[2] * image.size[0] / img_size_std
            result_regnet_bb_refined[1] = result_regnet_bb_refined_std[1] * image.size[1] / img_size_std
            result_regnet_bb_refined[3] = result_regnet_bb_refined_std[3] * image.size[1] / img_size_std

            if opts['use_gpu']:
                bbregnet_bbox = np.array(result_regnet_bb_refined.cpu())
            else:
                bbregnet_bbox = np.array(result_regnet_bb_refined)
        ###################################################

        # Bbox regression
        if success:
            if perform_refinement and use_lin_reg:
                bbreg_samples = samples[top_idx]
                bbreg_feats = forward_samples(model, image, bbreg_samples)
                bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
                bbreg_bbox = bbreg_samples.mean(axis=0)
            else:
                bbreg_bbox = target_bbox

        # Copy previous result at failure
        if not success:
            target_bbox = result[i - 1]
            if perform_refinement and use_lin_reg:
                bbreg_bbox = result_bb[i - 1]
            else:
                bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox  # bbreg_box already determined dependent on 'perform_refinement'
        ###########################################
        if use_regnet:
            result_regnet_bb[i] = bbregnet_bbox
        ###########################################

        # ###########################################
        # # previously, a tracking loss (IoU==0) check was here based in result_bb
        # # but then I decided to base it on result (pre-refinement)
        # if gt is not None:
        #     if i < gt.shape[0]:
        #         # identify tracking failure and abort when in VOT mode
        #         IoU = overlap_ratio(result_bb[i], gt[i])[0]
        #         if (IoU == 0) and init_after_loss:
        #             print('    * lost track in frame %d since init*' % (i))
        #             result_distances = scipy.spatial.distance.cdist(result_centers[:i], gt_centers[:i], metric='euclidean').diagonal()
        #             result_regnet_distances = scipy.spatial.distance.cdist(result_regnet_centers[:i], gt_centers[:i], metric='euclidean').diagonal()
        #             num_images_tracked = i - 1  # we don't count frame 0 and current frame (lost track)
        #
        #             # display failed frame
        #             if display:
        #                 im.set_data(image)
        #                 if gt is not None:
        #                     if i < gt.shape[0]:
        #                         gt_rect.set_xy(gt[i, :2])
        #                         gt_rect.set_width(gt[i, 2])
        #                         gt_rect.set_height(gt[i, 3])
        #                     else:
        #                         gt_rect.set_xy(np.array([np.nan, np.nan]))
        #                         gt_rect.set_width(np.nan)
        #                         gt_rect.set_height(np.nan)
        #
        #                 target_rect.set_xy(result[i, :2])
        #                 target_rect.set_width(result[i, 2])
        #                 target_rect.set_height(result[i, 3])
        #
        #                 if perform_refinement and use_lin_reg:
        #                     linreg_rect.set_xy(result_bb[i, :2])
        #                     linreg_rect.set_width(result_bb[i, 2])
        #                     linreg_rect.set_height(result_bb[i, 3])
        #                 ######################################################
        #                 if use_regnet:
        #                     regnet_rect.set_xy(result_regnet_bb[i, :2])
        #                     regnet_rect.set_width(result_regnet_bb[i, 2])
        #                     regnet_rect.set_height(result_regnet_bb[i, 3])
        #                 ######################################################
        #
        #                 #########################################
        #                 if use_opencv:
        #                     for tracker_String in tracker_strings_selected:
        #                         refined_patch_dict[tracker_String].set_xy(refined_bb_dict[tracker_String][:2])
        #                         refined_patch_dict[tracker_String].set_width(refined_bb_dict[tracker_String][2])
        #                         refined_patch_dict[tracker_String].set_height(refined_bb_dict[tracker_String][3])
        #                         # draw rectangle based on refined_bb_dict[tracker_String]
        #                 #########################################
        #
        #                 plt.pause(2)  # pause longer to observe failure
        #                 plt.draw()
        #
        #             return result[:i], result_bb[:i], num_images_tracked, spf_total, result_distances, result_ious[:i], result_regnet_distances, result_regnet_ious[:i], True
        # ########################################

        #################
        if gt is not None:
            if i < gt.shape[0]:
                result_ious[i] = IoU
                result_centers[i] = result_bb[i, :2] + result_bb[i, 2:] / 2

                result_regnet_ious[i] = overlap_ratio(result_regnet_bb[i], gt[i])[0]
                result_regnet_centers[i] = result_regnet_bb[i, :2] + result_regnet_bb[i, 2:] / 2
        #################

        # Data collect
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(pos_generator, target_bbox,
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox,
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = forward_samples(model, image, pos_examples)
            neg_feats = forward_samples(model, image, neg_examples)
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

            ######################
            # Extract pos/neg IoUs
            if loss_index == 2:
                # we could also try to use bbreg_bbox instead of target_bbox  ????????????????????????????????????????
                pos_ious = overlap_ratio(pos_examples, target_bbox)
                neg_ious = overlap_ratio(neg_examples, target_bbox)
                pos_ious_all.append(pos_ious)
                neg_ious_all.append(neg_ious)
                if len(pos_ious_all) > opts['n_frames_long']:
                    del pos_ious_all[0]
                if len(neg_ious_all) > opts['n_frames_short']:
                    del neg_ious_all[0]
            ######################

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            ######################
            if loss_index == 2:
                pos_iou_data = np.concatenate(pos_ious_all[-nframes:])
                neg_iou_data = np.concatenate(neg_ious_all)
            ######################
            if detailed_printing:
                print('      short term update')
            num_short_updates += 1
            if loss_index == 2:
                train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], \
                      iou_loss=iou_loss2, pos_ious=pos_iou_data, neg_ious=neg_iou_data, loss_index=loss_index)
            else:
                train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], \
                      loss_index=loss_index)

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)

            ######################
            if loss_index == 2:
                pos_iou_data = np.concatenate(pos_ious_all)
                neg_iou_data = np.concatenate(neg_ious_all)
            ######################

            if detailed_printing:
                print('      long term update')
            if loss_index == 2:
                train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], \
                      iou_loss=iou_loss2, pos_ious=pos_iou_data, neg_ious=neg_iou_data, loss_index=loss_index)
            else:
                train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], \
                      loss_index=loss_index)

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                if i<gt.shape[0]:
                    gt_rect.set_xy(gt[i, :2])
                    gt_rect.set_width(gt[i, 2])
                    gt_rect.set_height(gt[i, 3])

                else:
                    gt_rect.set_xy(np.array([np.nan,np.nan]))
                    gt_rect.set_width(np.nan)
                    gt_rect.set_height(np.nan)

            target_rect.set_xy(result[i, :2])
            target_rect.set_width(result[i, 2])
            target_rect.set_height(result[i, 3])

            if perform_refinement and use_lin_reg:
                linreg_rect.set_xy(result_bb[i, :2])
                linreg_rect.set_width(result_bb[i, 2])
                linreg_rect.set_height(result_bb[i, 3])

            ######################################################
            if use_regnet:
                regnet_rect.set_xy(result_regnet_bb[i, :2])
                regnet_rect.set_width(result_regnet_bb[i, 2])
                regnet_rect.set_height(result_regnet_bb[i, 3])
            ######################################################

            #########################################
            if use_opencv:
                for tracker_String in tracker_strings_selected:
                    refined_patch_dict[tracker_String].set_xy(refined_bb_dict[tracker_String][:2])
                    refined_patch_dict[tracker_String].set_width(refined_bb_dict[tracker_String][2])
                    refined_patch_dict[tracker_String].set_height(refined_bb_dict[tracker_String][3])
                    # draw rectangle based on refined_bb_dict[tracker_String]
            #########################################

            if display:

                if gt is not None:
                    if i < gt.shape[0]:
                        target_iou = overlap_ratio(result[i], gt[i])[0]
                        str = 'IoU | target: %.3f' % target_iou
                        if perform_refinement and use_lin_reg:
                            linreg_iou = overlap_ratio(result_bb[i], gt[i])[0]
                            str = str + ' | linreg: %.3f' % linreg_iou
                        if use_regnet:
                            regnet_iou = overlap_ratio(result_regnet_bb[i], gt[i])[0]
                            str = str + ' | regnet: %.3f' % regnet_iou
                        plt.xlabel(str)
                plt.pause(.01)
                # plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '%04d.jpg' % (i)), dpi=dpi)

        if detailed_printing:
            if gt is None:
                print("      Frame %d/%d, Score %.3f, Time %.3f" % \
                      (i, num_images-1, target_score, spf))
            else:
                if i<gt.shape[0]:
                    print("      Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                        (i, num_images-1, overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))
                else:
                    print("      Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                        (i, num_images-1, overlap_ratio(np.array([np.nan,np.nan,np.nan,np.nan]), result_bb[i])[0], target_score, spf))

    # plt.close()

    # result_distances = np.linalg.norm(result_centers - gt_centers, ord=2)
    result_distances = scipy.spatial.distance.cdist(result_centers, gt_centers, metric='euclidean').diagonal()
    result_regnet_distances = scipy.spatial.distance.cdist(result_regnet_centers, gt_centers, metric='euclidean').diagonal()
    # fps = num_images / spf_total
    num_images_tracked = num_images-1  # I don't want to count initialization frame (i.e. frame 0)
    print('    main loop finished, %d frames, %d short updates' % (num_images, num_short_updates))

    return result, result_bb, num_images_tracked, spf_total, result_distances, result_ious, result_regnet_distances, result_regnet_ious, False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='seq_list', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-dd', '--dont_display', action='store_true')  # default: display frames with BBs

    ################
    parser.add_argument('-sh', '--seq_home', default=seq_home, help='input seq_home')
    parser.add_argument('-at', '--attributes', default='selection', help='attributes separated by -')
    parser.add_argument('-i', '--init_after_loss', action='store_true')
    parser.add_argument('-l', '--lmt_seq', action='store_true')
    parser.add_argument('-ln', '--seq_len_lmt', default='0', help='sequence length limit')

    parser.add_argument('-p', '--plt_bnch', action='store_true')  # plot benchmark graphs from saved results
    parser.add_argument('-tr', '--perform_tracking', action='store_true')  # plot benchmark graphs from saved results
    ################

    args = parser.parse_args()
    assert (args.seq != '' or args.json != '')

    ################
    if args.lmt_seq:
        fewer_images = True
        args.seq_len_lmt = int(float(args.seq_len_lmt))
        if args.seq_len_lmt > 0:
            sequence_len_limit = args.seq_len_lmt
        # else, sequence_len_limit stays with default value

    init_after_loss = args.init_after_loss

    if args.attributes == 'selection':
        select_attributes_strings = OTB_select_attributes_strings
    elif args.attributes == 'all':
        select_attributes_strings = list(OTB_attributes_dict.keys())
    else:
        select_attributes_strings = args.attributes.split('-')
        select_attributes_strings = list(set(select_attributes_strings).intersection(set(OTB_attributes_dict.keys())))  # clean typos
        select_attributes_strings.sort()
    print('attributes selected: ', select_attributes_strings)

    sequence_wish_list = []
    for att in select_attributes_strings:
        sequence_wish_list.extend(OTB_attributes_dict[att])
    sequence_wish_list = list(set(sequence_wish_list))
    sequence_wish_list.sort()
    my_sequence_list = list(set(my_sequence_list).intersection(set(sequence_wish_list)))
    my_sequence_list.sort()

    if args.seq == 'seq_list':
        sequence_list = my_sequence_list
    elif args.seq == 'all':
        sequence_list = next(os.walk(seq_home))[1]
    else:
        sequence_list = [args.seq]  # e.g. -s DragonBaby

    perform_tracking = args.perform_tracking
    display_benchmark_results = args.plt_bnch
    ################

    # ------

    # tracking + online training + save results
    if perform_tracking:
        # for loss_index in loss_indices_for_tracking:  # we comapare several loss functions
        tracking_started = time.time()

        # model_index - iterate over different weights learnt
        # loss_index - iterate over different loss functions for online training
        # sequnce - iterate over different sequences
        for model_index, loss_index, sequence in itertools.product(models_indices_for_tracking, loss_indices_for_tracking, sequence_list):

            # ------
            # img_list - list of (relative path) file names of the jpg images
            #   example: '../dataset/OTB/DragonBaby/img/img####.jpg'
            # gt - a (2-dim, N x 4) list of 4 coordinates of ground truth BB for each image
            # init_bbox - this is gt[0]

            # Generate sequence config
            # img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

            # Generate sequence of princeton dataset config
            args.seq = sequence
            img_list, init_bbox, gt, savefig_dir, display, result_path = prin_gen_config(args)
            # ------

            tracking_start = time.time()
            print('')
            print('tracking: | model ' + models_strings[model_index] + ' | loss ' + losses_strings[loss_index] + ' | sequence ' + sequence)

            # each run is random, so we need to average before comparing
            # each iteration starts from the finish of the offline training
            # there is no dependency between iterations
            for avg_iter in np.arange(0, avg_iters_per_sequence):

                print('  iteration %d / %d started' % (avg_iter+1, avg_iters_per_sequence))
                iteration_start = time.time()

                if init_after_loss:  # loss means loss of tracking
                    init_frame_index = 0
                    while init_frame_index < len(img_list) - 1:  # we want at least one frame for tracking after init
                        result, result_bb, num_images_tracked, spf_total, result_distances, result_ious, result_regnet_distances, result_regnet_ious, lost_track = run_mdnet(img_list[init_frame_index:], gt[init_frame_index], gt=gt[init_frame_index:], savefig_dir=savefig_dir, display=display, loss_index=loss_index, model_path=models_paths[model_index], seq_name=sequence)
                        if init_frame_index == 0:
                            result_ious_tot = result_ious
                            result_regnet_ious_tot = result_regnet_ious
                            num_images_tracked_tot = num_images_tracked
                            spf_total_tot = spf_total

                            # init_frame_index does not include init frame nor frame where tracking was lost
                            if lost_track:
                                lost_track_tot = 1
                                init_frame_index = num_images_tracked + 1 + 5
                            else:
                                lost_track_tot = 0
                                init_frame_index = len(img_list)
                        else:
                            result_ious_tot = np.concatenate((result_ious_tot, result_ious))
                            result_regnet_ious_tot = np.concatenate((result_regnet_ious_tot, result_regnet_ious))
                            num_images_tracked_tot += num_images_tracked
                            spf_total_tot += spf_total

                            if lost_track:
                                lost_track_tot += 1
                                init_frame_index += num_images_tracked + 1 + 5
                            else:
                                init_frame_index = len(img_list)
                    accuracy = np.mean(result_ious_tot)
                    regnet_accuracy = np.mean(result_regnet_ious_tot)
                    fps = num_images_tracked_tot / spf_total_tot
                else:  # i.e. not init_after_loss:
                    lost_track_tot = 0
                    result, result_bb, num_images_tracked, spf_total, result_distances, result_ious, result_regnet_distances, result_regnet_ious, lost_track = run_mdnet(
                        img_list, gt[0], gt=gt,
                        savefig_dir=savefig_dir, display=display, loss_index=loss_index,
                        model_path=models_paths[model_index], seq_name=sequence)
                    accuracy = np.mean(result_ious)
                    regnet_accuracy = np.mean(result_regnet_ious)
                    fps = num_images_tracked / spf_total

                    # compute step of running average of results over current sequence
                    # since we don't init after loss, it's always the same size of result arrays, so we can average
                    if avg_iter == 0:
                        result_distances_avg = result_distances
                        result_regnet_distances_avg = result_regnet_distances
                        result_ious_avg = result_ious
                        result_regnet_ious_avg = result_regnet_ious
                        result_bb_avg = result_bb
                    else:
                        result_distances_avg = (result_distances_avg*avg_iter + result_distances) / (avg_iter+1)
                        result_regnet_distances_avg = (result_regnet_distances_avg * avg_iter + result_regnet_distances) / (avg_iter + 1)
                        result_ious_avg = (result_ious_avg * avg_iter + result_ious) / (avg_iter + 1)
                        result_regnet_ious_avg = (result_regnet_ious_avg * avg_iter + result_regnet_ious) / (avg_iter + 1)
                        result_bb_avg = (result_bb_avg * avg_iter + result_bb) / (avg_iter + 1)

                if avg_iter == 0:
                    failures_per_seq_avg = lost_track_tot
                    accuracy_avg = accuracy
                    regnet_accuracy_avg = regnet_accuracy
                else:
                    failures_per_seq_avg = (failures_per_seq_avg * avg_iter + lost_track_tot) / (avg_iter + 1)
                    accuracy_avg = (accuracy_avg * avg_iter + accuracy) / (avg_iter + 1)
                    regnet_accuracy_avg = (regnet_accuracy_avg * avg_iter + regnet_accuracy) / (avg_iter + 1)

                iteration_time = time.time() - iteration_start
                print('  iteration time elapsed: %.3f' % (iteration_time))


            # Save result
            res = {}
            res['type'] = 'rect'
            res['fps'] = fps
            if not init_after_loss:
                res['res'] = result_bb_avg.round().tolist()
                res['ious'] = result_ious_avg.tolist()
                res['regnet_ious'] = result_regnet_ious_avg.tolist()
                res['distances'] = result_distances_avg.tolist()
                res['regnet_distances'] = result_regnet_distances_avg.tolist()
            # else:
            res['fails_per_seq'] = failures_per_seq_avg
            res['accuracy'] = accuracy_avg
            res['regnet_accuracy'] = regnet_accuracy_avg
            result_fullpath = os.path.join(result_path, 'result_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_init-' + str(init_after_loss) + '.json')
            json.dump(res, open(result_fullpath, 'w'), indent=2)

            tracking_time = time.time() - tracking_start
            print('tracking: | model ' + models_strings[model_index] + ' | loss ' + losses_strings[loss_index] + ' | sequence ' + sequence + ' | elapsed %.3f' % (tracking_time))

        tracking_time = time.time() - tracking_started
        print('finished %d losses x %d models x %d sequences - elapsed %d' % (len(loss_indices_for_tracking), len(models_indices_for_tracking), len(sequence_list), tracking_time))

    # ------

    if display_benchmark_results:

        for model_index, loss_index in itertools.product(models_indices_for_tracking, loss_indices_for_tracking):

            if show_average_over_sequences:
                if display_VOT_benchmark:
                    avg_accuracy = []
                    avg_regnet_accuracy = []
                    avg_fails = []
                if display_OTB_benchmark:
                    avg_success_rate = np.zeros((len(sequence_list),np.arange(0, 1.01, step=0.01).size))
                    avg_regnet_success_rate = np.zeros((len(sequence_list), np.arange(0, 1.01, step=0.01).size))
                    avg_precision = np.zeros((len(sequence_list),np.arange(0, 50.5, step=0.5).size))
                    avg_regnet_precision = np.zeros((len(sequence_list), np.arange(0, 50.5, step=0.5).size))

            for seq_iter, sequence in enumerate(sequence_list):

                args.seq = sequence
                img_list, init_bbox, gt, savefig_dir, display, result_path = prin_gen_config(args)

                if display_OTB_benchmark:

                    result_fullpath = os.path.join(result_path,'result_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_init-False' + '.json')
                    if not os.path.exists(result_fullpath):
                        print('no file named: ' + result_fullpath)
                        continue
                    with open(result_fullpath, "r") as read_file:
                        res = json.load(read_file)

                    result_distances = np.asarray(res['distances'])
                    result_ious = np.asarray(res['ious'])
                    result_regnet_distances = np.asarray(res['regnet_distances'])
                    result_regnet_ious = np.asarray(res['regnet_ious'])

                    overlap_threshold = np.arange(0, 1.01, step=0.01)  # X axis
                    success_rate = np.zeros(overlap_threshold.size)
                    regnet_success_rate = np.zeros(overlap_threshold.size)
                    for i in range(overlap_threshold.shape[0]):
                        success_rate[i] = np.sum(result_ious > overlap_threshold[i]) / result_ious.shape[0]
                        regnet_success_rate[i] = np.sum(result_regnet_ious > overlap_threshold[i]) / result_regnet_ious.shape[0]
                    # AUC = accuracy = sum(success_rate)

                    location_error_threshold = np.arange(0, 50.5, step=0.5)  # X axis
                    precision = np.zeros(location_error_threshold.size)
                    regnet_precision = np.zeros(location_error_threshold.size)
                    for i in range(location_error_threshold.shape[0]):
                        precision[i] = np.sum(result_distances < location_error_threshold[i]) / result_distances.shape[0]
                        regnet_precision[i] = np.sum(result_regnet_distances < location_error_threshold[i]) / result_regnet_distances.shape[0]

                    if show_average_over_sequences:
                        avg_success_rate[seq_iter,:] = success_rate
                        avg_regnet_success_rate[seq_iter,:] = regnet_success_rate
                        avg_precision[seq_iter,:] = precision
                        avg_regnet_precision[seq_iter, :] = regnet_precision
                        if sequence == sequence_list[-1]:
                            avg_success_rate = avg_success_rate.mean(axis=0)
                            avg_regnet_success_rate = avg_regnet_success_rate.mean(axis=0)
                            avg_precision = avg_precision.mean(axis=0)
                            avg_regnet_precision = avg_regnet_precision.mean(axis=0)
                            plt.figure(11)
                            plt.plot(avg_success_rate,label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
                            plt.plot(avg_regnet_success_rate,label='regnet_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
                            plt.ylabel('success rate')
                            plt.xlabel('overlap threshold')
                            plt.legend()

                            plt.figure(12)
                            plt.plot(avg_precision, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
                            plt.plot(avg_regnet_precision, label='regnet_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
                            plt.ylabel('precision')
                            plt.xlabel('location error threshold')
                            plt.legend()
                    if show_per_sequence:
                        plt.figure(2)
                        # plt.plot(result_distances, label=losses_strings[loss_index])
                        plt.plot(result_distances, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.plot(result_regnet_distances, label='regnet_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('distances')
                        plt.xlabel('image number')
                        plt.legend()

                        plt.figure(3)
                        plt.plot(result_ious, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.plot(result_regnet_ious, label='regnet_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('ious')
                        plt.xlabel('image number')
                        plt.legend()

                        plt.figure(4)
                        plt.plot(success_rate, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.plot(regnet_success_rate, label='regnet_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('success rate')
                        plt.xlabel('overlap threshold')
                        plt.legend()

                        plt.figure(5)
                        plt.plot(precision, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.plot(regnet_precision, label='regnet_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('precision')
                        plt.xlabel('location error threshold')
                        plt.legend()

                if display_VOT_benchmark:

                    result_fullpath = os.path.join(result_path,'result_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_init-' + str(init_after_loss) + '.json')
                    if not os.path.exists(result_fullpath):
                        print('no file named: ' + result_fullpath)
                        continue
                    with open(result_fullpath, "r") as read_file:
                        res = json.load(read_file)

                    if show_average_over_sequences:
                        avg_accuracy.append(res['accuracy'])
                        avg_regnet_accuracy.append(res['regnet_accuracy'])
                        avg_fails.append(res['fails_per_seq'])
                        if sequence == sequence_list[-1]:
                            avg_accuracy = statistics.mean(avg_accuracy)
                            avg_regnet_accuracy = statistics.mean(avg_regnet_accuracy)
                            avg_fails = statistics.mean(avg_fails)
                            plt.figure(6)
                            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']) *
                                                       cycler('marker', ["o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "h", "H", "X", "D", "d"])))
                            plt.plot([avg_fails], [avg_accuracy],
                                     label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
                            plt.plot([avg_fails], [avg_regnet_accuracy],
                                     label='regnet_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
                            plt.ylabel('accuracy')
                            plt.xlabel('failures per sequence')
                            plt.legend()
                    if show_per_sequence:
                        plt.figure(13)
                        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']) *
                                               cycler('marker',["o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "h", "H", "X", "D", "d"])))
                        plt.plot([res['fails_per_seq']], [res['accuracy']], label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.plot([res['fails_per_seq']], [res['regnet_accuracy']], label='regnet_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('accuracy')
                        plt.xlabel('failures per sequence')
                        plt.legend()


        plt.show()
