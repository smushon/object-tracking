import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt

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
#from pynvml import *

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)


###########################################
# tracking: speed-ups
if opts['use_gpu']:
    load_features_from_file = False
    avg_iters_per_sequence = 3
    fewer_images = False
else:  # minimalist - just see the code works
    load_features_from_file = True
    avg_iters_per_sequence = 1
    fewer_images = True

save_features_to_file = False
detailed_printing = False

if load_features_from_file:
    save_features_to_file = False

# benchmarking
losses_strings = {1:'original-focal', 2:'average-with-iou'}
loss_indices_for_tracking = [1, 2]
models_strings = {1:'original-git', 2:'new-learnt'}
models_paths = {1:opts['model_path'], 2:opts['new_model_path']}
models_indices_for_tracking = [1, 2]
perform_tracking = True
display_benchmark_results = True
###########################################


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False, loss_index=1, model_path=opts['model_path']):

    # num_images include frame 0
    if fewer_images:
        num_images = 3
    else:
        num_images = len(img_list)

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((num_images, 4))
    result_bb = np.zeros((num_images, 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    # Init model
    model = MDNet(model_path)
    if opts['use_gpu']:
        model = model.cuda()
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

    # Train bbox regressor
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
    if load_features_from_file:
        pos_feats = torch.load('../features/pos_feats.pt')
        neg_feats = torch.load('../features/neg_feats.pt')
    else:
        # Extract pos/neg features
        if detailed_printing:
            print('       extracting features from BB samples...')
        pos_feats = forward_samples(model, image, pos_examples)
        neg_feats = forward_samples(model, image, neg_examples)
        if save_features_to_file:
            torch.save(pos_feats, '../features/pos_feats.pt')
            torch.save(neg_feats, '../features/neg_feats.pt')
        if detailed_printing:
            print('       finished extracting features from BB samples.')
    ######################
    feat_dim = pos_feats.size(-1)

    ######################
    # Extract pos/neg IoUs
    pos_ious = overlap_ratio(pos_examples, target_bbox)
    neg_ious = overlap_ratio(neg_examples, target_bbox)
    pos_ious_tensor = torch.from_numpy(pos_ious)
    neg_ious_tensor = torch.from_numpy(neg_ious)
    ######################


    # Initial training
    if detailed_printing:
        print('       first training pass on FC layers...')
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], \
          iou_loss=iou_loss2, pos_ious=pos_ious, neg_ious=neg_ious, loss_index=loss_index)
    # train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], \
    #       iou_loss=iou_loss2, pos_ious=pos_ious_tensor, neg_ious=neg_ious_tensor)
    if detailed_printing:
        print('       finished first training pass on FC layers.')

    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]

    ######################
    # Init pos/neg ious for update
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

        fig = plt.figure(1,frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

        result_ious = np.nan
        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
            #################
            num_gts = np.minimum(gt.shape[0], num_images)
            gt_centers = gt[:num_gts, :2] + gt[:num_gts, 2:] / 2
            result_centers = np.zeros_like(gt[:num_gts, :2])
            result_centers[0] = gt_centers[0]
            result_ious = np.zeros(num_gts, dtype='float64')
            result_ious[0] = 1.
            #################

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    print('    main loop...')
    num_short_updates = 0
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

        # Estimate target bbox
        samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']

        # Expand search area at failure
        if success:
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Copy previous result at failure
        if not success:
            target_bbox = result[i - 1]
            bbreg_bbox = result_bb[i - 1]

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

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
            # we could also try to use bbreg_bbox instead of target_bbox
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
            pos_iou_data = np.concatenate(pos_ious_all[-nframes:])
            neg_iou_data = np.concatenate(neg_ious_all)
            pos_ious_data_tensor = torch.from_numpy(pos_iou_data)
            neg_ious_data_tensor = torch.from_numpy(neg_iou_data)
            ######################
            if detailed_printing:
                print('      short term update')
            num_short_updates += 1
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], \
                  iou_loss=iou_loss2, pos_ious=pos_iou_data, neg_ious=neg_iou_data, loss_index=loss_index)
            # train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], \
            #       iou_loss=iou_loss2, pos_ious=pos_ious_data_tensor, neg_ious=neg_ious_data_tensor)

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            ######################
            pos_iou_data = np.concatenate(pos_ious_all)
            neg_iou_data = np.concatenate(neg_ious_all)
            pos_ious_data_tensor = torch.from_numpy(pos_iou_data)
            neg_ious_data_tensor = torch.from_numpy(neg_iou_data)
            ######################
            if detailed_printing:
                print('      long term update')
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], \
                  iou_loss=iou_loss2, pos_ious=pos_iou_data, neg_ious=neg_iou_data, loss_index=loss_index)
            # train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], \
            #       iou_loss=iou_loss2, pos_ious=pos_ious_data_tensor, neg_ious=neg_ious_data_tensor)

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
                    #################
                    result_ious[i] = overlap_ratio(result_bb[i], gt[i])[0]
                    result_centers[i] = result_bb[i, :2] + result_bb[i, 2:] / 2
                    #################
                else:
                    gt_rect.set_xy(np.array([np.nan,np.nan]))
                    gt_rect.set_width(np.nan)
                    gt_rect.set_height(np.nan)


            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
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

    #############
    # result_distanes = np.linalg.norm(result_centers - gt_centers, ord=2)
    result_distanes = scipy.spatial.distance.cdist(result_centers, gt_centers, metric='euclidean').diagonal()

    print('    main loop finished, %d short updates' % (num_short_updates))

    # overlap_threshold = np.arange(0,1.01,step=0.01)
    # success_rate = np.zeros(overlap_threshold.size)
    # for i in range(overlap_threshold.shape[0]):
    #     success_rate[i] = np.sum(result_ious > overlap_threshold[i]) / result_ious.shape[0]
    #
    # location_error_threshold = np.arange(0,50.5,step=0.5)
    # precision = np.zeros(location_error_threshold.size)
    # for i in range(location_error_threshold.shape[0]):
    #     precision[i] = np.sum(result_distanes < location_error_threshold[i]) / result_distanes.shape[0]
    #
    # if display:
    #     plt.figure(2)  # new figure
    #     # plt.pause(.01)
    #     plt.plot(result_distanes)
    #     # print('distances ', result_distanes)
    #     plt.ylabel('distances')
    #     plt.xlabel('image number')
    #     # plt.show(block=False)
    #
    #     plt.figure(3)  # new figure
    #     # plt.pause(.01)
    #     plt.plot(result_ious)
    #     # print('ious ', result_ious)
    #     plt.ylabel('ious')
    #     plt.xlabel('image number')
    #     # plt.show(block=False)
    #
    #     plt.figure(4)  # new figure
    #     # plt.pause(.01)
    #     plt.plot(success_rate)
    #     plt.ylabel('success rate')
    #     plt.xlabel('overlap threshold')
    #     # plt.show(block=False)
    #
    #     plt.figure(5)  # new figure
    #     # plt.pause(.01)
    #     plt.plot(precision)
    #     plt.ylabel('precision')
    #     plt.xlabel('location error threshold')
    #     # plt.show(block=False)
    #
    #     # plt.pause(.01)
    #     # plt.show(block=False)
    #     # plt.show()
    #############

    fps = num_images / spf_total
    return result, result_bb, fps, result_distanes, result_ious


if __name__ == "__main__":

    # device = torch.device('cuda:0')
    # total_mem = torch.cuda.get_device_properties(device).total_memory

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='DragonBaby', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true', default=False)
    parser.add_argument('-d', '--display', action='store_true', default=True)

    args = parser.parse_args()
    assert (args.seq != '' or args.json != '')

    # ------
    # img_list - list of (relative path) file names of the jpg images
    #   example: '../dataset/OTB/DragonBaby/img/img####.jpg'
    # gt - a (2-dim, N x 4) list of 4 coordinates of ground truth BB for each image
    # init_bbox - this is gt[0]

    # Generate sequence config
    # img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    # Generate sequence of princeton dataset config
    img_list, init_bbox, gt, savefig_dir, display, result_path = prin_gen_config(args)

    # ------

    print('')
    # tracking + online training + save results
    if perform_tracking:
        # for loss_index in loss_indices_for_tracking:  # we comapare several loss functions
        tracking_started = time.time()
        for loss_index, model_index in itertools.product(loss_indices_for_tracking, models_indices_for_tracking):

            tracking_start = time.time()
            print('tracking: model ' + models_strings[model_index] + ' loss ' + losses_strings[loss_index])

            # each run is random, so we need to average before comparing
            for avg_iter in np.arange(0, avg_iters_per_sequence):

                print('  iteration %d / %d started' % (avg_iter+1, avg_iters_per_sequence))
                iteration_start = time.time()

                # Run tracker (+ online training)
                result, result_bb, fps, result_distanes, result_ious = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display, loss_index=loss_index, model_path=models_paths[model_index])

                if avg_iter == 0:
                    result_distanes_avg = result_distanes
                    result_ious_avg = result_ious
                else:
                    result_distanes_avg = (result_distanes_avg*avg_iter + result_distanes) / (avg_iter+1)
                    result_ious_avg = (result_ious_avg * avg_iter + result_ious) / (avg_iter + 1)

                iteration_time = time.time() - iteration_start
                print('  iteration time elapsed: %.3f' % (iteration_time))


            # Save result
            res = {}
            res['res'] = result_bb.round().tolist()
            res['type'] = 'rect'
            res['fps'] = fps
            res['ious'] = result_ious_avg.tolist()
            res['distances'] = result_distanes_avg.tolist()
            result_fullpath = os.path.join(result_path, 'result_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '.json')
            json.dump(res, open(result_fullpath, 'w'), indent=2)

            tracking_time = time.time() - tracking_start
            print('tracking: model ' + models_strings[model_index] + ' loss ' + losses_strings[loss_index] + ' - elapsed %.3f' % (tracking_time))

        tracking_time = time.time() - tracking_started
        print('finished %d losses x %d models - elapsed %d' % (len(loss_indices_for_tracking), len(models_indices_for_tracking), tracking_time))

    # ------

    if display_benchmark_results:
        for loss_index, model_index in itertools.product(loss_indices_for_tracking, models_indices_for_tracking):
        # for loss_index in loss_indices_for_tracking:
            # result_fullpath = os.path.join(result_path, 'result' + str(loss_index) + '.json')
            result_fullpath = os.path.join(result_path, 'result_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '.json')
            with open(result_fullpath, "r") as read_file:
                res = json.load(read_file)
            result_distanes = np.asarray(res['distances'])
            result_ious = np.asarray(res['ious'])

            overlap_threshold = np.arange(0,1.01,step=0.01)
            success_rate = np.zeros(overlap_threshold.size)
            for i in range(overlap_threshold.shape[0]):
                success_rate[i] = np.sum(result_ious > overlap_threshold[i]) / result_ious.shape[0]

            location_error_threshold = np.arange(0,50.5,step=0.5)
            precision = np.zeros(location_error_threshold.size)
            for i in range(location_error_threshold.shape[0]):
                precision[i] = np.sum(result_distanes < location_error_threshold[i]) / result_distanes.shape[0]

            plt.figure(2)  # new figure
            # plt.plot(result_distanes, label=losses_strings[loss_index])
            plt.plot(result_distanes, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
            plt.ylabel('distances')
            plt.xlabel('image number')
            plt.legend()

            plt.figure(3)  # new figure
            plt.plot(result_ious, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
            plt.ylabel('ious')
            plt.xlabel('image number')
            plt.legend()

            plt.figure(4)  # new figure
            plt.plot(success_rate, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
            plt.ylabel('success rate')
            plt.xlabel('overlap threshold')
            plt.legend()

            plt.figure(5)  # new figure
            plt.plot(precision, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
            plt.ylabel('precision')
            plt.xlabel('location error threshold')
            plt.legend()


        plt.show()
