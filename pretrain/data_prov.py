import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

sys.path.insert(0,'../modules')
from sample_generator import *
from utils import *


class FCDataset(data.Dataset):
    def __init__(self, img_dir, img_list, gt, opts):

        self.img_list = np.array([os.path.join(img_dir, img) for img in img_list])
        self.gt = gt
        self.batch_frames = opts['batch_frames']
        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0


class PosRegionDataset(data.Dataset):
    def __init__(self, img_dir, img_list, gt, opts, is_cuda, generate_std=False, pre_generate=False):

        self.img_list = np.array([os.path.join(img_dir, img) for img in img_list])

        self.generate_std = generate_std
        if generate_std:
            self.image_std_as_numpy = np.empty((len(self.img_list), 107, 107, 3))
            for i, img_path in enumerate(self.img_list):
                image = Image.open(img_path).convert('RGB')
                image_std = image.resize((107,107))
                image_std_as_numpy = np.asarray(image_std)
                self.image_std_as_numpy[i] = image_std_as_numpy

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']

        self.overlap_pos = opts['overlap_pos']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0

        image = Image.open(self.img_list[0]).convert('RGB')
        self.image_size = image.size
        # self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.1, True)

        # I choose same parameters as tracker sample_generator
        # pos_generator and neg_generator have different parameters, the whole concept starting to seem wierd
        if not generate_std:
            self.pos_generator = SampleGenerator('gaussian', image.size, trans_f=0.6, scale_f=1.05, aspect_f=None, valid=True)
        else:
            self.pos_generator = SampleGenerator('gaussian', self.crop_size, trans_f=0.6, scale_f=1.05, aspect_f=None, valid=True)

        if not generate_std:
            self.gt = gt.copy()
        self.gt_std_as_numpy = gt.copy()
        self.gt_std_as_numpy[:, 0] = gt[:, 0] * self.crop_size / self.image_size[0]
        self.gt_std_as_numpy[:, 2] = gt[:, 2] * self.crop_size / self.image_size[0]
        self.gt_std_as_numpy[:, 1] = gt[:, 1] * self.crop_size / self.image_size[1]
        self.gt_std_as_numpy[:, 3] = gt[:, 3] * self.crop_size / self.image_size[1]
        if is_cuda:
            self.gt_std_as_tensor = torch.from_numpy(self.gt_std_as_numpy).float().cuda()
        else:
            self.gt_std_as_tensor = torch.from_numpy(self.gt_std_as_numpy).float()

        self.pre_generate = pre_generate
        if pre_generate:
            pos_regions = np.empty((0, 3, self.crop_size, self.crop_size))
            pos_bbs = np.empty((0, 4))
            for i, (img_path, bbox) in enumerate(zip(self.img_list, self.gt)):
                image = Image.open(img_path).convert('RGB')
                image = np.asarray(image)

                pos_examples = gen_samples(self.pos_generator, bbox, self.batch_pos)
                pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)), axis=0)
                pos_bbs = np.concatenate((pos_bbs, np.array(pos_examples, dtype='float32')), axis=0)

            self.pos_regions = torch.from_numpy(pos_regions).float()

            pos_bbs_std = pos_bbs
            # assuming all frames in given sequence have the same size
            pos_bbs_std[:,0] = pos_bbs[:,0] * self.crop_size / self.image_size[0]
            pos_bbs_std[:,2] = pos_bbs[:,2] * self.crop_size / self.image_size[0]
            pos_bbs_std[:,1] = pos_bbs[:,1] * self.crop_size / self.image_size[1]
            pos_bbs_std[:,3] = pos_bbs[:,3] * self.crop_size / self.image_size[1]
            if torch.cuda.is_available():
                self.pos_bbs_std_as_tensor = torch.Tensor(pos_bbs_std).cuda()
            else:
                self.pos_bbs_std_as_tensor = torch.Tensor(pos_bbs_std)



    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]  # a batch of frame numbers
        if len(idx) < self.batch_frames:  # if out of frame indexes, reshuffle and restart collecting
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        if self.pre_generate:
            return idx
        else:
            pos_regions = np.empty((0, 3, self.crop_size, self.crop_size))
            pos_bbs = np.empty((0,4))
            # image_path_list = []
            # gt_bbox_list = []
            num_example_list = []

            if self.generate_std:
                for i, (img_path, image_std, bbox_std) in enumerate(zip(self.img_list[idx], self.image_std_as_numpy[idx], self.gt_std_as_numpy[idx])):
                    # image_path_list.append(img_path)
                    n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)

                    # pos_examples = gen_samples(self.pos_generator, bbox, n_pos, overlap_range=self.overlap_pos)
                    pos_examples_std = gen_samples(self.pos_generator, bbox_std, n_pos)
                    pos_regions = np.concatenate((pos_regions, self.extract_regions(image_std, pos_examples_std)), axis=0)
                    num_example_list.append(len(pos_examples_std))
                    pos_bbs = np.concatenate((pos_bbs, np.array(pos_examples_std, dtype='float32')), axis=0)

            if not self.generate_std:
                for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
                    # image_path_list.append(img_path)

                    image = Image.open(img_path).convert('RGB')
                    image = np.asarray(image)

                    n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)

                    # pos_examples = gen_samples(self.pos_generator, bbox, n_pos, overlap_range=self.overlap_pos)
                    pos_examples = gen_samples(self.pos_generator, bbox, n_pos)
                    pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)), axis=0)
                    num_example_list.append(len(pos_examples))

                    # no need for these any more
                    # image = torch.from_numpy(image)
                    # image_list.append(image)
                    # bbox = torch.from_numpy(bbox).float()
                    # gt_bbox_list.append(bbox)

                    pos_bbs = np.concatenate((pos_bbs, np.array(pos_examples, dtype='float32')), axis=0)

            pos_regions = torch.from_numpy(pos_regions).float()

            # returns:
            #   pos_regions - crop of images, multiple per image
            #   pos_bbs - coordinates bounding boxes, multiple per image
            #   num_example_list - how many samples per image

            # no need toreturn this
            # self.gt_std_as_tensor[idx]

            return pos_regions, pos_bbs, num_example_list, idx

    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            # tracking-time forward samples function work with valid=False...
            regions[i] = crop_image(image, sample, self.crop_size, self.padding) #, True)

        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions


class RegionDataset(data.Dataset):
    def __init__(self, img_dir, img_list, gt, opts):

        self.img_list = np.array([os.path.join(img_dir, img) for img in img_list])
        self.gt = gt

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']
        
        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0
        
        image = Image.open(self.img_list[0]).convert('RGB')
        self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.1, True)
        self.neg_generator = SampleGenerator('uniform', image.size, 1, 1.2, 1.1, True)

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]  # a batch of frame numbers
        if len(idx) < self.batch_frames:  # if out of frame indexes, reshuffle and restart collecting
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions = np.empty((0, 3, self.crop_size, self.crop_size))
        neg_regions = np.empty((0, 3, self.crop_size, self.crop_size))
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)

            n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)
            n_neg = (self.batch_neg - len(neg_regions)) // (self.batch_frames - i)
            pos_examples = gen_samples(self.pos_generator, bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = gen_samples(self.neg_generator, bbox, n_neg, overlap_range=self.overlap_neg)
            
            pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)),axis=0)
            neg_regions = np.concatenate((neg_regions, self.extract_regions(image, neg_examples)),axis=0)

        pos_regions = torch.from_numpy(pos_regions).float()
        neg_regions = torch.from_numpy(neg_regions).float()
        return pos_regions, neg_regions
    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)
        
        regions = regions.transpose(0,3,1,2)
        regions = regions.astype('float32') - 128.
        return regions
