# py3.5-MDNet
This is MDNet with Python3.5

- The official MDNet MATLAB code is available [here](https://github.com/HyeonseobNam/MDNet) 
- The official MDNet Python2.7 code is [here](https://github.com/HyeonseobNam/py-MDNet)

For details about MDNet please refer to the [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Nam_Learning_Multi-Domain_Convolutional_CVPR_2016_paper.pdf) MDNet: Learning Multi-Domain Convolutional Neural Networks for Visual Tracking

## ToDo List
~~1.Transfer from python2 to python3~~
2.TensorBoard support
3.Use FocalLoss instead of traditional CrossEntropy Loss
4.Use MobileNet as the backbone net

## Prerequisites
  - python 3.5
  - [PyTorch](http://pytorch.org/) and its dependencies
 
## Usage
 
### Tracking
  ```bash
   cd tracking
   python run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
  ```
   You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python run_tracker.py -s [seq name]```
   - ```python run_tracker.py -j [json path]```
   
### Pretraining
   - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
   - Download [VOT](http://www.votchallenge.net/) datasets into "dataset/vot201x"
	```bash
	cd pretrain
   python prepro_data.py
   python train_mdnet.py
	```