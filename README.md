## Turbo-boosting Object Detector


A PyTorch implementation, originally forked from a public
[repository](https://github.com/multimodallearning/pytorch-mask-rcnn) based on Mask-RCNN.

### Overview

### Installation
1. Clone this repository.

        git clone --recursive https://github.com/hli2020/turbo-boost-detection.git

    
2. We use functions from other repositories that need to be build with the right `--arch` option for cuda support.
The functions are Non-Maximum Suppression from ruotianluo's [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)
repository and longcw's [RoiAlign](https://github.com/longcw/RoIAlign.pytorch) and RoiPool. Thanks to them!

        sh setup.sh

3. As we use the [COCO dataset](http://cocodataset.org/#home),
install the [Python COCO API](https://github.com/cocodataset/cocoapi) and
create a symlink.

        ln -s /path/to/coco datasets/coco

4. Download the pretrained models on COCO and ImageNet from
[Google Drive](https://drive.google.com/open?id=1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc).


### Test and Demo


### Train
See the `script` folder to get a sense of training/evaluation commands in terminal.

The training schedule, learning rate, and other parameters can be set in the `class`
object of `CocoConfig` in `lib/config.py`.

        sh script/base_8gpu.sh 105/meta_105_quick_1_roipool

### Results

COCO results for bounding box and segmentation are reported based on training
with the default configuration and backbone initialized with pretrained
**ImageNet** weights. Used metric is AP on IoU=0.50:0.95.

|    | from scratch | converted from keras | Matterport's Mask_RCNN | Mask R-CNN paper | Ours
| --- | --- | --- | --- | --- | --- |
| bbox | TODO | 0.347 | 0.347 | 0.382 | TODO |
| segm | TODO | 0.296 | 0.296 | 0.354 | TODO |



#### Minor installation problems

You might encounter the following warnings on interpolation:

``~/anaconda3/lib/python3.6/site-packages/scipy/ndimage/interpolation.py:616``

Please comment the source files if necessary. Also install the `future` package via conda:

``conda install -c anaconda future``

If you use visdom:

    git clone --recursive https://github.com/facebookresearch/visdom.git
    cd visdom
    pip install -e .

You can activate on a remote server (probably in ``tmux``):
``python -m visdom.server -port=$PORT_ID``



