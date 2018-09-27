## Turbo-boosting Object Detector


A PyTorch implementation, originally forked from a public
[repository](https://github.com/multimodallearning/pytorch-mask-rcnn) based on the Mask-RCNN work.

### Overview

![alt text](https://github.com/hli2020/turbo-boost-detection/blob/master/demo/assets/motivation_new.png "Logo Title Text 1")

- [ ] Better documentation
- [ ] Switch to PyTorch `0.4.x`


For installation, please check the [`INSTALL`](INSTALL.md) documentation.

### Test and Demo

TODO.


### Train
See the `script` folder to get a sense of how to execuate train/evaluation commands in terminal.

The training schedule, learning rate, and other parameters can be set in the `class`
object of `CocoConfig` in `lib/config.py`.

        sh script/base_8gpu.sh 105/meta_105_quick_1_roipool

### Results

Results for bounding box and segmentation on COCO are reported based on the default configuration and backbone initialized with pretrained
**ImageNet** weights. The metric is mAP on IoU=0.50:0.95.

|    | from scratch | converted from keras | [Matterport's repo](https://github.com/matterport/Mask_RCNN) | original paper
| --- | --- | --- | --- | --- | 
| bbox | TODO | 0.347 | 0.347 | 0.382 | 
| segm | TODO | 0.296 | 0.296 | 0.354 | 

TODO: more results to come







