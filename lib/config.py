import math
import numpy as np
import os
import datetime
from tools.utils import print_log
from tools.collections import AttrDict


class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # NUMBER OF GPUs to use. For CPU use 0
    # GPU_COUNT = 1

    # (deprecated)
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    # IMAGES_PER_GPU = 1

    # (deprecated)
    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    # STEPS_PER_EPOCH = 1000

    # (deprecated)
    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    # VALIDATION_STEPS = 50

    # ==================================
    MODEL = AttrDict()
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    MODEL.BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # Path to pretrained imagenet model # TODO: loading is buggy
    MODEL.PRETRAIN_IMAGENET_MODEL = os.path.join('datasets/pretrain_model', "resnet50_imagenet.pth")
    # Path to pretrained weights file
    MODEL.PRETRAIN_COCO_MODEL = os.path.join('datasets/pretrain_model', 'mask_rcnn_coco.pth')
    MODEL.INIT_FILE_CHOICE = 'last'  # or file (xxx.pth)

    # ==================================
    DATASET = AttrDict()
    # Number of classification classes (including background)
    DATASET.NUM_CLASSES = 81
    DATASET.YEAR = '2014'
    DATASET.PATH = 'datasets/coco'

    # ==================================
    RPN = AttrDict()
    # Length of square anchor side in pixels
    RPN.ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN.ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN.ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more proposals.
    RPN.NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN.TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum suppression (training and inference)
    RPN.POST_NMS_ROIS_TRAINING = 2000
    RPN.POST_NMS_ROIS_INFERENCE = 1000

    RPN.TARGET_POS_THRES = .7
    RPN.TARGET_NEG_THRES = .3

    # ==================================
    MRCNN = AttrDict()
    # If enabled, resize instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    MRCNN.USE_MINI_MASK = True
    MRCNN.MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    # Pooled ROIs
    MRCNN.POOL_SIZE = 7
    MRCNN.MASK_POOL_SIZE = 14
    MRCNN.MASK_SHAPE = [28, 28]

    # ==================================
    DATA = AttrDict()
    # Input image resize
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    DATA.IMAGE_MIN_DIM = 800
    DATA.IMAGE_MAX_DIM = 1024
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    DATA.IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    DATA.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Maximum number of ground truth instances to use in one image
    DATA.MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    DATA.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    DATA.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # ==================================
    ROIS = AttrDict()
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting the RPN NMS threshold.
    ROIS.TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROIS.ROI_POSITIVE_RATIO = 0.33

    # ==================================
    TEST = AttrDict()
    # Max number of final detections
    TEST.DET_MAX_INSTANCES = 100
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    TEST.DET_MIN_CONFIDENCE = 0
    # Non-maximum suppression threshold for detection
    TEST.DET_NMS_THRESHOLD = 0.3
    TEST.SAVE_IM = False

    # ==================================
    TRAIN = AttrDict()
    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    TRAIN.LEARNING_RATE = 0.01
    TRAIN.LEARNING_MOMENTUM = 0.9
    # Weight decay regularization
    TRAIN.WEIGHT_DECAY = 0.0001
    TRAIN.GAMMA = 0.1
    TRAIN.LR_POLICY = 'steps_with_decay'
    # in epoch
    TRAIN.SCHEDULE = [10, 5, 5]
    TRAIN.LR_WARM_UP = True

    TRAIN.SAVE_FREQ_WITHIN_EPOCH = 10

    TRAIN.CLIP_GRAD = True
    TRAIN.MAX_GRAD_NORM = 5.0

    TRAIN.DO_VALIDATION = True
    # (deprecated)
    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to train the RPN.
    # USE_RPN_ROIS = True

    # ==============================
    CTRL = AttrDict()
    CTRL.SHOW_INTERVAL = 20
    CTRL.USE_VISDOM = False

    # for train and inference
    CTRL.BATCH_SIZE = 6

    # ==============================
    MISC = AttrDict()

    def _set_value(self):
        """Set values of computed attributes."""

        if self.CTRL.DEBUG:
            self.CTRL.SHOW_INTERVAL = 1
            self.DATA.IMAGE_MIN_DIM = 320
            self.DATA.IMAGE_MAX_DIM = 512

        # set folder
        self.MISC.RESULT_FOLDER = os.path.join(
            'results', self.CTRL.CONFIG_NAME.lower(), self.CTRL.PHASE)

        if not os.path.exists(self.MISC.RESULT_FOLDER):
            os.makedirs(self.MISC.RESULT_FOLDER)

        # MUST be left at the end
        # Input image size
        self.DATA.IMAGE_SHAPE = np.array(
            [self.DATA.IMAGE_MAX_DIM, self.DATA.IMAGE_MAX_DIM, 3])

        # Compute backbone size from input image size
        self.MODEL.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.DATA.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.DATA.IMAGE_SHAPE[1] / stride))]
             for stride in self.MODEL.BACKBONE_STRIDES])

    def display(self, log_file):
        """Display Configuration values."""
        now = datetime.datetime.now()
        print_log('\nStart timestamp: {:%Y%m%dT%H%M}'.format(now), file=log_file, init=True)
        print_log("Configurations:", file=log_file)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                value = getattr(self, a)
                if isinstance(value, AttrDict):
                    print_log("{}:".format(a), log_file)
                    for _, key in enumerate(value):
                        print_log("\t{:30}\t\t{}".format(key, value[key]), log_file)
                else:
                    print_log("{}\t{}".format(a, value), log_file)
        print_log("\n", log_file)


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """

    def __init__(self, args):
        super(CocoConfig, self).__init__()

        self.CTRL.CONFIG_NAME = args.config_name
        self.CTRL.PHASE = args.phase
        self.CTRL.DEBUG = args.debug

        self.MISC.DEVICE_ID = [int(x) for x in args.device_id.split(',')]
        self.MISC.GPU_COUNT = len(self.MISC.DEVICE_ID)

        # ================ (CUSTOMIZED CONFIG) ======================
        if self.CTRL.CONFIG_NAME == 'all_new':
            self.MODEL.INIT_FILE_CHOICE = 'coco_pretrain'
            self.DATA.IMAGE_MIN_DIM = 256
            self.DATA.IMAGE_MAX_DIM = 320

        elif self.CTRL.CONFIG_NAME == 'base_101':
            self.MODEL.INIT_FILE_CHOICE = 'coco_pretrain'
            self.CTRL.BATCH_SIZE = 8
        elif self.CTRL.CONFIG_NAME == 'base_101_8gpu':
            self.CTRL.BATCH_SIZE = 16
        else:
            print('WARNING: unknown config name!!! use default setting.')

        self._set_value()