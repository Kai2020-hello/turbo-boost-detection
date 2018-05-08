"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import colorsys
import itertools
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from tools import utils
from scipy.misc import imread
from tools.utils import print_log, mkdirs

if "DISPLAY" not in os.environ:
    plt.switch_backend('agg')


############################################################
#  Visualization
############################################################
def display_images(images, titles=None, cols=4, cmap=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = \
            np.where(mask == 1,
                     image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                     image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names, scores=None, title="", figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    # plt.show()
    # plt.imshow()
    

def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores, overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictins and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with differnt
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominant each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            x = random.randint(x1, (x1 + x2) // 2)
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def plot_loss(loss, val_loss, save=True, log_dir=None):
    plt.figure("loss")
    plt.gcf().clear()
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)


class Visualizer(object):
    def __init__(self, opt, train_data, val_data):
        self.opt = opt
        if self.opt.MISC.USE_VISDOM:
            import visdom
            self.vis = visdom.Visdom(port=opt.MISC.VIS.PORT, env=opt.CTRL.CONFIG_NAME)
            self.dis_im_cnt, self.dis_im_cycle = 0, 4
            self.loss_data = {'X': [], 'Y': [], 'legend': self.MISC.VIS.LOSS_LEGEND}
            # for visualization
            # TODO: visualize in the training process
            self.num_classes = dataset.num_classes
            self.class_name = dataset.COCO_CLASSES_names
            self.color = plt.cm.hsv(np.linspace(0, 1, (self.num_classes-1))).tolist()
            # for both train and test
            self.save_det_res_path = os.path.join(self.opt.save_folder, 'det_result')
            mkdirs(self.save_det_res_path)

    def plot_loss(self, errors, progress, others=None):
        """draw loss on visdom console"""
        #TODO: set figure height and width in visdom
        loss, loss_l, loss_c = errors[0].data[0], errors[1].data[0], errors[2].data[0]
        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        x_progress = epoch + float(iter_ind/epoch_size)

        self.loss_data['X'].append([x_progress, x_progress, x_progress])
        self.loss_data['Y'].append([loss, loss_l, loss_c])

        self.vis.line(
            X=np.array(self.loss_data['X']),
            Y=np.array(self.loss_data['Y']),
            opts={
                'title': 'Train loss over epoch',
                'legend': self.loss_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.opt.MISC.VIS.LINE
        )

    def print_loss(self, errors, progress, others=None):
        """show loss info in console"""
        loss, loss_l, loss_c = errors[0].data[0], errors[1].data[0], errors[2].data[0]
        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        t0, t1 = others[0], others[1]
        msg = '[{:s}]\tepoch/iter [{:d}/{:d}][{:d}/{:d}] ||\t' \
              'Loss: {:.4f}, loc: {:.4f}, cls: {:.4f} ||\t' \
              'Time: {:.4f} sec/image'.format(
                self.opt.experiment_name, epoch, self.opt.max_epoch, iter_ind, epoch_size,
                loss, loss_l, loss_c, (t1 - t0)/self.opt.batch_size)
        print_log(msg, self.opt.file_name)

    def print_info(self, progress, others):
        """print useful info on visdom console"""
        #TODO: test case and set size

        epoch, iter_ind, epoch_size = progress[0], progress[1], progress[2]
        still_run, lr, time_per_iter = others[0], others[1], others[2]

        left_time = time_per_iter * (epoch_size-1-iter_ind + (self.opt.max_epoch-1-epoch)*epoch_size) / 3600 if \
            still_run else 0
        status = 'RUNNING' if still_run else 'DONE'
        dynamic = 'start epoch: {:d}, iter: {:d}<br/>' \
                  'curr lr {:.8f}<br/>' \
                  'progress epoch/iter [{:d}/{:d}][{:d}/{:d}]<br/><br/>' \
                  'est. left time: {:.4f} hours<br/>' \
                  'time/image: {:.4f} sec<br/>'.format(
                    self.opt.start_epoch, self.opt.start_iter,
                    lr,
                    epoch, self.opt.max_epoch, iter_ind, epoch_size,
                    left_time, time_per_iter/self.opt.batch_size)
        common_suffix = '<br/><br/>-----------<br/>batch_size: {:d}<br/>' \
                        'optim: {:s}<br/>'.format(
                            self.opt.batch_size, self.opt.optim)

        msg = 'phase: {:s}<br/>status: <b>{:s}</b><br/>'.format(self.opt.phase, status)\
              + dynamic + common_suffix
        self.vis.text(msg, win=self.dis_win_id_txt)

    def show_image(self, progress, others=None):
        """for test, print log info in console and show detection results on visdom"""
        if self.opt.phase == 'test':
            name = os.path.basename(os.path.dirname(self.opt.det_file))
            i, total_im, test_time = progress[0], progress[1], progress[2]
            all_boxes, im, im_name = others[0], others[1], others[2]

            print_log('[{:s}][{:s}]\tim_detect:\t{:d}/{:d} {:.3f}s'.format(
                self.opt.experiment_name, name, i, total_im, test_time), self.opt.file_name)

            dets = np.asarray(all_boxes)
            result_im = self._show_detection_result(im, dets[:, i], im_name)
            result_im = np.moveaxis(result_im, 2, 0)
            win_id = self.dis_win_id_im + (self.dis_im_cnt % self.dis_im_cycle)
            self.vis.image(result_im, win=win_id,
                           opts={
                               'title': 'subfolder: {:s}, name: {:s}'.format(
                                   os.path.basename(self.opt.save_folder), im_name),
                               'height': 320,
                               'width': 400,
                           })
            self.dis_im_cnt += 1

    def _show_detection_result(self, im, results, im_name):

        plt.figure()
        plt.axis('off')     # TODO, still the axis remains
        plt.imshow(im)
        currentAxis = plt.gca()

        for cls_ind in range(1, len(results)):
            if results[cls_ind] == []:
                continue
            else:

                cls_name = self.class_name[cls_ind-1]
                cls_color = self.color[cls_ind-1]
                inst_num = results[cls_ind].shape[0]
                for inst_ind in range(inst_num):
                    if results[cls_ind][inst_ind, -1] >= self.opt.visualize_thres:

                        score = results[cls_ind][inst_ind, -1]
                        pt = results[cls_ind][inst_ind, 0:-1]
                        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                        display_txt = '{:s}: {:.2f}'.format(cls_name, score)

                        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=cls_color, linewidth=2))
                        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': cls_color, 'alpha': .5})
                    else:
                        break
        result_file = '{:s}/{:s}.png'.format(self.save_det_res_path, im_name[:-4])

        plt.savefig(result_file, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        # ref: https://github.com/facebookresearch/visdom/issues/119
        # plotly_fig = tls.mpl_to_plotly(fig)
        # self.vis._send({
        #     data=plotly_fig.data,
        #     layout=plotly_fig.layout,
        # })
        result_im = imread(result_file)
        return result_im
