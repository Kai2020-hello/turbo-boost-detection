from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from tools.utils import print_log

import numpy as np


def prepare_data(config, dataset, results, log_file):

    print_log('\tUse a few? {}'.format(config.TSNE.A_FEW), log_file)
    print_log('\tsample choice is [ {} ]'.format(config.TSNE.SAMPLE_CHOICE), log_file)
    choice = config.TSNE.SAMPLE_CHOICE

    # Step 0 - get all data
    num_results = len(results)
    feat_dim = results[0]['feature'].shape[0] - 1

    stats = {'sample_num': np.zeros((80, 3), dtype=np.int32)}
    data = np.zeros((num_results, feat_dim), dtype=np.float32)
    y = np.zeros(num_results, dtype=np.int32)  # label, start from 1 - 80
    area_stats = np.zeros(num_results, dtype=np.float32)

    for ind, inst in enumerate(results):
        cat_id = inst['category_id']
        internal_id = dataset.map_source_class_id('coco.{}'.format(cat_id))  # 1-80
        stats['sample_num'][internal_id-1, 0] += 1
        area = inst['feature'][-1]    # 0-1
        score = inst['score']  # to split more accurate

        data[ind, :] = inst['feature'][:-1]
        y[ind] = internal_id
        area_stats[ind] = area

    print_log('\t[ALL DATA] median area size is {:.5f}, min {:.5f}, max {}, among {} samples.'.format(
        np.median(area_stats), np.min(area_stats), np.max(area_stats), len(results)
    ), log_file)
    # for num in stats['sample_num'][:, 0]:
    #     print(num)

    # Step 1 - generate data and y
    if choice == 'set1':
        cls_list = [1, 57, 3, 74, 40, 42, 46, 61, 55, 10]
        label_list = ['person', 'chair', 'car', 'book', 'bottle',
                      'cup', 'bowl', 'dining table', 'donut', 'traffic light']
        sample_per_cls = 100
        small_percent = 0.3
        large_percent = 0.7

        for ind, curr_cls in enumerate(cls_list):
            curr_data = data[y == curr_cls]
            curr_data_area = area_stats[y == curr_cls]

            _chosen_ind = np.random.choice(curr_data.shape[0], sample_per_cls)

            curr_data = curr_data[_chosen_ind]
            curr_data_area = curr_data_area[_chosen_ind]

            sorted_area = np.sort(curr_data_area)
            small_box_thres = sorted_area[int(np.floor(small_percent * len(_chosen_ind)))]
            large_box_thres = sorted_area[int(np.floor(large_percent * len(_chosen_ind)))]
            curr_box_size = np.zeros(len(_chosen_ind), dtype=np.int32)
            curr_box_size[curr_data_area <= small_box_thres] = -1
            curr_box_size[curr_data_area >= large_box_thres] = 1

            # NOTE: we change the label here: the list index is the new label
            curr_label = (ind+1) * np.ones(curr_data.shape[0])

            # accumulate data
            if ind > 0:
                new_data = np.vstack((new_data, curr_data))
                new_y = np.hstack((new_y, curr_label))
                new_box_size = np.hstack((new_box_size, curr_box_size))
            else:
                # init new data here
                new_data = curr_data
                new_y = curr_label
                new_box_size = curr_box_size

        print_log('\t[NEW DATA] total_cls: {}, sample_per_cls: {}'.format(len(cls_list), sample_per_cls), log_file)

    # Step 2
    n_points = len(new_y)
    metric = config.TSNE.METRIC
    perplexity = config.TSNE.PERPLEXITY
    # TODO: transfer to GPU tensor

    # TODO(mid): change to other metric other than euclidean
    dist2 = pairwise_distances(new_data, metric=metric, squared=True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(dist2, perplexity, False)
    # Convert to n x n prob array
    pij2 = squareform(pij)

    i, j = np.indices(pij2.shape)
    i, j = i.ravel(), j.ravel()
    out_p_ij = pij2.ravel().astype('float32')
    # remove self-indices
    idx = i != j
    i, j, out_p_ij = i[idx], j[idx], out_p_ij[idx]
    print_log('\t[NEW DATA] Done! Ready to train!', log_file)

    return n_points, out_p_ij, i, j, new_y, new_box_size, label_list
