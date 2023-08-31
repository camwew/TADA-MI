import numpy as np
from skimage import measure
import os
import tifffile as tiff
import scipy.io as sio

##########################################################################
# General Usage
##########################################################################


def generate_class_label(pred_root):

    test_mask_root = pred_root + 'valid_pred.npy'
    out_gt_root = pred_root + 'pred_label_class/'
    
    os.makedirs(out_gt_root, exist_ok = True)

    test_mask = np.load(test_mask_root)

    for ind in range(test_mask.shape[0]):
        test_mask_slice = test_mask[ind]
        test_mask_slice_inst_map = test_mask_slice[..., 0]
        test_mask_slice_class_map = test_mask_slice[..., 1]
        box_centroids = []
        labels_np = []
        for ins in np.unique(test_mask_slice_inst_map):
            if ins == 0:
                continue
            inverse_centroid = np.mean(np.argwhere(test_mask_slice_inst_map == ins), axis=0)
            centroid = [inverse_centroid[1], inverse_centroid[0]]
            cat = np.unique(test_mask_slice_class_map[test_mask_slice_inst_map == ins])
            assert cat.shape[0] == 1, "More than one class value!!!"
            box_centroids.append(centroid)
            labels_np.append(cat[0])
        box_centroids = np.array(box_centroids)
        labels_np = np.expand_dims(np.array(labels_np), 1)
        class_mat_name = str(ind + 1) + '.mat'
        class_dict = {'inst_centroid': box_centroids, 'inst_type': labels_np}
        sio.savemat(os.path.join(out_gt_root, class_mat_name), class_dict)
        
        
