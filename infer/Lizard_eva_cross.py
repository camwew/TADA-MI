import os
from tqdm.auto import tqdm
import glob
import numpy as np
from metrics import get_pq, get_multi_pq_info, get_dice, get_multi_dice_info

def compute_pq_mpq(pred_path, true_path, nr_classes):
    seg_metrics_names = ["pq", "multi_pq+", "dice", "multi_dice+"]
    # check to make sure input is a single numpy array
    pred_format = pred_path.split(".")[-1]
    true_format = true_path.split(".")[-1]
    if pred_format != "npy" or true_format != "npy":
        raise ValueError("pred and true must be in npy format.")

    # initialise empty placeholder lists
    all_metrics = {}
    pq_list = []
    mpq_info_list = []
    dice_list = []
    mdice_info_list = []
    # load the prediction and ground truth arrays
    pred_array = np.load(pred_path)
    true_array = np.load(true_path)

    nr_patches = pred_array.shape[0]

    for patch_idx in tqdm(range(nr_patches)):
        # get a single patch
        pred = pred_array[patch_idx]
        true = true_array[patch_idx]

        # instance segmentation map
        pred_inst = pred[..., 0]
        true_inst = true[..., 0]
        # classification map

        # ===============================================================

        for idx, metric in enumerate(seg_metrics_names):
            if metric == "pq":
                # get binary panoptic quality
                try:
                    pq = get_pq(true_inst, pred_inst)
                    pq = pq[0][2]
                    pq_list.append(pq)
                except:
                    print(metric, idx)
            elif metric == "multi_pq+":
                # get the multiclass pq stats info from single image
                try:
                    mpq_info_single = get_multi_pq_info(true, pred, nr_classes=nr_classes)
                    mpq_info = []
                    # aggregate the stat info per class
                    for single_class_pq in mpq_info_single:
                        tp = single_class_pq[0]
                        fp = single_class_pq[1]
                        fn = single_class_pq[2]
                        sum_iou = single_class_pq[3]
                        mpq_info.append([tp, fp, fn, sum_iou])
                    mpq_info_list.append(mpq_info)
                except:
                    print(metric, idx)
            elif metric == "dice":
                dice_score = get_dice(true_inst, pred_inst)[2]
                dice_list.append(dice_score)
            elif metric == "multi_dice+":
                mdice_info_single = get_multi_dice_info(true, pred, nr_classes=nr_classes)
                mdice_info = []
                for single_class_dice in mdice_info_single:
                    inter_pixel = single_class_dice[0]
                    total_pixel = single_class_dice[1]
                    mdice_info.append([inter_pixel, total_pixel])
                mdice_info_list.append(mdice_info)
            else:
                raise ValueError("%s is not supported!" % metric)

    pq_metrics = np.array(pq_list)
    pq_metrics_avg = np.mean(pq_metrics, axis=-1)  # average over all images
    if "multi_pq+" in seg_metrics_names:
        mpq_info_metrics = np.array(mpq_info_list, dtype="float")
        # sum over all the images
        total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)
    dice_metrics = np.array(dice_list)
    dice_metrics_avg = np.mean(dice_metrics, axis=-1)  # average over all images
    if "multi_dice+" in seg_metrics_names:
        mdice_info_metrics = np.array(mdice_info_list, dtype="float")
        # sum over all the images
        total_mdice_info_metrics = np.sum(mdice_info_metrics, axis=0)

    for idx, metric in enumerate(seg_metrics_names):
        if metric == "multi_pq+":
            mpq_list = []
            # for each class, get the multiclass PQ
            for cat_idx in range(total_mpq_info_metrics.shape[0]):
                total_tp = total_mpq_info_metrics[cat_idx][0]
                total_fp = total_mpq_info_metrics[cat_idx][1]
                total_fn = total_mpq_info_metrics[cat_idx][2]
                total_sum_iou = total_mpq_info_metrics[cat_idx][3]

                # get the F1-score i.e DQ
                dq = total_tp / ((total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6)
                # get the SQ, when not paired, it has 0 IoU so does not impact
                sq = total_sum_iou / (total_tp + 1.0e-6)
                mpq_list.append(dq * sq)
            mpq_metrics = np.array(mpq_list)
            all_metrics['class_pq'] = mpq_metrics
            all_metrics[metric] = [np.mean(mpq_metrics)]
        elif metric == "pq":
            all_metrics[metric] = [pq_metrics_avg]
        elif metric == "dice":
            all_metrics[metric] = [dice_metrics_avg]
        elif metric == "multi_dice+":
            mdice_list = []
            # for each class, get the multiclass DICE
            for cat_idx in range(total_mdice_info_metrics.shape[0]):
                assert total_mdice_info_metrics[cat_idx].shape[0] == 2, "?"
                total_inter_pixel = total_mdice_info_metrics[cat_idx][0]
                total_total_pixel = total_mdice_info_metrics[cat_idx][1]

                # get the DICE-score
                total_dice = 2 * total_inter_pixel / (total_total_pixel + 1.0e-6)
                mdice_list.append(total_dice)
            mdice_metrics = np.array(mdice_list)
            all_metrics['class_dice'] = mdice_metrics
            all_metrics[metric] = [np.mean(mdice_metrics)]
            
    return all_metrics
        
