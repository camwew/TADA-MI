import os
import pathlib
import torch
import numpy as np
import logging
import cv2
from IPython.utils import io as IPyIO
from tqdm import tqdm
import joblib
import sys
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
from models.hovernet.net_desc_conic import HoVerNetConic
from tiatoolbox.utils.visualization import overlay_prediction_contours


def recur_find_ext(root_dir, ext_list):
    """Recursively find all files in directories end with the `ext` such as `ext='.png'`.
    Args:
        root_dir (str): Root directory to grab filepaths from.
        ext_list (list): File extensions to consider.
    Returns:
        file_path_list (list): sorted list of filepaths.
    """
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list
    
def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return

def process_segmentation(np_map, hv_map, tp_map, model):
    # HoVerNet post-proc is coded at 0.25mpp so we resize
    np_map = cv2.resize(np_map, (0, 0), fx=2.0, fy=2.0)
    hv_map = cv2.resize(hv_map, (0, 0), fx=2.0, fy=2.0)
    tp_map = cv2.resize(
                    tp_map, (0, 0), fx=2.0, fy=2.0,
                    interpolation=cv2.INTER_NEAREST)

    inst_map = model._proc_np_hv(np_map[..., None], hv_map)
    inst_dict = model._get_instance_info(inst_map, tp_map)

    # Generating results match with the evaluation protocol
    type_map = np.zeros_like(inst_map)
    inst_type_colours = np.array([
        [v['type']] * 3 for v in inst_dict.values()
    ])
    type_map = overlay_prediction_contours(
        type_map, inst_dict,
        line_thickness=-1,
        inst_colours=inst_type_colours)

    pred_map = np.dstack([inst_map, type_map])
    # The result for evaluation is at 0.5mpp so we scale back
    pred_map = cv2.resize(
                    pred_map, (0, 0), fx=0.5, fy=0.5,
                    interpolation=cv2.INTER_NEAREST)
    return pred_map


def run(PRETRAINED, save_paths):
    model = HoVerNetConic(num_types=7)
    pretrained = torch.load(PRETRAINED)

    weights = {}
    for key in pretrained['desc'].keys():
        #if 'tp_source' in key:
        #    continue
        if '_feat' in key:
            weights[key.replace('_feat', '')] = pretrained['desc'][key]
        elif 'tp_layer' in key and 'source' in key:
            continue
        elif 'tp_layer' in key:
            weights[key.replace('tp_layer', 'decoder.tp.u0.2')] = pretrained['desc'][key]
        elif 'np_layer' in key:
            weights[key.replace('np_layer', 'decoder.np.u0.2')] = pretrained['desc'][key]
        elif 'hv_layer' in key:
            weights[key.replace('hv_layer', 'decoder.hv.u0.2')] = pretrained['desc'][key]
        else:
            weights[key] = pretrained['desc'][key]
            #weights[key.replace('module.', '')] = pretrained['desc'][key]
    
    model.load_state_dict(weights)

    predictor = SemanticSegmentor(
        model=model,
        num_loader_workers=2,
        batch_size=6,
    )
    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {'units': 'baseline', 'resolution': 1.0},
        ],
        output_resolutions=[
            {'units': 'baseline', 'resolution': 1.0},
            {'units': 'baseline', 'resolution': 1.0},
            {'units': 'baseline', 'resolution': 1.0},
        ],
        save_resolution={'units': 'baseline', 'resolution': 1.0},
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
    )
    logger = logging.getLogger()
    logger.disabled = True
    
    OUT_DIR = 'Please type in the path towards the test folder'

    #infer_img_paths = recur_find_ext(f'{OUT_DIR}/images/', ['.png'])
    infer_img_paths = [OUT_DIR + 'images/' + str(i + 1) + '_img.png' for i in range(len(os.listdir(OUT_DIR + 'images/')))]

    #rmdir(f'{OUT_DIR}/raw/')
    save_paths_each = save_paths + 'subfile/'
    # capture all the printing to avoid cluttering the console

    with IPyIO.capture_output() as captured:
        output_file = predictor.predict(
            infer_img_paths,
            masks=None,
            mode='tile',
            on_gpu=True,
            ioconfig=ioconfig,
            crash_on_exception=True,
            save_dir=save_paths_each
        )


    output_file = save_paths_each + 'file_map.dat'
    output_info = joblib.load(output_file)

    semantic_predictions = []
    composition_predictions = []
    for input_file, output_root in tqdm(output_info):
        img = cv2.imread(input_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np_map = np.load(f'{output_root}.raw.0.npy')
        hv_map = np.load(f'{output_root}.raw.1.npy')
        tp_map = np.load(f'{output_root}.raw.2.npy')

        pred_map = process_segmentation(np_map, hv_map, tp_map, model)
        semantic_predictions.append(pred_map)
    
    semantic_predictions = np.array(semantic_predictions)

    # Saving the results for segmentation
    np.save(save_paths + 'valid_pred.npy', semantic_predictions)



    semantic_pred = np.load(save_paths + 'valid_pred.npy')

    semantic_true_all = 'Please type in the path for the test_mask file'


    output_file = save_paths_each + 'file_map.dat'
    output_info = joblib.load(output_file)

    draw_save_path = save_paths + 'vis/'
    os.makedirs(draw_save_path, exist_ok = True)

    SEED = 5
    np.random.seed(SEED)
    selected_indices = np.random.choice(599, 30)

    PERCEPTIVE_COLORS = [
        (  0,   0,   0),
        (255, 0,   0),
        (  0, 255,   0),
        (0,   0,   255),
    ]
    PERCEPTIVE_COLORS_Lizard_6 = [
        (  0,   0,   0),
        (255, 165,   0),
        (  0, 255,   0),
        (255,   0,   0),
        (130, 130,   255),
        (  0, 0,   255),
        (255,   255,   0),
    ]
    import matplotlib.pyplot as plt
    for idx in selected_indices:
        img = cv2.imread(output_info[idx][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inst_map = semantic_pred[idx][..., 0]
        type_map = semantic_pred[idx][..., 1]
        pred_inst_dict = model._get_instance_info(inst_map, type_map)

        inst_type_colours = np.array([
            PERCEPTIVE_COLORS_Lizard_6[v['type']]
            for v in pred_inst_dict.values()
        ])
        overlaid_pred = overlay_prediction_contours(
            img, pred_inst_dict,
            inst_colours=inst_type_colours,
            line_thickness=1
        )
        #cv2.imwrite(os.path.join(draw_save_path, str(idx + 1) + '_pred.png'), overlaid_pred[...,::-1])
    

        inst_map = semantic_true_all[idx][..., 0]
        type_map = semantic_true_all[idx][..., 1]
        true_inst_dict = model._get_instance_info(inst_map, type_map)

        inst_type_colours = np.array([
            PERCEPTIVE_COLORS_Lizard_6[v['type']]
            for v in true_inst_dict.values()
        ])
        overlaid_true = overlay_prediction_contours(
            img, true_inst_dict,
            inst_colours=inst_type_colours,
            line_thickness=1
        )
        output_fig = np.concatenate((img, overlaid_pred, overlaid_true), 1)
        cv2.imwrite(os.path.join(draw_save_path, str(idx + 1) + '.png'), output_fig[...,::-1])
    '''
    inst_map = semantic_true_all_6[idx][..., 0]
    type_map = semantic_true_all_6[idx][..., 1]
    true_inst_dict_6 = model._get_instance_info(inst_map, type_map)

    inst_type_colours = np.array([
        PERCEPTIVE_COLORS_Lizard_6[v['type']]
        for v in true_inst_dict_6.values()
    ])
    overlaid_true_6 = overlay_prediction_contours(
        img, true_inst_dict_6,
        inst_colours=inst_type_colours,
        line_thickness=1
    )
    output_fig = np.concatenate((img, overlaid_pred, overlaid_true, overlaid_true_6), 1)
    cv2.imwrite(os.path.join(draw_save_path, str(idx + 1) + '.png'), output_fig[...,::-1])
    '''
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(overlaid_pred)
    plt.title('Prediction')
    plt.axis('off')
    plt.show()
    '''











