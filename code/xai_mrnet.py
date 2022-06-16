'''
Model Inference:
- Load data
- Load model
- Get heatmaps
- Evaluate on heatmaps

Based on submit_prediction.py
'''
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import os
import pickle
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

# XAI import
import logging
from xai.heatmap_eval import BaseXAIevaluator, Acc_Drop
from xai.heatmap_utlis import *
from utils.parse_config import ConfigParser
from utils.util import prepare_device
from xai.modality_ablation import shapley_result_csv, get_shapley, compute_xai_mod_shapley_corr, aggregate_msfi, compute_mfsi
# model and data import
sys.path.append('/local-scratch/authorid/BRATS_IDH/MRNet')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MRNet.model import TripleMRNet
from MRNet.loader import load_data
import csv
from tqdm import tqdm
import model.metric as module_metric
import itertools, math
from sklearn.metrics import jaccard_score

import gc
import glob
# global variables
INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

def modality_shapley_mrnet(config, model, save_path, fold, device, task, machine, timestamp=True):
    modalities = config['xai']['modality']
    print(modalities)
    # generate modality combinations
    N_sets = list(itertools.product([0, 1], repeat=len(modalities)) ) # set of all_combinations
    for modality_selection in N_sets:
        print(modality_selection, 'modality_selection')
        ablated_dataloader = load_data(task=task, use_gpu=True, test_loader = True, machine = machine, input_modality = modality_selection)
        get_test_result(model, ablated_dataloader, save_path, fold, device, timestamp, modality_selection = modality_selection)


def get_test_result(model, dataloader, save_path, fold, device, timestamp = True, modality_selection = None):
    metric_fns = [getattr(module_metric, met) for met in ["accuracy", "f1"]]
    total_metrics = torch.zeros(len(metric_fns))
    if modality_selection:
        mod_fn = '-'.join([str(i) for i in modality_selection])
        csv_filename = save_path / 'cv_result_fold_{}{}.csv'.format(fold,  '-'+mod_fn)
    else:
        csv_filename = save_path / 'cv_result_fold_{}.csv'.format(fold)
    fnames = ['dataID',  'gt', 'pred']
    file_exists = os.path.isfile(csv_filename)
    if (not timestamp) and file_exists :# or save_inputs):
        logging.info("{} file exist, pass\n".format(csv_filename))
        return
    if timestamp:
        dateTimeObj = datetime.now().strftime("%Y%m%d_%H%M")
        csv_filename = csv_filename.parent/ '{}-{}'.format(dateTimeObj, csv_filename.name)
    logging.info("CV will be saved at: {}".format(csv_filename))
    with torch.no_grad():
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
            if timestamp or (not file_exists):
                csv_writer.writeheader()
            for idx, data in enumerate(tqdm(dataloader)):
                images = data[0], data[1], data[2]
                target, dataIDs = data[3], data[4]
                images_val = []
                for img in images:
                    images_val.append(img.to(device))

                model.to(device)
                pred = model.forward(*images_val)
                pred = torch.sigmoid(pred)
                pred_npy = pred.data.cpu().numpy()[0][0]

                csv_record = {'dataID': dataIDs[0], 'gt': target.cpu().numpy()[0][0],
                              'pred': pred_npy}
                csv_writer.writerow(csv_record)
                print(csv_record)




def get_model(model_path, device = 'gpu', backbone="alexnet"):
    meniscal_model = TripleMRNet(backbone=backbone, training=False)
    print('count', torch.cuda.device_count())
    # print(meniscal_model)
    print(model_path)
    state_dict = torch.load(model_path)
    meniscal_model.load_state_dict(state_dict)
    meniscal_model.to(device)
    meniscal_model.eval()
    return meniscal_model

def get_hm(config, data_loader, model, fold=0):
    '''
    Generate heatmaps
    :param config:
    :param data_loader:
    :param model:
    :param fold: seed number for the model
    :return:
    '''
    # setup xai evaluator, load model and data
    xai_eval = BaseXAIevaluator(config, load_model= False, load_data = False)
    result_dir = xai_eval.generate_save_heatmap(data_loader, model, 'MRNet_get_hm_fold_{}'.format(fold))
    logging.info(result_dir)
    return result_dir


def get_save_modality_hm_value_mrnet(hm_save_dir, result_save_dir, fold, method_list, penalize= False, portion_metrics= True, positiveHMonly = True,  segment_path = None, modalities= ["t1", "t1ce", "t2", "flair"]):
    '''
    Since read hm is time consuming, read and save the hm values for each method
    :param hm_save_dir:
    :param method_list:
    :param shapley_csv:
    :param localize_feature: if True, calculate the sum of hm values using lesion masks.
    Get positive+negative values inside tumor regions, minus positive values outside tumor regions
    (penalty for positive values outside tumor)
    :return:
    '''

    columns = modalities+ ['XAI', 'dataID']
    if segment_path:
        columns += ['iou_{}'.format(m) for m in modalities]
    for method in method_list:
        result_csv = Path(result_save_dir) / 'modalityHM_fold-{}-{}.csv'.format(fold, method)
        file_exists = os.path.isfile(result_csv)
        if file_exists:
            print("{} exists, pass".format(method))
            continue
        result_df = pd.DataFrame(columns=columns)
        value = {}
        # post-process hms
        # print(method)
        if type(hm_save_dir) == dict: #heatmap shortcut experiment
            hm_dict_fold = dict()
            for sd in hm_save_dir:
                hm_dict, data_record = get_heatmaps(hm_save_dir[sd], method, by_data=False, hm_as_array=False, return_mri=False)
                hm_dict_fold[sd] = hm_dict
        else:
            hm_dict, data_record = get_heatmaps(hm_save_dir, method, by_data=False, hm_as_array=False, return_mri=False)
        print("Number of data for {}: {}".format(method, len(hm_dict.keys())))  # , hm_dict.keys())
        # print(hm_dict.keys())
        for dataID, hm in hm_dict.items():
            # print(dataID, hm[0].shape) # 1187 (37, 3, 224, 224)
            # print(hm[0].min(), hm[0].max(), dataID) # -0.0077733933 0.0071549467 1187
            if type(hm_save_dir) == dict:  # heatmap shortcut experiment
                hm_fold = [] # hm for dataID across different seed models
                for fold_id, hm_dict in hm_dict_fold.items():
                    hm = hm_dict[dataID]
                    hm_fold.append(hm)
                print('average {} heatmaps'.format(len(hm_fold)))
                mean_hm_list = []
                for i in range(len(modalities)):
                    mean_hm = np.mean( np.array([ hm_fold[fd][i] for fd in range(len(hm_fold))]), axis=0 )
                    # print(mean_hm.shape, 'mean_hm') # (27, 3, 224, 224)
                    mean_hm_list.append(mean_hm)
                hm = mean_hm_list
            post_hm = postprocess_heatmaps_mrnet(hm, no_neg=True)
            # print(len(post_hm), post_hm[0].min(), post_hm[0].max(), segment_path)
            # post_hm = postprocess_heatmaps(hm, no_neg=positiveHMonly) # (C, H,W,D) # the postprocessed hm is already non-negative
            if  segment_path:
                gt_list = []
                for m in modalities:
                    seg_path = os.path.join(segment_path, '{}_{}.npy'.format(dataID, m))
                    seg = np.load(seg_path)# .get_fdata()
                    # crop the seg maps the same as in loader.py (25, 256, 256) --> (25, 244, 244)
                    pad = int((256 - 224) / 2)
                    seg = seg[:, pad:-pad, pad:-pad]
                    gt_list.append(seg)
                # seg = np.rot90(seg, k=3, axes=(0, 1)) # important, avoid bug of seg, saliency map mismatch
                hm_values, ious = get_modality_feature_hm_value_mrnet(post_hm, gt_list, penalize=False, portion = portion_metrics)
                # print(hm_values, ious)
            else:
                hm_values = []
                for k in range(len(modalities)):
                    hm_values.append(np.sum(post_hm[k]))
                # hm_values = np.sum(post_hm, axis = tuple([i for i in range(len(modalities))][1:]))
                print('heatmap modality sum', hm_values)
            # print(method, dataID, corr, p_value)
            value["XAI"] = method
            value['dataID'] = dataID
            for i, mod in enumerate(modalities):
                value[mod] = hm_values[i]
                if  segment_path:
                    value['iou_{}'.format(mod)] = ious[i]
            result_series= pd.Series(value, index=columns)
            result_df= result_df.append(result_series, ignore_index=True)
            # print(result_df)
    # result_df = pd.DataFrame.from_dict(result, orient = 'index')
        result_df.to_csv(result_csv)
        print("modalityHM Saved at: {}".format(result_csv))
        if type(hm_save_dir) == dict: #heatmap shortcut experiment
            print('shortcut experiment done for {}'.format(method))
            del hm_dict_fold
        del hm_dict
        gc.collect()
    return result_csv

def get_modality_feature_hm_value_mrnet(post_hm, segs, penalize = True, portion = True):
    """
    Get positive+negative values inside tumor regions, minus positive values outside tumor regions
    (penalty for positive values outside tumor)
    :param post_hm: np array of the same shape with seg
    :param penalize: old parameter. No longer needed with new parameter portion
    :return:
    """
    assert segs[0].shape == post_hm[0].shape, "segmentation map shape {} and processed hm shape {} does not match!".format(segs[0].shape, post_hm[0].shape)
    # binary_seg = seg[seg>0]
    # edge = 20
    # dilated_seg = []
    # for s in range(seg.shape[-1]):
    #     dilated= binary_dilation(seg[:,:,s], selem = np.ones([edge for i in range(seg[:,:,s].ndim)]))
    #     dilated_seg.append(dilated)
    # dilated_seg = np.stack(dilated_seg, axis = -1)
    # print((seg>0).sum()/seg.size, (dilated_seg>0).sum()/dilated_seg.size, dilated_seg.shape)
    hm_values = [] # feature portion and MSFI
    ious = []
    for i, hm in enumerate(post_hm):
        seg = segs[i]
        feature = hm[(seg>0) & (hm>0)]
        non_feature = hm[(seg==0) & (hm>0)]
        if portion:
            v = feature.sum() / ( feature.sum() + non_feature.sum() )
            if (v < 0):
                print( feature.sum() , feature.shape, non_feature.shape,non_feature.sum())
        else:
            v = feature.sum()
            if penalize:
                v -= non_feature.sum()
        hm_values.append(v)
        # calculate iou
        lesion_flatten = (seg == 1).astype(int).reshape(-1)
        hm_flatten = hm.astype(int).reshape(-1)
        iou = jaccard_score(lesion_flatten, hm_flatten)
        ious.append(iou)
    # print(hm_values, np.sum(post_hm, axis = tuple([i for i in range(4)][1:])), '\n')
    return hm_values, ious


def compare_with_gt_dataset_mrnet(save_dir, seg_path, method_list, modalities):
    """
    Calculate feature portion metric FP.
    Compare the hm with gt segmentation maps. The same function in heatmap_eval, but modified for MRNet heatmap data
    :param save_dir: heatmap saved dir. Also used to save the computed csv files.
    :param seg_path: segmentation mask dir
    :return:
    """
    # data_ids, df = get_data_ids(save_dir)
    # heatmap compare with seg map gt
    # before_file = glob.glob(os.path.join(save_dir, '*before.pkl'))
    # assert len(before_file) == 1, print('ERROR! multiple before files {} in {}'.format(before_file, pickle_path))
    # heatmap_dict = pickle.load(open(before_file[0], "rb"))
    gt_compare_col = None
    for m in method_list:
        # check if already have such files saved
        fn = os.path.join(save_dir, "compareGTids_data*-{}-*.csv".format(m))
        if glob.glob(fn):
            print("{} exists, pass!".format(fn))
            continue
        else:
            print("Working on {}".format(fn))
        gt_results = list()
        # post-process hms
        hm_dict, data_record = get_heatmaps(save_dir, m, by_data=False, hm_as_array=False, return_mri=False)
        print("Number of data to be evaluated:", len(hm_dict.keys()))  # , hm_dict.keys())
        new_hm_dict = {}
        for dataID, hm in hm_dict.items():
            post_hm = postprocess_heatmaps_mrnet(hm, no_neg = True) # modality hm in a list
            # if post_hm.min() == post_hm.max():
            #     print("Warning, for data {}, min {} == max {}".format(dataID, post_hm.min(), post_hm.max()))
            #     continue
            # print(dataID, hm.shape, post_hm.shape)
            # new_hm_dict[dataID] = post_hm
        # hms, _ = get_heatmaps(save_dir, m, by_data=False, hm_as_array=False)
        # for bratsID in data_ids:
        #     seg = read_seg(seg_path, dataID)
            # load seg mask
            gt_list = []
            for mod in modalities:
                seg_fn = os.path.join(seg_path, '{}_{}.npy'.format(dataID, mod))
                seg = np.load(seg_fn)  # .get_fdarta()
                # print('seg path', seg_fn, seg.shape)
                # crop the seg maps the same as in loader.py (25, 256, 256) --> (25, 244, 244)
                pad = int((256 - 224) / 2)
                seg = seg[:, pad:-pad, pad:-pad]
                gt_list.append(seg)
            # seg = np.rot90(seg, k=3, axes=(0, 1))  # important, avoid bug of seg, saliency map mismatch
            # print(post_hm.shape) #(4, 240, 240, 155)
            # print(gt_list[0].shape, post_hm[0].shape, 'seg hm shape') # (40, 224, 224) (40, 224, 224) seg hm shape
            hm_np = np.concatenate(post_hm)
            seg_np = np.concatenate(gt_list)
            assert hm_np.shape == seg_np.shape, "{} hm {} and seg shape {} are not the same".format(dataID, hm_np.shape, seg_np.shape)
            assert hm_np.max()<=1 and hm_np.min()>=0 , "hm max {}, min {}, is not in 0-1".format(hm_np.max(), hm_np.min())
            fp = hm_np[seg_np == 1 ].sum() / hm_np.sum()
            print('fp is', fp, 'for data', dataID, 'seg unique', np.unique(seg_np), 'shape', hm_np.shape, seg_np.shape)
            # return

            # gt_r, gt_compare_col = compare_with_gt(seg, post_hm) # [tumor_port + IoU, 4 mods]
            gt_results.append(fp)
            # print(np.array(gt_r).shape) # (7, 4)

            # vis for sanity check
            # dateTimeObj = datetime.now()
            # time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
            # img_name = 'fp_iou-{}-{}-{}'.format(m, dataID, time_stamp)
            # vis_dir = image3d_to_gif(save_dir = '/local-scratch/authorid/vis_brats/{}'.format(img_name),
            #                          heatmaps = post_hm, seg = seg, keep_png = True)
            # return
            # vis for sanity check, end

        # aggregate gt score for all datapoints
        gt_results = np.array(gt_results) # 3d array [bratsID, tumor_port + IoU, 4 mods]
        # print(gt_results.shape, gt_results) # (74, 7, 4)
        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        # pickle.dump(gt_results, open(os.path.join(save_dir, 'compareGTabsCUTOFF_data{}-{}-{}.pkl'.format(len(hm_dict.keys()), m, time_stamp)), 'wb'))
        # previous abs cut off not working as the quantile cutoff, so switch back to quantile cutoff to get binary hm
        pickle.dump(gt_results, open(os.path.join(save_dir, 'compareGT_data{}-{}-{}.pkl'.format(len(hm_dict.keys()), m, time_stamp)), 'wb'))
        # save dataID list for records
        dataIDs = pd.Series(hm_dict.keys())
        dataIDs.to_csv(os.path.join(save_dir, 'compareGTids_data{}-{}-{}.csv'.format(len(hm_dict.keys()), m, time_stamp)))

        # post-process results, skip
        # mean_allmod = np.mean(gt_results, axis=(0,2)) # a list of tumor_port + IoU metrics
        # std_allmod = np.std(gt_results, axis=(0,2))
        # # sum all bratsid to 2d array
        # gt_results = np.mean(gt_results, axis=(0))
        # gt_results_std = np.std(gt_results, axis=(0))
        # # pd dataframe [tumor_port + IoU, 4 mods]
        # cols = self.modality
        # rows = gt_compare_col
        # df = pd.DataFrame(gt_results, columns = cols, index = rows)
        # df_std = pd.DataFrame(gt_results_std, columns = cols, index = rows)
        # metric_allmod = pd.DataFrame([mean_allmod, std_allmod], index=['mean', 'std'], columns=gt_compare_col)
        #
        # # save the results to csv
        # # get filename w/ timestamp
        # dateTimeObj = datetime.now()
        # time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        # gt_filename = os.path.join(save_dir, 'compareGT-{}-{}.csv'.format(m, time_stamp))
        # metric_allmod_filename = os.path.join(save_dir, 'compareGT_allmod-{}-{}.csv'.format(m, time_stamp))
        # # save to csv
        # df.to_csv(gt_filename)
        # df_std.to_csv(os.path.join(save_dir, 'compareGT_std-{}-{}.csv'.format(m, time_stamp)))
        # metric_allmod.to_csv(metric_allmod_filename)
        print("Saved results: {}".format(m))

    # return df, metric_allmod

def get_hm_pipeline():
    parser = get_parser()
    args = parser.parse_args()
    task = args.task
    fold = args.seed
    job = args.job
    machine = args.machine
    model_path = args.model_path
    backbone = args.backbone
    prefix = 'trained_model/MRNet/'
    if machine == 'cc':
        prefix == ''
    best_model_path_folds = {6: model_path + prefix + "seed6_model_val0.8510_train0.9782_epoch27",
                             7: model_path + prefix + "seed7_model_val0.8288_train0.9822_epoch27",
                             8: model_path + prefix + "seed8_model_val0.8291_train0.9959_epoch27",
                             9: model_path + prefix + "seed9_model_val0.8534_train0.9962_epoch27",
                             10: model_path + prefix + "seed10_model_val0.8353_train0.9954_epoch27"
                             }
    if machine == 'solar':
        hm_root = '/project/labname-lab/authorid/shortcut/log/MRNet/gethm/MRNet/'
    elif machine == 'ts':
        hm_root = '/local-scratch/authorid/log/MRNet/'
        hm_root = '/local-scratch/authorid/log/MRNet/' #todo
    elif machine == 'cc':
        hm_root = '/scratch/authorid/results_brats_rerun/MRNet/heatmaps'
    hm_saved_dir = {6:  hm_root + '1202_121131_fold_6/MRNet_get_hm_fold_{}_fold_{}'.format(6, 6),
                    7:  hm_root + '1202_115635_fold_7/MRNet_get_hm_fold_{}_fold_{}'.format(7, 7),
                    8:  hm_root + '1202_121142_fold_8/MRNet_get_hm_fold_{}_fold_{}'.format(8, 8),
                    9:  hm_root + '1202_121151_fold_9/MRNet_get_hm_fold_{}_fold_{}'.format(9, 9),
                    10: hm_root + '1202_121309_fold_10/MRNet_get_hm_fold_{}_fold_{}'.format(10, 10),
                    }
    modification = {'best_model': best_model_path_folds[fold], 'data_loader;args;fold': fold}
    config = ConfigParser.from_args(parser, updates = modification)
    dateTimeObj = datetime.now().strftime("%Y%m%d_%H%M")
    device, device_ids = prepare_device(config['n_gpu'])
    if job in ['acc_drop', 'acc_drop_bg', 'acc_drop_nb', 'test', 'val', 'mi', 'hm']:
        model = get_model(best_model_path_folds[fold], backbone= backbone, device = device)
    test_loader = load_data(task=task, use_gpu=True, test_loader = True, machine = machine)
    if machine == 'solar':
        segment_path ='/project/labname-lab/authorid/dld_data/MRNet-v1.0/bbox_test'
    elif machine == 'ts':
        segment_path = '/local-scratch/authorid/dld_data/Annotation_MRNet/anno_numpy'
    elif machine == 'cc':
        segment_path ='/project/labname-lab/authorid/dld_data/MRNet-v1.0/bbox_test'
    save_path = None
    modalities = config['xai']['modality']
    method_list = config['xai']['method_list']
    if job == 'test':
        csv_save_dir = config['trainer']['save_dir']
        save_path = Path(csv_save_dir) / 'test'
        save_path.mkdir(parents=True, exist_ok=True)
        get_test_result(model, test_loader, save_path=save_path, fold = fold, device = device)
    elif job == 'val':
        csv_save_dir = config['trainer']['save_dir']
        save_path = Path(csv_save_dir) / 'val'
        save_path.mkdir(parents=True, exist_ok=True)
        print(task)
        tr, val_loader = load_data(task=task, use_gpu=True, test_loader=False, machine = machine)
        get_test_result(model, val_loader, save_path=save_path, fold = fold, device = device)
    elif job == 'hm':
        get_hm(config, test_loader, model, fold=fold)
    elif job == 'msfi_readhm' or job == 'mi_readhm': # calculate msfi, iou, tumor portion
        csv_save_dir = config['trainer']['save_dir']
        hm_dir = hm_saved_dir[fold]
        if job == 'msfi_readhm':
            # record FP for each modality
            save_path = Path(csv_save_dir) / 'msfi'
            save_path.mkdir(parents=True, exist_ok=True)
            get_save_modality_hm_value_mrnet(hm_save_dir=hm_dir, result_save_dir=save_path, segment_path=segment_path,
                                             modalities=modalities, fold=fold, method_list=method_list)
        elif job == 'mi_readhm':
            # record sum for each modality heatmap
            save_path = Path(csv_save_dir) / 'mi_hm_sum'
            save_path.mkdir(parents=True, exist_ok=True)
            get_save_modality_hm_value_mrnet(hm_save_dir=hm_dir, result_save_dir=save_path, # segment_path=segment_path,
                                             modalities=modalities, fold=fold, method_list=method_list)
    elif job == 'msfi_shortcut':
        csv_save_dir = config['trainer']['save_dir']
        save_path = Path(csv_save_dir) / job
        save_path.mkdir(parents=True, exist_ok=True)
        get_save_modality_hm_value_mrnet(hm_save_dir = hm_saved_dir, result_save_dir = save_path, segment_path = segment_path, modalities = modalities, fold = 0, method_list=method_list)
    elif job == 'mi_corr' or job == 'mi_corr_aggr_fold_data' or job =='msfi_aggr_fold_data' or job =='shortcut_msfi_aggr_fold_data':
        corr_name = 'kendalltau'
        save_dir = '/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/'
        mi_dir = '/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/mi'
        misc_dir = '/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/msfi_shortcut'

        if job == 'mi_corr':
            hm_result_csv_root = '/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/mi_hm_sum'
            compute_xai_mod_shapley_corr(hm_result_csv_root=hm_result_csv_root, gt_csv_path=mi_dir,
                                         modalities=modalities, corr_name=corr_name)
        # elif job == 'mi_corr_aggr_fold_data' :
            # get individial data point mi corr measure
            saved_kendalltau = Path(save_dir)/'kendalltau'
            aggregate_msfi(target_folder="kendalltau_mrnet", col_name='kendalltau', root = saved_kendalltau, save_dir = Path(save_dir)/'kendalltau_mrnet', fold_range = [6,7,8,9,10])
            # aggregate_msfi(target_folder="kendalltau_shortcut", col_name='kendalltau', root = mi_dir, save_dir = Path(save_dir)/'kendalltau') # cannot calculate mi corr as no gt

            # hm_result_csv_root = '/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/msfi_shortcut' # cannot calculate mi for shortcut, as there is no gt MI for the five models
        elif job =='msfi_aggr_fold_data':
            # calculate MSFI for MRNet
            saved_msfi = Path(save_dir) / 'msfi_result_fold_wise'
            compute_mfsi(hm_result_csv_root=os.path.join(save_dir, 'msfi'), gt_csv_path=mi_dir, msfi_save_dir = saved_msfi,
                         modalities=modalities, normalization_method = 'scale')
            aggregate_msfi(target_folder="msfi_reporting_mrnet", col_name='msfi', root = saved_msfi, save_dir = Path(save_dir)/'msfi_reporting_mrnet', fold_range = [6,7,8,9,10])
        elif job =='shortcut_msfi_aggr_fold_data':
            # calculate MSFI for MRNet shortcut
            saved_msfi = Path(save_dir) / 'shortcut_msfi_result_fold_wise_mrnet'
            csv_filename = '/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/mi/mi_aggr_shortcut/aggregated_performance_fold_0.csv'
            # get_shapley(csv_filename, modalities=modalities)
            mi_aggr_shortcut = '/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/mi/mi_aggr_shortcut'
            compute_mfsi(hm_result_csv_root=os.path.join(save_dir, 'msfi_shortcut'), gt_csv_path=mi_aggr_shortcut, msfi_save_dir = saved_msfi,
                         modalities=modalities, normalization_method = 'scale')
            aggregate_msfi(target_folder="shortcut_msfi_result_fold_wise_mrnet", col_name='msfi', root = saved_msfi, save_dir = Path(save_dir)/'shortcut_msfi_reporting_mrnet', fold_range = [0])

        # calculate MI corr with heatmaps
        # compute_xai_mod_shapley_corr(hm_result_csv_root=os.path.join(save_dir, "modality_pos_values"),
        #                              gt_csv_path=target_dir,
        #                              modalities=modalities, corr_name=corr_name)
        # aggregate_msfi(target_folder="kendalltau", col_name='kendalltau')
    elif job == 'mi' or job == 'mi_reporting':
        csv_save_dir = config['trainer']['save_dir']
        save_path = Path(csv_save_dir) / 'mi'/'seed_{}'.format(fold)
        save_path.mkdir(parents=True, exist_ok=True)
        if job == 'mi':
            modality_shapley_mrnet(config, model= model,task = task, save_path=save_path, fold = fold, device = device, machine = machine)
        elif job == 'mi_reporting':
            # modality priotiritization MI
            csv_filename = shapley_result_csv(fold = fold, root = save_path, modalities= modalities, metric = 'auc')
            get_shapley(csv_filename, modalities=modalities)
    elif job in ['acc_drop', 'acc_drop_bg', 'acc_drop_nb']:
        if machine == 'solar':
            csv_dir = '/project/labname-lab/authorid/shortcut/log/MRNet/test'
        elif machine == 'ts':
            csv_dir = '/local-scratch/authorid/log/MRNet/test'
        test_record_csv = {6: '20211206_1100-cv_result_fold_6.csv',
                           7: '20211206_1113-cv_result_fold_7.csv',
                           8: '20211206_1114-cv_result_fold_8.csv',
                           9: '20211206_1114-cv_result_fold_9.csv',
                           10: '20211206_1114-cv_result_fold_10.csv'
        }
        save_dir = config['trainer']['save_dir']
        save_path = Path(save_dir)/'{}'.format(job)/'seed_{}'.format(fold)
        save_path.mkdir(parents=True, exist_ok=True)
        hm_dir = hm_saved_dir[fold]
        ad = Acc_Drop(config=config, load_model = False, load_data = False, baseline_value_type = job)
        ad.acc_drop_mrnet(save_dir=hm_dir, test_record_csv = Path(csv_dir) / test_record_csv[fold], model = model, valid_data_loader = test_loader)
        logging.info("{} Done!".format(job))
    elif job in ['acc_drop_reporting', 'acc_drop_nb_reporting','acc_drop_bg_reporting']:
        hm_dir = hm_saved_dir[fold]
        ad = Acc_Drop(config=config, load_model=False, load_data=False)
        modified_input_save_root = Path(hm_dir)/'{}'.format(job.replace('_reporting', ''))
        aggregate_df, graph = ad.plot_acc_drop(modified_input_save_root)
    elif job == 'fp':
        hm_dir = hm_saved_dir[fold]
        # calculate feature portion fp
        compare_with_gt_dataset_mrnet(save_dir=hm_dir, seg_path=segment_path,
                                      method_list = method_list, modalities= modalities)
    else:
        logging.info("No job specified!")
    if not save_path:
        save_path = config['trainer']['save_dir']
    with open(Path(save_path) / 'args_seed{}_{}.json'.format(fold, dateTimeObj), 'w') as out:
        json.dump(vars(args), out, indent=4)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--machine', default='solar', type=str, required=True)
    parser.add_argument('-d', '--device', default='all', type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('--task', type=str, default="meniscus")
    parser.add_argument('--job', type=str, default="hm")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--backbone', default="alexnet", type=str)
    parser.add_argument('-c', '--config', default="sh/mrnet_gethm/mrnet_xai_config.json", type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to saved hms')
    return parser

if __name__ == "__main__":
    print(torch.cuda.device_count())
    get_hm_pipeline()

