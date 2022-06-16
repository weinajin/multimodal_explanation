"""
Genereate ablated modality images. One time use code.
Modality ablation experiment. Generate and save the ablated brats images
Generate dataset with
Save in the directory: Path(brats_path).parent / "ablated_brats", and can be loaded with the script:
        T1    = os.path.join(image_path_list[0], bratsID, bratsID+'_t1.nii.gz') # (240, 240, 155)
        T1c   = os.path.join(image_path_list[1], bratsID, bratsID+'_t1ce.nii.gz')
        T2    = os.path.join(image_path_list[2], bratsID, bratsID+'_t2.nii.gz')
        FLAIR = os.path.join(image_path_list[3], bratsID, bratsID+'_flair.nii.gz')

For the original brats image, ablate it by filling in the non-zero value with random values ~ N(image_mean, image_std)
"""
import nibabel
import os
from pathlib import Path
import numpy as np
import pandas as pd
import itertools, math
from monai.data import write_nifti
from monai.transforms import LoadNifti
import monai
# from .heatmap_utils import get_heatmaps
from .heatmap_utlis import *
from scipy import stats
from scipy.stats import spearmanr as spr
from scipy.stats import kendalltau

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix
import csv
import itertools, math
import copy
from validate import test
from datetime import datetime
# from skimage.morphology import binary_dilation
# print(monai.__version__)

from sklearn.metrics import auc, roc_curve


def generate_ablated_dataset(modalities = ["t1", "t1ce", "t2", "flair"], ablation_mode = 'allzero'):
    """
    One time function to get and save the ablated modalities.
    :param allzero: replace the modality with all zeros
    :return:
    """
    data_root = "/local-scratch/authorid/dld_data/brats2020/MICCAI_BraTS2020_TrainingData/all_tmp"
    data_root = Path(data_root)
    if ablation_mode == 'allzero':  # ablate the whole modality, and replace with 0s
        saved_path = data_root.parent / "zero_ablated_brats"
    elif ablation_mode == 'allnoise': # ablate the whole modality, and replace with nontumor signal noises
        saved_path = data_root.parent / "ablated_brats"
    elif ablation_mode == 'lesionzero': # ablate the lesion only on the modality, and replace with 0s
        saved_path = data_root.parent / "lesionzero"
    # read brain MRI
    ids = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    for id in ids:
        seg_path = data_root / id / "{}_seg.nii.gz".format(id)
        seg = nibabel.load(seg_path).get_fdata()
        for m in modalities:
            path = data_root/id / "{}_{}.nii.gz".format(id, m)
            # mri = nibabel.load(path)
            # img_data = mri.get_fdata()
            loader = LoadNifti(image_only = False)
            img_data, header = loader(path)
            if ablation_mode == "allzero":
                ablate_array = np.zeros(img_data.shape)
            elif ablation_mode == 'allnoise':
                ablate_array = ablate_signal(img_data, seg)
            elif ablation_mode == 'lesionzero':
                ablate_array = ablate_tumor_only(img_data, seg)
            # nibabel.save(ablate_array, "{}_{}.nii.gz".format(id, m))
            output_root = saved_path/id
            output_root.mkdir(exist_ok=True, parents=True)
            print(header['affine'], header['original_affine'])
            write_nifti(ablate_array,
                        affine= header['affine'],
                        target_affine = header['original_affine'],
                        file_name = output_root/"{}_{}.nii.gz".format(id, m))
            # saver = NiftiSaver(data_root_dir =  output_root, output_postfix = None, output_ext='.nii.gz')
            # saver.save(ablate_array, {'filename_or_obj': "{}_{}".format(id, m)})

def ablate_tumor_only(array, seg):
    edge = 10
    dilated_seg = []
    for s in range(array.shape[-1]):
        dilated= binary_dilation(seg[:,:,s], selem = np.ones([edge for i in range(seg[:,:,s].ndim)]))
        dilated_seg.append(dilated)
    dilated_seg = np.stack(dilated_seg, axis=-1)
    ablated_array = np.copy(array)
    ablated_array[dilated_seg > 0] = 0
    return ablated_array

def ablate_signal(array, seg):
    """Helper: given a image array, replace the non-zero value by sampling from the rest non-tumor regions (with replacement, so
    that to keep the same distribution)
    """
    non_tumor = array[(array != 0) & (seg != 1) & (seg != 2) & (seg != 4)].flatten() # get brain region with non-tumor part [0. 1. 2. 4.]
    print(np.unique(seg))
    print(non_tumor.shape)
    # mean = np.mean(array)
    # std = np.std(array)
    ablated_array = np.random.choice(non_tumor, size=array.shape, replace=True)
    ablated_array[array == 0] = 0
    print('ablated_array', ablated_array.shape)

    return ablated_array


### Utlities to get gt modality shapley value, and compare hm value with this gt ###
def modality_shapley(config, ablated_image_folder, csv_save_dir = "/local-scratch/authorid/BRATS_IDH/log/mod_shapley"):
    """
    Modality ablation experiment. Generate and save the ablated brats images
    Generate dataset with
    Save in the directory: Path(brats_path).parent / "ablated_brats", and can be loaded with the script:
            T1    = os.path.join(image_path_list[0], bratsID, bratsID+'_t1.nii.gz') # (240, 240, 155)
            T1c   = os.path.join(image_path_list[1], bratsID, bratsID+'_t1ce.nii.gz')
            T2    = os.path.join(image_path_list[2], bratsID, bratsID+'_t2.nii.gz')
            FLAIR = os.path.join(image_path_list[3], bratsID, bratsID+'_flair.nii.gz')

    For the original brats image, ablate it by filling in the non-zero value with random values ~ N(image_mean, image_std)
    """
    modalities = config['xai']['modality']
    print(modalities)
    # generate modality combinations
    N_sets = list(itertools.product([0, 1], repeat=len(modalities)) ) # set of all_combinations
    for modality_selection in N_sets:
        test(config, timestamp = False, ablated_image_folder = ablated_image_folder, csv_save_dir = csv_save_dir, modality_selection= modality_selection)

def shapley_result_csv(fold = 1, root = '/local-scratch/authorid/BRATS_IDH/log/mod_shapley/test/', modalities= ["t1", "t1ce", "t2", "flair"], metric = 'acc'):
    """
    From the individual test records, get the summarized csv of modality: accuracy pair.
    :param fold:
    :param path:
    :return:
    """
    # get all csvs in the folder
    save_path = Path(root)/"shapley"
    save_path.mkdir(parents = True, exist_ok= True)
    csv_filename = save_path / 'aggregated_performance_fold_{}.csv'.format(fold)
    file_exists = os.path.isfile(csv_filename)
    fnames = modalities+["accuracy"]

    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
        # if not file_exists:
        csv_writer.writeheader()
        for f in Path(root).rglob('*cv_result_fold*.csv'):
            fn = f.name.split(".")[0].split("-")
            if len(fn) == len(modalities)+1:
                fold_num = fn[0].split("_")[-1]
            else:
                fold_num = fn[1].split("_")[-1]
            fold_num = int(fold_num)
            if fold_num == fold:
                if len(fn) == len(modalities) + 1:
                    modelity_selection = [int(i) for i in fn[1:]]
                else:
                    modelity_selection = [int(i) for i in fn[2:]]
                # print( fold_num, modelity_selection)
                results = pd.read_csv(f)
                gt = results['gt']
                pred = results['pred']
                if metric == 'auc':
                    fpr, tpr, threshold = roc_curve(results['gt'].to_list(), results['pred'].to_list())
                    accuracy = auc(fpr, tpr)
                else:
                    accuracy = accuracy_score(gt, pred)
                csv_record = {'accuracy': accuracy}
                for i, m in enumerate(modalities):
                    csv_record[m]= modelity_selection[i]

                csv_writer.writerow(csv_record)
                print("Fold {}: modality: {}, accuracy: {}".format(fold, modelity_selection, accuracy))
    print("Saved at {}".format(csv_filename))
    return csv_filename



def get_shapley(csv_filename, modalities = ["t1", "t1ce", "t2", "flair"]):
    """
    calculate modality shapeley value
    CSV with column: t1, t1c, t2, flair, of 0 / 1. and perforamnce value.
    :param csv:
    :return:
    """
    # convert csv to dict: {(0, 0, 1, 0): 10} {tuple: performance}
    df = pd.read_csv(csv_filename)
    fold = Path(csv_filename).name.split('.')[0].split('_')[-1]
    # print(fold)
    df_dict = df.to_dict(orient='records')
    # print(df_dict)
    v_dict = {} #
    for row in df_dict:
        mod_lst = []
        for m in modalities:
            mod_lst.append(row[m])
        v_dict[tuple(mod_lst)] = row['accuracy']
    # print(v_dict)

    n = len(modalities)
    # sanity check if all mod combinations are exists
    N_sets = list(itertools.product([0,1],repeat = len(modalities))) # set of all_combinations
    for s in N_sets:
        if tuple(s) not in v_dict:
            print("ERROR in get_shapley! {} missing".format(s))
    N_sets_array = np.array(N_sets)  # array([[0, 0, 0, 0], [0, 0, 0, 1],
    mod_shapley = {}
    # for each mod, calculate its shapley value:
    for i, mod in enumerate(modalities):
        # get combination not including mod
        n_not_i =  N_sets_array[N_sets_array[:, i]==0]# # a list containing all subsets that don't contains i todo
        # print(n_not_i, i)

        phi_i= 0
        for s in n_not_i:
            # print('s', s)
            v_s = v_dict[tuple(s)]
            sANDi = copy.deepcopy(s)
            sANDi[i] =1
            v_sANDi = v_dict[tuple(sANDi)]
            # print(s , s.sum(), i, mod)
            phi_i += (v_sANDi -  v_s) * math.factorial(s.sum()) * (math.factorial(n - s.sum() - 1)) / math.factorial(n)
        mod_shapley[mod] = phi_i
    mod_shapley['fold'] = fold
    print(mod_shapley)
    # save gt shapley to csv
    with open(Path(csv_filename).parent/'fold_{}_modality_shapley.csv'.format(fold), 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(mod_shapley.keys()))
        csv_writer.writeheader()
        csv_writer.writerow(mod_shapley)
        # for key in mod_shapley.keys():

            # f.write("%s,%s\n" % (key, mod_shapley[key]))
    return mod_shapley


def get_shapley_gt_multiple_runs_pipeline(config, run_num, ablated_image_folder, csv_save_dir):
    """Since the shapley value gt is not deterministic, run multiple run_num to get the distribution of gt modality shapley value."""
    modalities = config['xai']['modality']
    fold = config['data_loader']['args']['fold']
    # support multiple runtime, check if file exists
    existing_runs = [f for f in os.listdir(csv_save_dir) if os.path.isdir(os.path.join(csv_save_dir, f))]
    existing_runs.sort()
    starting_run = -1
    for i in existing_runs:
        i = int(i)
        shapley_csv = os.path.join(csv_save_dir, "{}".format(i), 'shapley', 'fold_{}_modality_shapley.csv'.format(fold))
        file_exists = os.path.isfile(shapley_csv)
        if file_exists:
            starting_run = i
        else:
            break
    if starting_run >= run_num:
        return
    for run_i in range(starting_run+1, run_num):
        run_dir = os.path.join(csv_save_dir, "{}".format(run_i))
        modality_shapley(config, ablated_image_folder = ablated_image_folder, csv_save_dir= run_dir)
        csv_filename = shapley_result_csv(fold = fold, modalities=modalities, root = run_dir)
        print(csv_filename)
        get_shapley(csv_filename, modalities=modalities)

def aggregate_shapley_gt_mean_std(fold, csv_save_dir, modalities):
    # calculate the mean and std of the multiple run shapley
    result_list = []
    runs = [f for f in os.listdir(csv_save_dir) if os.path.isdir(os.path.join(csv_save_dir, f))]
    for run_i in runs:
        shapley_csv = os.path.join(csv_save_dir, "{}".format(run_i), 'shapley', 'fold_{}_modality_shapley.csv'.format(fold))
        file_exists = os.path.isfile(shapley_csv)
        if file_exists:
            df = pd.read_csv(shapley_csv)
            df = df.iloc[0]#.to_dict('list')
            # print(df)
            gt_shapley = [df[m] for m in modalities]
            result_list.append(gt_shapley)
    result_array = np.array(result_list)
    shapley_mean = result_array.mean(axis = 0)
    shapley_std = result_array.std(axis = 0)
    print(result_array)
    print("Shapley mean: {}, std {}".format(shapley_mean, shapley_std))
    # save the mean and std as two csv files
    mean_shapley, std_shapley = {}, {}
    mean_shapley['fold'], std_shapley['fold']  = fold, fold
    # for each mod, calculate its shapley value:
    for i, mod in enumerate(modalities):
        mean_shapley[mod] = shapley_mean[i]
        std_shapley[mod] = shapley_std[i]
    with open(os.path.join(csv_save_dir, 'multirun_gt_shapley_fold_{}.csv'.format(fold)), 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(mean_shapley.keys()))
        csv_writer.writeheader()
        csv_writer.writerow(mean_shapley)
    with open(os.path.join(csv_save_dir, 'multirun_gt_shapleySTD_fold_{}.csv'.format(fold)), 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(std_shapley.keys()))
        csv_writer.writeheader()
        csv_writer.writerow(std_shapley)

def get_modality_feature_hm_value(post_hm, seg, penalize = True, portion = True):
    """
    Get positive+negative values inside tumor regions, minus positive values outside tumor regions
    (penalty for positive values outside tumor)
    :param post_hm: np array of the same shape with seg
    :param penalize: old parameter. No longer needed with new parameter portion
    :return:
    """
    assert seg.shape == post_hm.shape[1:], "segmentation map shape {} and processed hm shape {} does not match!".format(seg.shape, post_hm.shape[1:])
    # binary_seg = seg[seg>0]
    edge = 20
    dilated_seg = []
    for s in range(seg.shape[-1]):
        dilated= binary_dilation(seg[:,:,s], selem = np.ones([edge for i in range(seg[:,:,s].ndim)]))
        dilated_seg.append(dilated)
    dilated_seg = np.stack(dilated_seg, axis = -1)
    print((seg>0).sum()/seg.size, (dilated_seg>0).sum()/dilated_seg.size, dilated_seg.shape)
    hm_values = []
    for hm in post_hm:
        feature = hm[(dilated_seg>0) & (hm>0)]
        non_feature = hm[(dilated_seg==0) & (hm>0)]
        if portion:
            v = feature.sum() / ( feature.sum() + non_feature.sum() )
            if (v < 0):
                print( feature.sum() , feature.shape, non_feature.shape,non_feature.sum())
        else:
            v = feature.sum()
            if penalize:
                v -= non_feature.sum()
        hm_values.append(v)
    print(hm_values, np.sum(post_hm, axis = tuple([i for i in range(4)][1:])), '\n')
    return hm_values



def get_save_modality_hm_value(hm_save_dir, result_save_dir, fold, method_list, penalize= False, portion_metrics= True, positiveHMonly = True,  segment_path = None, modalities= ["t1", "t1ce", "t2", "flair"]):
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
    Path(result_save_dir).mkdir(parents=True, exist_ok=True)
    columns = modalities+ ['XAI', 'dataID']
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
        hm_dict, data_record = get_heatmaps(hm_save_dir, method, by_data=False, hm_as_array=False, return_mri=False)
        print("Number of data for {}: {}".format(method, len(hm_dict.keys())))  # , hm_dict.keys())
        for dataID, hm in hm_dict.items():
            print(hm.min(), hm.max(), dataID)
            post_hm = postprocess_heatmaps(hm, no_neg=positiveHMonly) # (C, H,W,D) # the postprocessed hm is already non-negative
            if  segment_path:
                seg_path = os.path.join(segment_path, dataID, dataID + '_seg.nii.gz')
                seg = nibabel.load(seg_path).get_fdata()
                seg = np.rot90(seg, k=3, axes=(0, 1)) # important, avoid bug of seg, saliency map mismatch
                hm_values = get_modality_feature_hm_value(post_hm, seg, penalize=penalize, portion = portion_metrics)
            else:
                if positiveHMonly:
                    positive_hm = np.copy(post_hm)
                    positive_hm[positive_hm <0] =0
                    hm_values = np.sum(positive_hm, axis = tuple([i for i in range(len(modalities))][1:]))
                else:
                    hm_values = np.sum(post_hm, axis = tuple([i for i in range(len(modalities))][1:]))
            # print(method, dataID, corr, p_value)
            value["XAI"] = method
            value['dataID'] = dataID
            for i, mod in enumerate(modalities):
                value[mod] = hm_values[i]
            result_series= pd.Series(value, index=columns)
            result_df= result_df.append(result_series, ignore_index=True)
            # print(result_df)
    # result_df = pd.DataFrame.from_dict(result, orient = 'index')
        result_df.to_csv(result_csv)
        print("modalityHM Saved at: {}".format(result_csv))
    return result_csv



# def corr_modality_shapley(hm_save_dir, method_list, shapley_csv, modalities= ["t1", "t1ce", "t2", "flair"]):
#     ''''''
#     fold = Path(shapley_csv).name.split('.')[0].split('_')[1] #fold_{}_modality_shapley.csv'
#     df = pd.read_csv(shapley_csv)
#     # print(df)
#     df = df.iloc[0]#.to_dict('list')
#     # print(df)
#     gt_shapley = [df[m] for m in modalities]
#     # print(gt_shapley)
#     columns = modalities+ ['XAI', 'correlation', 'p_value', 'dataID']
#
#     for method in method_list:
#         result_csv = Path(shapley_csv).parent / 'CorrModalityShapley_fold-{}-{}.csv'.format(fold, method)
#         file_exists = os.path.isfile(result_csv)
#         if file_exists:
#             print("{} exists, pass".format(file_exists))
#             continue
#         result_df = pd.DataFrame(columns=columns)
#         correlations = {}
#         gt_results = list()
#         # post-process hms
#         hm_dict, data_record = get_heatmaps(hm_save_dir, method, by_data=False, hm_as_array=False, return_mri=False)
#         print("Number of data to be evaluated for {}: {}".format(method, len(hm_dict.keys())))  # , hm_dict.keys())
#         for dataID, hm in hm_dict.items():
#             post_hm = postprocess_heatmaps(hm) # (C, H,W,D)
#             hm_values = np.sum(post_hm, axis = tuple([i for i in range(len(modalities))][1:]))
#             corr, p_value = spr(gt_shapley, hm_values)
#             # print(method, dataID, corr, p_value)
#             correlations["XAI"] = method
#             correlations["correlation"] = corr
#             correlations["p_value"] = p_value
#             correlations['dataID'] = dataID
#             for i, mod in enumerate(modalities):
#                 correlations[mod] = hm_values[i]
#             result_series= pd.Series(correlations, index=columns)
#             result_df= result_df.append(result_series, ignore_index=True)
#             print(result_df)
#     # result_df = pd.DataFrame.from_dict(result, orient = 'index')
#         result_df.to_csv(result_csv)
#         print("corr_modality_shapley Saved at: {}".format(result_csv))
#     return result_csv

def compute_xai_mod_shapley_corr(hm_result_csv_root, gt_csv_path, modalities= ["t1", "t1ce", "t2", "flair"], corr_name = "pearson"):
    fold_dict = {}
    if corr_name == 'pearson':
        corr_method = stats.pearsonr
    elif corr_name == 'spr':
        corr_method = spr
    elif corr_name == 'kendalltau':
        corr_method = kendalltau
    # get all hm value csv files for each
    for f in Path(hm_result_csv_root).rglob('modalityHM_fold*.csv'):
        fold = f.name.split('.')[0].split("-")[1]
        if fold in fold_dict:
            fold_dict[fold].append(f)
        else:
            fold_dict[fold] = [f]
    columns = ['XAI', 'fold', 'corr','p_value', 'data_wise_corr', 'data_wise_std' ] + modalities
    result_df = pd.DataFrame(columns=columns)

    for fold, files in fold_dict.items():
        # get mod shapley gt
        shapley_csv = os.path.join(gt_csv_path, 'multirun_gt_shapley_fold_{}.csv'.format(fold))
        if not os.path.isfile(shapley_csv):
            if gt_csv_path[-2:] =='mi':
                shapley_csv = Path(gt_csv_path) / 'seed_{}'.format(fold) /'shapley' / 'fold_{}_modality_shapley.csv'.format(fold)
            else:
                shapley_csv = Path(gt_csv_path) / 'shapley' / 'fold_{}_modality_shapley.csv'.format(fold)
        # elif not os.path.isfile(shapley_csv):
        #     shapley_csv = Path(gt_csv_path) / 'shapley' / 'fold_{}_modality_shapley.csv'.format(fold)
        # print("shapley_csv",    shapley_csv)
        df = pd.read_csv(shapley_csv)
        df = df.iloc[0]#.to_dict('list')
        # print(df)
        gt_shapley = [df[m] for m in modalities]
        print(fold, gt_shapley)
        for fl in files:
            method = fl.name.split('.')[0].split('-')[-1]
            hm_df = pd.read_csv(fl)
            # print("file", fl, hm_df )
            result = {}
            # print( hm_df.mean(axis=0))
            result['XAI'] = hm_df['XAI'].unique()[0]
            result['fold'] = fold
            hm_mean = hm_df.mean(axis=0)
            hm_value_dataset = [hm_mean[m] for m in modalities]
            hm_df['XAI'] = method
            hm_df['Fold'] = fold
            for i,m in enumerate(modalities):
                result[m] = hm_mean[m]
            # print(hm_value_dataset)
            result["corr"], result["p_value"] = corr_method(gt_shapley, hm_value_dataset)
            hm_df['kendalltau'] = hm_df.apply(lambda row: corr_method(gt_shapley, row[modalities]).correlation, axis=1)
            hm_df['pvalue'] = hm_df.apply(lambda row: corr_method(gt_shapley, row[modalities]).pvalue, axis=1)
            correlation = list(hm_df['kendalltau'])

            # hm_df['kendalltau'] = 0
            # hm_df['pvalue'] = 0
            # for index, row in hm_df.iterrows():
            #     corr, p_value = corr_method(gt_shapley, row[modalities])
            #     correlation.append(corr)
            #     hm_df['kendalltau'] = corr
            #     hm_df['pvalue'] = p_value
            kandall_dir = fl.parent.parent/ 'kendalltau'
            kandall_dir.mkdir(parents=True, exist_ok=True)
            hm_df.to_csv(os.path.join(kandall_dir, fl.name))
            data_wise_corr = np.array(correlation)
            result["data_wise_corr"] = data_wise_corr.mean()
            result["data_wise_std"] = data_wise_corr.std()
            # print("data wise corr: mean {}, std {}".format(data_wise_corr.mean(), data_wise_corr.std()))
            # print(result)
            result_series= pd.Series(result, index=columns)
            result_df= result_df.append(result_series, ignore_index=True)
    dt = datetime.now().strftime(r'%m%d_%H%M%S')
    sorted_df = result_df.sort_values(by='corr', ascending=False, na_position='last')
    print(sorted_df)
    hm_type = Path(hm_result_csv_root).name
    sorted_df.to_csv(os.path.join(gt_csv_path,  "mod_shapley_result-{}-{}-{}.csv".format(corr_name, hm_type, dt)))

def compute_mfsi(hm_result_csv_root, gt_csv_path, modalities= ["t1", "t1ce", "t2", "flair"], msfi_save_dir = 'msfi_featshapley', normalization_method = 'minmax'):#, corr_name = "pearson"):
    fold_dict = {}
    # if corr_name == 'pearson':
    #     corr_method = stats.pearsonr
    # else:
    #     corr_method = spr
    # get all hm value csv files for each
    for f in Path(hm_result_csv_root).rglob('modalityHM_fold*.csv'):
        fold = f.name.split('.')[0].split("-")[1]
        if fold in fold_dict:
            fold_dict[fold].append(f)
        else:
            fold_dict[fold] = [f]
    columns = ['XAI', 'fold', 'msfi', 'msfi_std' ]
    result_df = pd.DataFrame(columns=columns)

    for fold, files in fold_dict.items():
        # get mod shapley gt
        shapley_csv = os.path.join(gt_csv_path, 'multirun_gt_shapley_fold_{}.csv'.format(fold))
        if not os.path.isfile(shapley_csv):
            if gt_csv_path[-2:] =='mi':
                shapley_csv = Path(gt_csv_path) / 'seed_{}'.format(fold) /'shapley' / 'fold_{}_modality_shapley.csv'.format(fold)
            elif gt_csv_path[-8:] =='shortcut':
                print(gt_csv_path)
                shapley_csv = Path(gt_csv_path) / 'fold_{}_modality_shapley.csv'.format(fold)
            else:
                shapley_csv = Path(gt_csv_path) / 'shapley' / 'fold_{}_modality_shapley.csv'.format(fold)
        # print("shapley_csv",    shapley_csv)
        df = pd.read_csv(shapley_csv)
        df = df.iloc[0]#.to_dict('list')
        # print(df)
        gt_shapley = [df[m] for m in modalities]
        # normalize the gt_shapley value
        sh_min = min(gt_shapley)
        print('befor norm', df[modalities])
        if normalization_method == 'minmax':
            for m in modalities:
                df[m] = (df[m] - min(gt_shapley) ) / (max(gt_shapley) - min(gt_shapley))
        else:
            ratio = 1 / max(gt_shapley)
            for m in modalities:
                df[m]  = df[m] * ratio
        print('after mi norm')
        print(df, df[modalities], df[modalities].sum() )
        for fl in files:
            hm_df = pd.read_csv(fl)
            # print("file", fl, hm_df )
            result = {}
            result['XAI'] = hm_df['XAI'].unique()[0]
            result['fold'] = fold
            hm_df['msfi'] = 0
            for i,m in enumerate(modalities):
                hm_df[m] = hm_df[m] * df[m] # hm value weighted by gt_shapley
                hm_df['msfi'] += hm_df[m]
            hm_df['msfi'] /= df[modalities].sum()
            result["msfi"] = hm_df['msfi'].mean()
            result["msfi_std"] = hm_df['msfi'].std()
            print("{}  mean {}, std {}".format(result['XAI'], result["msfi"], result["msfi_std"] ))
            # print(result)
            result_series= pd.Series(result, index=columns)
            result_df= result_df.append(result_series, ignore_index=True)
            msfi_dir = (fl).parent.parent/ '{}'.format(msfi_save_dir)
            msfi_dir.mkdir(exist_ok=True, parents=True)
            hm_df.to_csv(msfi_dir/fl.name)
    dt = datetime.now().strftime(r'%m%d_%H%M%S')
    sorted_df = result_df.sort_values(by='msfi', ascending=False, na_position='last')
    print(sorted_df)
    hm_type = Path(hm_result_csv_root).name
    sorted_df.to_csv(os.path.join(gt_csv_path,  "mod_shapley_msfi_result-{}-{}.csv".format(hm_type, dt)))
    return msfi_save_dir


def aggregate_msfi(target_folder = "msfi", col_name = 'msfi', root = None, save_dir = None, fold_range = [1,2,3,4,5] ):
    """

    :param target_folder:
    :param col_name: msfi or kendalltau
    :return:
    """
    method_fl = dict()
    for fold in fold_range:
        if not root:
            root = '/local-scratch/authorid/trained_model/BRATS20_HGG/heatmaps/fold_{}/get_hm_fold_{}/{}/'.format(fold, fold, target_folder)
        print(root)
        for fl in Path(root).rglob('modalityHM_fold-{}-*.csv'.format(fold)):
            method = fl.name.split('.')[0].split('-')[-1]
            print(method)
            if method not in method_fl:
                method_fl[method] = [fl]
            else:
                method_fl[method].append(fl)
    all_method = dict()
    print(method_fl, target_folder)
    for method in method_fl:
        dfs = []
        for csv in method_fl[method]:
            print(csv)
            fold = csv.name.split('.')[0].split('-')[1]
            df = pd.read_csv(csv)
            df['Fold'] = fold
            df['XAI'] = method
            dfs.append(df)
        all_5_folds = pd.concat(dfs, ignore_index=True)
        # save the all_5_folds data
        if not save_dir:
            save_dir = Path('/local-scratch/authorid/trained_model/BRATS20_HGG/heatmaps/{}'.format(target_folder))
        save_dir.mkdir(exist_ok=True, parents=True)
        all_5_folds.to_csv(os.path.join(save_dir, '{}.csv'.format(method)))
        all_method[method] = (all_5_folds[col_name].mean(), all_5_folds)
    # print(all_method)
    sorted_method_mean = {k: v for k, v in sorted(all_method.items(), reverse=True, key=lambda item: item[1][0])}
    # print(sorted_method_mean)
    df_list = [v[1] for k, v in sorted_method_mean.items()]
    for m in sorted_method_mean:
        print(m, sorted_method_mean[m][0] )
    # print(sorted_method_mean.keys())
    all = pd.concat(df_list, ignore_index=True)
    all.to_csv(os.path.join(save_dir, 'all.csv'))



# if __name__ == '__main__':
#     main(ablation_mode = 'lesionzero')

