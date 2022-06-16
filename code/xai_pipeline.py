import argparse, sys
import collections
import torch
import numpy as np
import logging
from pathlib import Path

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.parse_config import ConfigParser
from utils.trainer import Trainer
from utils.util import prepare_device, read_json

from xai.heatmap_eval import BaseXAIevaluator, Acc_Drop
from xai.modality_ablation import *
from xai.tumorsyn_xai import *
from validate import test
# import itertools, math
import os
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, machine= 'ts', job = None):
    #=== default parameters ===#
    modalities = config['xai']['modality']
    method_list = config['xai']['method_list']
    fold = config['data_loader']['args']['fold']
    try:
        seed = config['seed']
    except:
        seed = None

    if machine == 'solar':
        root = '/project/labname-lab/'
        save_dir = Path(root+'authorid/BRATS_IDH/log/BRATS_HGG/fold_{}/get_hm_fold_{}'.format(fold, fold))

    elif machine == 'ts':
        root = '/local-scratch/'
        # path of saved hms
        save_dir = Path(root+'authorid/trained_model/BRATS20_HGG/heatmaps/fold_{}/get_hm_fold_{}'.format(fold, fold))
        if seed is not None:
            save_dir = Path(root+'authorid/brats_rerun_20220502/s{}_hm_fold_1/get_hm_seed_{}_fold_1'.format(seed, seed))
            save_dir = Path(root+'authorid/brats_rerun_20220502/s{}_hm_fold_1'.format(seed))

        # if 'seed' in config:
        #     save_dir = Path(root + ''.format(seed)) #todo

    elif machine == 'cc':
        root = config['trainer']['save_dir']#'/scratch/authorid/brats_rerun' # dir to save log
        save_dir = '/scratch/authorid/results_brats_rerun/heatmaps/'# heatmap saved dir
        try:
            seed = config['seed']
            save_dir = Path(save_dir)/ 'seed_{}'.format(seed)
        except:
            print('ERROR! no seed specified!')
            return
    logging.info("save_dir is:{}, for seed {}".format(save_dir, seed))
    print("save_dir is:{}, for seed {}".format(save_dir, seed))
    task = config['name']

    #=== call each function ===#

    # === general eval pipeline ===
    if task == 'BRATS_HGG':
        # run a single job
        if job != 'pipeline' and job != 'pipeline_nogethm' and job!= 'acc_drop_bgnb':
            vanilla_eval_pipeline(config, save_dir, job, root)
        elif job == 'pipeline' or job == 'pipeline_nogethm':
            # automatically run job pipeline for different evaluation jobs
            job_list = ['gethm', 'mi', 'mi_reporting', 'mi_readhm', 'msfi_readhm', 'fp_iou', 'acc_drop']
            if job == 'pipeline_nogethm':
                job_list.pop(0)
            for j in job_list:
                print('=====Running Job: {}======'.format(j))
                vanilla_eval_pipeline(config, save_dir, j, root)
        elif job == 'acc_drop_bgnb':
            # acc_drop experiment with two additional baseline value: bg: replace with background value, nb: with neighbor value
            for j in ['acc_drop_bg', 'acc_drop_nb']:
                vanilla_eval_pipeline(config, save_dir, j, root)
        else:
            print('ERROR! No job specified!')

    # === Generate Ablated Dataset for Modality Shapley ===#
    # generate_ablated_dataset(ablation_mode='lesionzero')

    # === Modality Shapley ===#
    # if task == 'BRATS_HGG':
        # compute_modality_shapley(config, save_dir, root)
        # eval2_1_msfi_brats(config, save_dir, root)
        # pass

    # === Tumor Synthesize Experiment ===#
    elif task == 'tumorsyn' or task == 'tumorsyn_shortcut':
        if job == 'test':
            # Step1 : get test performance, including MI with baseline/fist/second performance, save the input for step 2
            for i in range(5):
                tumorsyn_get_test_performance(config) # test to get gt modality importance
        # Step2: generate heatmaps
        elif job == 'gethm':
            tumorsyn_get_hm(config, machine)
        elif job == 'hmeval':
            tumorsyn_results()
        elif job == 'mi':
            compute_modality_shapley(config, save_dir, root)

def eval2_1_msfi_brats(config, save_dir, root):
    # obsolete method
    #=== default parameters ===#
    modalities = config['xai']['modality']
    method_list = config['xai']['method_list']
    data_path = config['data_loader']['args']['data_dir']
    fold = config['data_loader']['args']['fold']
    ablated_image_folder = "lesionzero"
    target_dir = root + "authorid/BRATS_IDH/log/zeroLesion_gt_shapley_multiple_run"
    get_save_modality_hm_value(hm_save_dir=save_dir, method_list=method_list, modalities=modalities, fold=fold,
                               positiveHMonly=True,
                               segment_path=os.path.join(data_path, 'all_tmp'),
                               result_save_dir=os.path.join(save_dir, "seg_rotated" ))
    compute_mfsi(hm_result_csv_root = os.path.join(save_dir, 'seg_rotated'), gt_csv_path = target_dir, modalities=modalities)

    # after all 5 folds
    aggregate_msfi(target_folder="msfi_featshapley", col_name='msfi')

    return

def compute_modality_shapley(config, save_dir, root):
    """ obsolete method, used for prototyping, too messy
    :param config:
    :param save_dir: path of saved hms
    :return:
    """

    #=== default parameters ===#
    modalities = config['xai']['modality']
    method_list = config['xai']['method_list']
    data_path = config['data_loader']['args']['data_dir']
    fold = config['data_loader']['args']['fold']
    # path of saved hms
    # save_dir = Path(root + 'authorid/trained_model/BRATS20_HGG/heatmaps/get_hm_fold_{}'.format(fold))
    # save_dir = Path(root + 'authorid/BRATS_IDH/log/BRATS_HGG/0517_122700_fold_1_Gradient_HM/get_hm_fold_{}'.format(fold))


    # === compare with gt modality shapley value ===#

    ablation_mode = ["zero_ablated_brats",  "ablated_brats", "lesionzero"]
    ablated_image_folder = ablation_mode[2]
    allzero_dir = root + "authorid/BRATS_IDH/log/zeroR_gt_shapley_multiple_run"
    result_save_dir =  root + "authorid/BRATS_IDH/log/nontumorR_gt_shapley_multiple_run"
    lesionzero = root + "authorid/BRATS_IDH/log/zeroLesion_gt_shapley_multiple_run"
    Path(allzero_dir).mkdir(parents = True, exist_ok=True)
    Path(result_save_dir).mkdir(parents = True, exist_ok=True)
    Path(lesionzero).mkdir(parents = True, exist_ok=True)
    if ablated_image_folder == "zero_ablated_brats":
        target_dir = allzero_dir
    elif ablated_image_folder == "ablated_brats":
        target_dir = result_save_dir
    elif ablated_image_folder == "lesionzero":
        target_dir = lesionzero

    # new 0607 Modality-specific feature importance, using MSFI metric
    # aggregate_msfi()
    # corr_name =  "pearson" # "spr"#
    # get_save_modality_hm_value(hm_save_dir = save_dir, method_list = method_list, modalities = modalities, fold = fold,
    #                            portion_metrics = True,
    #                            segment_path = os.path.join(data_path, 'all_tmp'),
    #                            result_save_dir = os.path.join(save_dir, 'portion') )

    # Eval 2.1 : MSFI, using lesionzero as gt, and compute msfi with that gt
    # compute_mfsi(hm_result_csv_root = os.path.join(save_dir, 'portion'), gt_csv_path = target_dir, modalities=modalities)
    # after running 5 folds
    aggregate_msfi(target_folder="msfi_featshapley", col_name = 'msfi')

    return

    # eval 1: modality importance, using kendalltau correlation
    corr_name = "kendalltau"  # "pearson" "spr" kendalltau
    get_save_modality_hm_value(hm_save_dir=save_dir, method_list=method_list, modalities=modalities, fold=fold,
                               positiveHMonly=True,
                               # segment_path=os.path.join(data_path, 'all_tmp'),
                               result_save_dir=os.path.join(save_dir, "modality_pos_values"))
    compute_xai_mod_shapley_corr(hm_result_csv_root=os.path.join(save_dir, "modality_pos_values"), gt_csv_path=target_dir,
                                 modalities=modalities, corr_name=corr_name)
    aggregate_msfi(target_folder="kendalltau", col_name  = 'kendalltau')
    return

    ### get shapley gt
    get_shapley_gt_multiple_runs_pipeline(config, run_num = 2, ablated_image_folder= ablated_image_folder, csv_save_dir= target_dir)
    aggregate_shapley_gt_mean_std(fold = fold, csv_save_dir = target_dir, modalities = modalities)

    ###  get hm importance value
    hm_value_lst = ['modality_and_feature_values_penalize', 'modality_and_feature_values', 'modality_values']
    for t in [2]:
        hm_type = hm_value_lst[t]
        penalize = False
        if hm_type == 'modality_and_feature_values_penalize':
            penalize = True
        print(hm_type)
        corr_name = "kendalltau" # "pearson" "spr" kendalltau
        get_save_modality_hm_value(hm_save_dir = save_dir, method_list = method_list, modalities = modalities, fold = fold,
                                   penalize = penalize,
                                   segment_path = os.path.join(data_path, 'all_tmp'),
                                   result_save_dir = os.path.join(save_dir, hm_type) )


        # - for single run
        # modality_shapley(config, ablated_image_folder = ablated_image_folder, csv_save_dir= target_dir)
        # csv_filename = shapley_result_csv(fold = fold, modalities=modalities, root = target_dir+'/test/')
        # print(csv_filename)
        # get_shapley(os.path.join(target_dir, "test", "shapley", "aggregated_performance_fold_{}.csv".format(fold)))

        ###  compute correlation
        compute_xai_mod_shapley_corr(hm_result_csv_root = os.path.join(save_dir, hm_type), gt_csv_path = target_dir, modalities=modalities, corr_name = corr_name)

    # compute_xai_mod_shapley_corr(os.path.join(target_dir, "test"), modalities=modalities)
    # corr_modality_shapley(hm_save_dir = save_dir, method_list = method_list, modalities = modalities,
    #                           shapley_csv = os.path.join(target_dir, "test", "shapley", "fold_{}_modality_shapley.csv".format(fold)))


def vanilla_eval_pipeline(config, save_dir, job, root):
    '''

    :param config:
    :param save_dir: path of saved hms
    :param job:
    :param root:
    :return:
    '''
    #=== default parameters ===#
    modalities = config['xai']['modality']
    method_list = config['xai']['method_list']
    data_path = config['data_loader']['args']['data_dir']
    segment_path = os.path.join(data_path, 'all')
    fold = config['data_loader']['args']['fold']
    csv_save_dir = config['trainer']['save_dir'] # root of the dir to save the evaluation results
    seed = None
    try:
        seed = config['seed']
    except:
        seed = None
    # === 0. generate hms ===#
    if job == 'gethm':
        create_heatmaps_pipeline(config)

    # === 1. compare segmentation maps ===#
    elif job == 'fp_iou': # todo add seed in save files
        compare_seg_map(config, save_dir)

    # === 2. Acc drop ===#
    elif job in ['acc_drop', 'acc_drop_bg', 'acc_drop_nb']: # todo add seed in save files
        print(job, 'job')
        acc_drop_exp(config, save_dir, job)
    # # save_dir = '../exp_log/xai_exp/0219_multiModalEval_BRATS_IDH_ts/0219_095505_fold_1/test_vanilla_heatmaps_BRATS_IDH_fold_1/acc_drop'
    # g = acc_drop_plot(config, modified_input_save_root)
    # path to save csv
    # === Eval 3: MI ===
    elif job == 'mi' or job == 'mi_reporting' or job == 'mi_readhm' or job == 'mi_corr':
        if seed is not None:
            csv_save_dir_foldwise = Path(csv_save_dir) / 'mi'/'seed_{}'.format(seed)
        else:
            csv_save_dir_foldwise = Path(csv_save_dir) / 'mi' / 'fold_{}'.format(fold)
        csv_save_dir_foldwise.mkdir(parents=True, exist_ok=True)

        ablated_image_folder = 'zero_ablated_brats'
        # allzero_dir = root + "authorid/BRATS_IDH/log/zeroR_gt_shapley_multiple_run"
        # Path(allzero_dir).mkdir(parents=True, exist_ok=True)
        # target_dir = allzero_dir

        if job == 'mi':
            # get ground truth modality importance MI value, using shapeley value

            modality_shapley(config, ablated_image_folder = ablated_image_folder, csv_save_dir= csv_save_dir_foldwise)

        elif job == 'mi_reporting':
            if seed is not None:
                fold_or_seed = seed
            else:
                fold_or_seed = fold
            # modality importance MI reporting for each model
            csv_filename = shapley_result_csv(fold = fold_or_seed, modalities=modalities, root = csv_save_dir_foldwise)
            print(csv_filename)
            get_shapley(os.path.join(csv_save_dir_foldwise, "shapley", "aggregated_performance_fold_{}.csv".format(fold_or_seed)))

        elif job == 'mi_readhm':
            if seed is not None:
                fold_or_seed = seed
            else:
                fold_or_seed = fold
            # record sum for each modality heatmap
            save_path = Path(csv_save_dir) / 'mi_hm_sum'
            save_path.mkdir(parents=True, exist_ok=True)

            if config['machine'] == 'cc': # fix get_heatmaps() bug
                save_dir = Path(save_dir) / 'get_hm_seed_{}_fold_1'.format(seed)

            get_save_modality_hm_value(hm_save_dir=save_dir, method_list=method_list, modalities=modalities, fold=fold_or_seed,
                                       positiveHMonly=True, portion_metrics = False,
                                       # segment_path=os.path.join(data_path, 'all_tmp'),
                                       result_save_dir=save_path)

        elif job == 'mi_corr': #todo debug
            # not in pipeline, as will need to aggregate all models
            corr_name = "kendalltau"  # "pearson" "spr" kendalltau

            hm_result_csv_root = Path(csv_save_dir) / 'mi_hm_sum' # path to save the mi_readhm job results on hm sum  #'/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/mi_hm_sum'
            compute_xai_mod_shapley_corr(hm_result_csv_root=hm_result_csv_root, gt_csv_path=Path(csv_save_dir)/'mi',
                                         modalities=modalities, corr_name=corr_name)
        # elif job == 'mi_corr_aggr_fold_data' :
            # get individial data point mi corr measure
            saved_kendalltau = Path(save_dir)/'kendalltau'
            aggregate_msfi(target_folder="kendalltau_mrnet", col_name='kendalltau', root = saved_kendalltau, save_dir = Path(save_dir)/'kendalltau_mrnet', fold_range = [6,7,8,9,10])
            # aggregate_msfi(target_folder="kendalltau_shortcut", col_name='kendalltau', root = mi_dir, save_dir = Path(save_dir)/'kendalltau') # cannot calculate mi corr as no gt

            # hm_result_csv_root = '/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/msfi_shortcut' # cannot calculate mi for shortcut, as there is no gt MI for the five models

    elif job == 'msfi_readhm':
        # record sum(heatmap) within gt seg map, for each modality
        save_path = Path(csv_save_dir) / 'msfi'
        if seed is not None:
            save_path = Path(save_path)/ 'seed_{}'.format(seed)
            fold_or_seed = seed
        else:
            fold_or_seed = fold
        save_path.mkdir(parents=True, exist_ok=True)

        if config['machine'] == 'cc': # fix get_heatmaps() bug
            save_dir = Path(save_dir) / 'get_hm_seed_{}_fold_1'.format(seed)

        get_save_modality_hm_value(hm_save_dir=save_dir, result_save_dir=save_path,
                                   segment_path=segment_path,  # important difference from mi_readhm
                                   positiveHMonly=True, portion_metrics = True,
                                   modalities=modalities, fold=fold_or_seed, method_list=method_list)

    elif job == 'msfi_aggr_fold_data': #todo debug
        # compute msfi using sum(heatmap) and modality importance valoue, and aggregate different models
        compute_mfsi(hm_result_csv_root= Path(csv_save_dir) / 'msfi', gt_csv_path=Path(csv_save_dir)/'mi',
                     modalities=modalities)

        # after all 5 folds
        aggregate_msfi(target_folder="msfi_featshapley", col_name='msfi') #todo change target dir
    else:
        print("ERROR! Typed the wrong {} job!".format(job))


def compare_seg_map(config, save_dir):
    evaluator = BaseXAIevaluator(config)
    # save_dir = config.get_root_dir()
    # save_dir = '../exp_log/xai_exp/0219_multiModalEval_BRATS_IDH_ts/0224_085524_fold_1/test_vanilla_heatmaps_BRATS_IDH_fold_1'
    # save_dir = '../exp_log/xai_exp/0219_multiModalEval_BRATS_IDH_ts/0219_095505_fold_1/test_vanilla_heatmaps_BRATS_IDH_fold_1'
    # root = '/local-scratch/'
    fold = config['data_loader']['args']['fold']
    # save_dir = Path(root + 'authorid/trained_model/BRATS20_HGG/heatmaps/get_hm_fold_{}'.format(fold)) # directory of saved hms
    print("HM dir:", save_dir)
    data_root = config['data_loader']['args']['data_dir']
    seg_path =  Path(data_root)/'all'
    seed = None
    try:
        seed = config['seed']
    except:
        seed = None
    if (config['machine'] in ['ts', 'cc']) and (seed is not None):  # fix get_heatmaps() bug on local ts machine
        save_dir = Path(save_dir) / 'get_hm_seed_{}_fold_1'.format(seed)

    evaluator.compare_with_gt_dataset(save_dir, seg_path)


# def acc_drop_plot(config, save_dir):
#     ad = Acc_Drop(config=config)
#     ad.plot_acc_drop(save_dir)




def acc_drop_exp(config, save_dir, baseline_value_type ='acc_drop'):
    '''
    :param config:
    :param save_dir: the directory for heatmaps
    :param output_dir_name: the experiment label to save the output files
    :return:
    '''
    ad = Acc_Drop(config= config, baseline_value_type = baseline_value_type)
    # model =ad.get_model()
    # val_loader = ad.get_val_dataloader()
    # save_dir = '../exp_log/xai_exp/0219_multiModalEval_BRATS_IDH_ts/0219_095505_fold_1/test_vanilla_heatmaps_BRATS_IDH_fold_1'
    # save_dir = config.get_root_dir()
    # fold = config['data_loader']['args']['fold']
    seed = config['seed']
    save_dir = Path(save_dir)/'get_hm_seed_{}_fold_1'.format(seed)
    ad.pipeline(save_dir)
    # aggregate_df, g, modified_input_save_root = ad.pipeline(save_dir)
    # modified_input_save_root = ad.acc_drop(save_dir)
    logging.info("Acc Drop Done!")
    # ad.plot_acc_drop(modified_input_save_root)

def create_heatmaps_pipeline(config):
    # root_dir = config.get_root_dir()
    print('best model',  config['best_model'], 'fold: ', config['data_loader']['args']['fold'])

    ad = Acc_Drop(config= config)#, root_dir = root_dir)
    model =ad.get_model()
    val_loader = ad.get_val_dataloader()
    try:
        seed = '_seed_{}'.format(config['seed'])
        print('seed is', config['seed'])
    except:
        seed = ''
    print('seed is', seed)
    save_dir = ad.generate_save_heatmap(val_loader, model, 'get_hm{}'.format(seed))

    # save_dir = '../exp_log/xai_exp/0219_multiModalEval_BRATS_IDH_ts/0219_095505_fold_1/test_vanilla_heatmaps_BRATS_IDH_fold_1'
    logging.info(save_dir)
    accuracy = ad.save_dir_sanity_check(save_dir)
    if accuracy:
        logging.info("Accuracy is {}".format(accuracy))




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='XAI_MIA')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-t', '--ts', default=None, type=str,
                      help='Machine: ts or cc')
    args.add_argument('-j', '--job', default=None, type=str,
                      help='Tumorsyn which task')
    args.add_argument('--seed', default=42, type=int)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--fold'], type=int, target='data_loader;args;fold')
    ]


    tmp_config  = read_json(Path(sys.argv[ sys.argv.index("--config") +1]))
    task = tmp_config['name']
    machine = sys.argv[ sys.argv.index("--ts") +1]
    logging.info("===Task Name: {}, Machine: {}===".format(task, machine))
    job = None
    # add interface to be able to denote which job to run for the script
    if '--job' in sys.argv: #machine ='ts':
        job = sys.argv[ sys.argv.index("--job") +1]
    if '--ts' in sys.argv: #machine ='ts':
        if machine == 'ts':
            if task == "BRATS_HGG":
                prefix = "/local-scratch/authorid/trained_model/BRATS20_HGG" # obsolete, trained using 5fold cv
                # prefix = "/local-scratch/authorid/trained_model/brats20_5seed_traind_models/" # used for reporting, trained using 5 random seeds
            elif task == "BRATS_IDH":
                prefix = '/local-scratch/authorid/trained_model_BRATSIDH/'
            elif task == "tumorsyn":
                prefix = '/local-scratch/authorid/trained_model/tumorsyn/'
        elif machine == 'cc': # machine = cc
            # prefix = '/scratch/authorid/BRATS_IDH/log/'
            prefix = '/scratch/authorid/results_brats_rerun/trained_models/'
        elif machine == 'solar':
            if task == "BRATS_HGG":
                prefix = '/project/labname-lab/authorid/trained_model/BRATS20_HGG/'
                prefix = '/project/labname-lab/authorid/results_brats_rerun/trained_models/'
            elif task == "tumorsyn":
                prefix = '/project/labname-lab/authorid/trained_model/tumorsyn/'
            elif task == "tumorsyn_shortcut":
                prefix = '/project/labname-lab/authorid/trained_model/tumorsyn_epoch55_shortcut_model/'
    # print(machine, prefix)

    # customized to get fold number as input to run the experiments
    # best_model_path_folds = {1: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_023438_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_1/models/checkpoint-epoch34.pth",
    #                          2: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_043903_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_2/models/checkpoint-epoch37.pth",
    #                          3: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_053928_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_3/models/checkpoint-epoch92.pth",
    #                          4: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_055949_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_4/models/checkpoint-epoch54.pth",
    #                          5: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_064940_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_5/models/checkpoint-epoch11.pth"
    #                          }
    if task ==  "BRATS_HGG":
        best_model_path_folds = {1: prefix+"/fold1-epoch32.pth",
                                 2: prefix+"/fold2-epoch49.pth",
                                 3: prefix+"/model_best_epoch_55_fold_3_BRATS_HGG.pth",
                                 4: prefix+"/model_best_epoch_65_fold_4_BRATS_HGG.pth",
                                 5: prefix+"/model_best_epoch_30_fold_5_BRATS_HGG.pth"
                                 }
        best_model_path_seed = {10: prefix+"s10-checkpoint-epoch45.pth",
                                20: prefix+"s20_checkpoint-epoch66.pth",
                                30: prefix+"s30-checkpoint-epoch54.pth",
                                40: prefix+"s40-checkpoint-epoch87.pth",
                                50: prefix+"s50-checkpoint-epoch46.pth"}
    elif task == "tumorsyn":
        best_model_path_folds = {3: prefix+"model_best_epoch_86_fold_3_tumorsyn.pth",
                                 6: prefix+"model_best_epoch_68_fold_6_tumorsyn.pth",
                                 7: prefix+"model_best_epoch_69_fold_7_tumorsyn.pth",
                                 8: prefix+"model_best_epoch_120_fold_8_tumorsyn.pth",
                                 9: prefix+"model_best_epoch_157_fold_9_tumorsyn.pth"
                                 }
    elif task == "tumorsyn_shortcut":
        best_model_path_folds = {1: prefix+"s1_checkpoint-epoch55.pth",
                                 2: prefix+"s2_checkpoint-epoch55.pth",
                                 3: prefix+"s3_checkpoint-epoch56.pth",
                                 4: prefix+"s4_checkpoint-epoch55.pth",
                                 5: prefix+"s5_checkpoint-epoch55.pth"
                                 }
    # data_dir = '/local-scratch/authorid/dld_data/brats2020/MICCAI_BraTS2020_TrainingData/'


    if '--fold' in sys.argv: # if passed fold parameter in the bash script
        fold = int(sys.argv[ sys.argv.index("--fold") +1])
        if '--seed' in sys.argv:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
            modification = {'best_model': best_model_path_seed[seed], 'seed': seed, 'machine': machine}
        else:
            # corresponds fold with best_model path
            modification = {'best_model': best_model_path_folds[fold], 'machine': machine}
        modification['job'] = job
        config = ConfigParser.from_args(args, options, modification)
        main(config, machine, job)
    else:
        for i in best_model_path_folds:
            fold = i
            # corresponds fold with best_model path
            modification = {'best_model': best_model_path_folds[fold],  'data_loader;args;fold': fold}
            config = ConfigParser.from_args(args, options, modification)
            main(config, machine, job)


