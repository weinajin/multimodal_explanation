import argparse
import sys
import collections
from pathlib import Path
import csv
import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix
import model.model as module_arch
from utils.parse_config import ConfigParser
from datetime import datetime
import glob

def run_cv(fold_in_arg, seed_in_arg = None):
    # customized to get fold number as input to run the experiments
    best_model_path_folds = {1: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_023438_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_1/models/checkpoint-epoch34.pth",
                             2: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_043903_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_2/models/checkpoint-epoch37.pth",
                             3: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_053928_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_3/models/checkpoint-epoch92.pth",
                             4: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_055949_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_4/models/checkpoint-epoch54.pth",
                             5: prefix+"0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc/0210_064940_0210_miccai_eval_get_trained_model_5foldcv_BRATS_IDH_cc_fold_5/models/checkpoint-epoch11.pth"
                             }
    best_model_path_folds = {1: prefix+"/fold1-epoch32.pth",
                             2: prefix+"/fold2-epoch49.pth",
                             3: prefix+"/model_best_epoch_55_fold_3_BRATS_HGG.pth",
                             4: prefix+"/model_best_epoch_65_fold_4_BRATS_HGG.pth",
                             5: prefix+"/model_best_epoch_30_fold_5_BRATS_HGG.pth"
                             }
    best_model_path_seed = {10: prefix + "s10-checkpoint-epoch45.pth",
                            20: prefix + "s20_checkpoint-epoch66.pth",
                            30: prefix + "s30-checkpoint-epoch54.pth",
                            40: prefix + "s40-checkpoint-epoch87.pth",
                            50: prefix + "s50-checkpoint-epoch46.pth"}
    data_dir = '/local-scratch/authorid/dld_data/brats2020/MICCAI_BraTS2020_TrainingData/'
    data_dir = '/scratch/authorid/dld_data/brats2020/MICCAI_BraTS2020_TrainingData/'
    if fold_in_arg: # if passed fold parameter in the bash script
        fold = int(sys.argv[ sys.argv.index("--fold") +1])
        print('seed_in_arg fold', fold)
        if seed_in_arg:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
            modification = {'best_model': best_model_path_seed[seed], 'seed': seed}
        else:
            # corresponds fold with best_model path
            modification = {'best_model': best_model_path_folds[fold]}
        print("best model: {}".format(modification))
        config = ConfigParser.from_args(args, options, modification)
        test(config)
    else:
        for i in best_model_path_folds:
            fold = i
            # corresponds fold with best_model path
            modification = {'best_model': best_model_path_folds[fold],  'data_loader;args;fold': fold,  'trainer;save_dir': prefix, 'data_loader;args;data_dir' :data_dir}
            config = ConfigParser.from_args(args, updates = modification)
            test(config)

def test(config, timestamp = False,
         csv_save_dir = None, modality_selection= None, ablated_image_folder = None, # parameters for modality shapley
         image_label = None, saveinput = False, val_first_gt_align_prob = None, val_sec_gt_align_prob = None, combine_with_brain = [1,1,1,1] # parameters for tumorsyn
         ):
    """

    :param config:
    :param timestamp: if true, create multiple csv files that support multiple runs. Disable file_exist
    :param save_dir:
    :param modality_selection:
    :param ablated_image_folder:
    :param save_inputs:
    :param val_first_gt_align_prob:
    :param val_sec_gt_align_prob:
    :param combine_with_brain:
    :return:
    """
    val_not_test = config["data_loader"]["args"]["val_dataset"]
    val_or_test_label = ['test', 'val']
    logger = config.get_logger('cv_validation')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # build model architecture
    model = config.init_obj('arch', module_arch).to(device)
    # print('empty model', model.state_dict().keys())
    # print(config['trainer']['save_dir'])

    model_path = config["best_model"]
    logger.info("Loading model at {}".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    # print(state_dict.keys())
    ## prepare model for testing
    model.load_state_dict(state_dict)
    if config['n_gpu'] >= 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    # logger.debug(model)


    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=2,
    #     shuffle=False   )
    task = config['name']
    if task == "BRATS_HGG":
        data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.get_val_loader(modality_selection=modality_selection,
                                                       ablated_image_folder=ablated_image_folder)
    elif task == "tumorsyn":
        import data_loader.tumorgenerator_dataloader as tumorsyn_module_data
        data_loader = config.init_obj('data_loader', tumorsyn_module_data)
        # valid_data_loader = data_loader.get_train_loader(save_inputs = save_inputs) # just for debug
        logger.info('Validate.py: combine_with_brain = {}'.format(combine_with_brain))
        # print('test  val_first_gt_align_prob {}, val_sec_gt_align_prob  {}'.format(val_first_gt_align_prob, val_sec_gt_align_prob))
        valid_data_loader = data_loader.get_val_loader(val_first_gt_align_prob = val_first_gt_align_prob,
                                                       val_sec_gt_align_prob=val_sec_gt_align_prob,
                                                       save_inputs=saveinput,
                                                       combine_with_brain = combine_with_brain)
    fold = config['data_loader']['args']['fold']
    try:
        seed = config['seed']
    except:
        seed = None
    gts = []
    preds = []
    if image_label:
        image_key = image_label
    else:
        image_key = config['image_key']
    logger.info("--> Use label = {}".format(image_key))
    if not csv_save_dir: # keep modality shapley record in the same folder
        csv_save_dir = config['trainer']['save_dir']
    if modality_selection:
        save_path = Path(csv_save_dir)
    elif task == "tumorsyn": # test with different input ablation, specify whether save input test image or not
        if saveinput:
            save_path = Path(saveinput)
        else:
            save_path = Path(csv_save_dir)
    else:
        save_path = Path(csv_save_dir) / val_or_test_label[int(val_not_test)]
    save_path.mkdir(parents = True, exist_ok= True)
    if modality_selection:
        mod_fn = '-'.join([str(i) for i in modality_selection])
        if seed is not None:
            fold_or_seed = seed
        else:
            fold_or_seed = fold
        csv_filename = save_path / 'cv_result_fold_{}{}.csv'.format(fold_or_seed,  '-'+mod_fn)
    elif seed is not None:
        csv_filename = save_path / 'test_result_fold_{}_seed_{}.csv'.format(fold, seed)
    else:
        csv_filename = save_path/ 'cv_result_fold_{}.csv'.format(fold)

    fnames = ['dataID',  'gt', 'pred','softmax_0', 'softmax_1']
    file_exists = os.path.isfile(csv_filename)
    if (not timestamp) and file_exists and (modality_selection ):# or save_inputs):
        logger.info("{} file exist, pass\n".format(csv_filename))
        return
    if timestamp:
        dateTimeObj = datetime.now().strftime("%Y%m%d_%H%M")
        csv_filename = csv_filename.parent/ '{}-{}'.format(dateTimeObj, csv_filename.name)
    logger.info("CV will be saved at: {}".format(csv_filename))

    with torch.no_grad():
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
            if timestamp or (not file_exists):
                csv_writer.writeheader()
            for idx, data in enumerate(tqdm(valid_data_loader)):
                input, target, d_id = data[image_key], data['gt'], data['bratsID']
                input, target = input.to(device), target.to(device)

                output = model(input)
                responses = F.softmax(output, dim=1).cpu().numpy()
                responses = [responses[j] for j in range(responses.shape[0])]
                pred = torch.argmax(output, axis=1).cpu().detach().numpy()
                # print(output)
                # print(target)
                for i, r in enumerate(responses):
                    csv_record = {'dataID': d_id[i], 'gt': target.cpu().numpy()[i],
                                  'pred': pred[i], 'softmax_0': r[0], 'softmax_1': r[1]}
                    csv_writer.writerow(csv_record)

                #
                # save sample images, or do something with output here
                #

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data['gt'].shape[0]
                total_loss += loss.item() * batch_size
                for m, metric in enumerate(metric_fns):
                    total_metrics[m] += metric(output, target) * batch_size


    n_samples = len(valid_data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

def report_cv_result(log_path):
    reporting_csv = 'reporting_5foldcv_result.csv'
    cm = []
    precision = []
    recall = []
    f1 = []
    auroc = []
    acc = []
    acc_dict = dict()
    # csv_filename = Path(log_path).rglob('cv_result_*.csv')
    csv_filename = glob.glob( os.path.join(log_path, "*cv_result_fold_*.csv")) # only read csv in current dir, not subdir
    for f in csv_filename:
        f = Path(f)
        print(f)
        # print(f, f.name.split('.')[0].split('_'), int(f.name.split('.')[0].split('_')[-1]))
        if f.name != reporting_csv.split('.')[0]:
            # check if it has header
            with open(f) as fl:
                first = fl.read(1)
            has_header =  first not in ',.-0123456789'
            # print(has_header)
            if has_header:
                results = pd.read_csv(f)
            else:
                print("{} missing header".format(f))
                results = pd.read_csv(f, names=['dataID', 'gt', 'pred', 'softmax_0', 'softmax_1'])
            fold = int(f.name.split('.')[0].split('_')[-1])
            gt = results['gt']
            pred = results['pred']
            accuracy = accuracy_score(gt, pred)
            acc_dict[fold] = accuracy
            acc.append(accuracy)
            cm.append(confusion_matrix(gt, pred))
            print(fold, confusion_matrix(gt, pred))
            precision.append(precision_score(gt, pred, average=None))
            recall.append(recall_score(gt, pred, average=None))
            auroc.append(roc_auc_score(gt, pred, average=None))
            f1.append(f1_score(gt, pred, average=None))
    print(sorted(acc_dict.items()))
    print(precision)
    # calculate mean and std
    cm = np.asarray(cm)
    acc = [(c[0][0] + c[1][1] )/c.sum()  for c in cm]
    df = {'precision-0': np.asarray(precision).mean(axis=0)[0], \
            'precision-0-std':np.asarray(precision).std(axis=0)[0],\
            'precision-1':np.asarray(precision).mean(axis=0)[1],\
            'precision-1-std':np.asarray(precision).std(axis=0)[1],\
            'recall-0':np.asarray(recall).mean(axis=0)[0], \
            'recall-0-std':np.asarray(recall).std(axis=0)[0], \
            'recall-1':np.asarray(recall).mean(axis=0)[1], \
            'recall-1-std':np.asarray(recall).std(axis=0)[1], \
            'acc':np.asarray(acc).mean(), \
            'acc-std':np.asarray(acc).std(),\
            'f1':np.asarray(f1).mean(), \
            'f1-std':np.asarray(f1).std(), \
            'auroc':np.asarray(auroc).mean(), \
            'auroc-std':np.asarray(auroc).std(), \
            'cm':cm.sum(axis=0).tolist()\
            }
    pd.DataFrame(df).to_csv(os.path.join(log_path, reporting_csv))
    print(df)
    return

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='XAI_test')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--ts', default=None, type=str,
                      help='Machine: ts or cc')
    args.add_argument('--seed', default=None, type=int)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--fold'], type=int, target='data_loader;args;fold')
    ]
    task = "BRATS_HGG"

    if '--ts' in sys.argv: #machine ='ts':
        if task == "BRATS_HGG":
            prefix = "/local-scratch/authorid/trained_model/BRATS20_HGG"
            log_path = "/local-scratch/authorid/trained_model/BRATS20_HGG/test"  # log to save the 5folds cross-validation csv
            # new brats rerun study
            prefix = '/local-scratch/authorid/trained_model/brats20_5seed_traind_models/'
            log_path = '/local-scratch/authorid/trained_model/brats20_5seed_traind_models/test'

        elif task == "BRATS_IDH":
            prefix = '/local-scratch/authorid/trained_model_BRATSIDH/'
        if task == 'tumorsyn':
            log_path = "/local-scratch/authorid/trained_model/tumorsyn/tumor4mod_7/baseline"
    else: # machine = cc
        # prefix = '/scratch/authorid/BRATS_IDH/log/'
        prefix = '/scratch/authorid/results_brats_rerun/trained_models/'
        log_path = '/scratch/authorid/results_brats_rerun/trained_models/test'

    fold_in_arg = '--fold' in sys.argv
    seed_in_arg = '--seed' in sys.argv
    print('seed in arg', seed_in_arg)

    log_path = None # Set none to run test, comment out to aggregate multiple model performance
    if log_path:
        report_cv_result(log_path)
    else:
        run_cv(fold_in_arg = fold_in_arg, seed_in_arg = seed_in_arg)




