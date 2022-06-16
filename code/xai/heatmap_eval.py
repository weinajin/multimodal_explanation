'''
Experiment of randomize model parameters and check heatmap similarity change
Adapted from https://github.com/adebayoj/sanity_checks_saliency/blob/master/src/randomize_inceptionv3.py
'''

from pathlib import Path
from tqdm import tqdm

import os
import glob
import pickle
# import logging
import numpy as np
import pandas as pd
from datetime import datetime
# import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, auc, roc_curve
import torch

from monai.transforms import Resize, Compose, Flip, Flipd, Rotate, Rotated, Rotate90d, LoadNiftid,Orientationd, NormalizeIntensityd, Resized, ToTensord

from .generate_heatmaps import *
from .heatmap_utlis import *
from utils.util import read_json, write_json

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.parse_config import ConfigParser
from utils.trainer import Trainer
from utils.util import prepare_device
# from train import train
import shutil
import seaborn as sns

from skimage import measure
from scipy.ndimage import binary_dilation

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# modality = ['t1', 't1ce', 't2', 'flair']
# method_list = ['Gradient', 'GuidedBackProp', 'GuidedGradCAM']
# method_list = ['Gradient', 'GuidedBackProp', 'GuidedGradCAM']

class BaseXAIevaluator:
    def __init__(self, config, load_model = True, load_data = True):
        """
        Include 5 xai eval experiments
        :param train:
        """
        # xai specific configs
        self.logger = config.get_logger('xai', config['trainer']['verbosity'])
        self.config = config
        self.save_dir = config.get_root_dir()

        self.exp_name = self.config['name']
        self.modality = self.config['xai']['modality']
        self.method_list = self.config['xai']['method_list']

        # prepare for (multi-device) GPU training
        self.device, self.device_ids = prepare_device(config['n_gpu'])
        model_path = self.config["best_model"]
        self.logger.info('Loading checkpoint: {}'.format(model_path))
        if load_model:
            self.model = self.load_model(model_path, reinit=False) # load trained model to eval the XAI methods
        self.task = config['name']
        if self.task == "BRATS_HGG":
            self.data_loader = self.config.init_obj('data_loader', module_data)
        elif self.task == "tumorsyn" or self.task == "tumorsyn_shortcut":
            import data_loader.tumorgenerator_dataloader as tumorsyn_module_data
            self.data_loader = self.config.init_obj('data_loader', tumorsyn_module_data)
        self.fold = self.config['data_loader']['args']['fold']
        try:
            self.seed = self.config['seed']
        except:
            self.seed = None
        self.last_layer = self.config["xai"]["last_layer"]
        self.gnt_hm_ds_args = {"device": self.device, "method_list": self.method_list, "last_layer": self.last_layer, "task": self.task}
        if load_data:
            self.train_data_loader = self.data_loader.get_train_loader()
            self.valid_data_loader = self.data_loader.get_val_loader()
            self.gnt_hm_ds_args["image_key"]= self.config["image_key"]


        self.logger.info('\nIn fold {} seed {}, {} XAI methods to eval: {}'.format(self.fold, self.seed, len(self.method_list), self.method_list))

        self.gradient_method = ["Gradient", "InputXGradient", "SmoothGrad",
                                 "Deconvolution","GuidedBackProp", "GuidedGradCAM",
                                 "IntegratedGradients", "DeepLift", "GradientShap"
                                 ]
        self.perturbation_method = ["Occlusion", "FeatureAblation", "KernelShap", "ShapleyValueSampling",
                                   "FeaturePermutation", "Lime"]
        self.activation_method = ["GradCAM"]
    def get_model(self):
        return self.model
    def get_dataloader(self):
        return self.data_loader
    def get_val_dataloader(self):
        return self.valid_data_loader

    def load_model(self, model_path, reinit= False):
        '''
        build model architecture, then print to console
        :param reinit: if reinit: randomize model parameters. If not, load trained model parameters.
        :return:
        '''
        model = self.config.init_obj('arch', module_arch).to(self.device)

        # self.logger.debug(model)
        if reinit:
            # if len(self.device_ids) >= 1:
                # model = torch.nn.DataParallel(model, device_ids=self.device_ids).to(self.device)
            model.eval()
            return model
        # if len(self.device_ids) >= 1:
            # model = torch.nn.DataParallel(model, device_ids=self.device_ids).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device)['state_dict'])
        model.eval()
        return model


    # def __load_data(self):
    #     # setup data_loader instances
    #     self.data_loader = self.config.init_obj('data_loader', module_data)

    # def load_training_data(self):
    #     self.train_data_loader = self.data_loader.get_train_loader()
    #     return self.train_data_loader
    #
    # def load_val_data(self):
    #     self.valid_data_loader = self.data_loader.get_val_loader()
    #     return self.valid_data_loader

    def save_heatmap_pickle(self, save_dir, before = None, after = None, postfix= ''):
        '''
        Obsolete func
        '''
        if self.exp_name:
            save_dir = save_dir / self.exp_name
        else:
            dateTimeObj = datetime.now()
            save_dir += dateTimeObj.strftime("%Y%m%d_%H%M")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if postfix: # add prefix to specific saved file version
            postfix = '_'+postfix
        if before:
            pickle.dump(before,  open(os.path.join(save_dir, 'before{}.pkl'.format(postfix)), 'wb'))
        if after:
            pickle.dump(after, open(os.path.join(save_dir, 'after{}.pkl'.format(postfix)), 'wb'))
        return save_dir

    def save_experiment_results(self, heatmap_dict, file_name):
        '''
        obsolete function, as CPU OOM error when computing and keeping all dataloader in memeory
        create a new folder_name under self.save_dir
        :param heatmap_dict:
        :param folder_name:
        :return:
        '''
        # info in folder name, self.fold, timestamp
        dateTimeObj = datetime.now()
        file_name += "_fold_{}_{}".format(self.fold,  dateTimeObj.strftime("%Y%m%d_%H%M"))
        save_dir = self.save_dir #/ self.exp_name # exp_name is the experiment (aka class name)
        save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(heatmap_dict, open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb'))
        # save config file
        self.logger.info("\n{} saved at {}".format(file_name, save_dir))
        # save updated config file to the checkpoint dir
        return save_dir

    def generate_save_heatmap(self, data, model, exp_label):
        exp_label = "{}_fold_{}".format(exp_label, self.fold) # label of the
        save_dir = self.save_dir  / exp_label
        self.logger.info('Heatmaps saving dir: {}'.format(save_dir))
        save_dir.mkdir(parents=True, exist_ok=True)
        # default: generate heatmaps using CPU, unless specified
        self.gnt_hm_ds_args['device'] = "cpu"
        if 'Occlusion' in self.gnt_hm_ds_args["method_list"]: # process occlusion seperately, as it require GPU to pass many times
            self.gnt_hm_ds_args["method_list"].remove("Occlusion")
            occ_args = dict(self.gnt_hm_ds_args)
            occ_args["method_list"] = ["Occlusion"]
            occ_args['device'] = self.device
            generate_heatmap_dataset(data, model, save_dir, **occ_args)
        if self.task != "MRNet" and "FeaturePermutation" in self.gnt_hm_ds_args["method_list"]:  # default bs = 1, change bs if FeaturePermutation
            self.gnt_hm_ds_args["method_list"].remove("FeaturePermutation")
            bs = 4
            self.logger.info('Loading batch_size = {} for FeaturePermutation'.format(bs))
            if self.task == "BRATS_HGG":
                batched_val_loader = self.config.init_obj('data_loader', module_data,
                                                          **{"batch_size": bs, 'shuffle': True}).get_val_loader()
            elif self.task == "tumorsyn" or self.task == "tumorsyn_shortcut":
                import data_loader.tumorgenerator_dataloader as tumorsyn_module_data
                batched_val_loader = self.config.init_obj('data_loader', tumorsyn_module_data, **{"batch_size": bs, 'shuffle': True}).get_val_loader_presaved(saved_dir = "/local-scratch/authorid/trained_model/tumorsyn/7/test/get_hm_input")
            fp_args = dict(self.gnt_hm_ds_args)
            fp_args["method_list"] = ['FeaturePermutation']
            generate_heatmap_dataset(batched_val_loader, model, save_dir, **fp_args)
        if len(self.gnt_hm_ds_args["method_list"]) > 0:
            generate_heatmap_dataset(data, model,  save_dir, **self.gnt_hm_ds_args)

        # save_dir = self.save_experiment_results(heatmap_dict, file_name)
        return save_dir

    def save_dir_sanity_check(self, save_dir):
        return save_dir_sanity_check(save_dir, method_list = self.method_list, valid_data_loader = self.valid_data_loader)

    def find_before_after_pickle(self, pickle_path):
        '''
        Used for randomized exp
        :param pickle_path:
        :return:
        '''
        # find pickle name with before after
        # onlyfiles = [f for f in os.listdir(pickle_path) if os.path.isfile(os.path.join(pickle_path, f))]
        before_file = glob.glob(os.path.join(pickle_path, 'before*.pkl'))
        after_file = glob.glob(os.path.join(pickle_path, 'after*.pkl'))
        assert len(before_file) == 1, self.logger.info('ERROR! multiple before files {} in {}'.format(before_file, pickle_path))
        assert len(after_file) == 1, self.logger.info('ERROR! multiple after files {} in {}'.format(after_file, pickle_path))
        before_file = before_file[0]
        after_file = after_file[0]
        before_heatmap_dict = pickle.load(open(before_file, "rb"))
        after_heatmap_dict = pickle.load(open(after_file, "rb"))
        return before_heatmap_dict, after_heatmap_dict

    def find_before_after_pickle_multiple(self, pickle_path):
        '''
        Used for consistency exp: consistency_pipeline_from_pickle()
        :param pickle_path:
        :return:
        '''
        # get transform names from affine_dict
        affine_dict = pickle.load(open(os.path.join(pickle_path,'affine_dict.pkl'), "rb"))
        # find pickle name with before after
        before_file = glob.glob(os.path.join(pickle_path, 'before*.pkl'))
        after_file = glob.glob(os.path.join(pickle_path, 'after*.pkl'))
        assert len(before_file) == 1, self.logger.info('ERROR! multiple before files {} in {}'.format(before_file, pickle_path))
        before_file = before_file[0]
        ori_heatmaps = pickle.load(open(before_file, "rb"))
        after_hm_dict = dict()
        for k in affine_dict:
            for f in after_file:
                if k in f:
                    aft = pickle.load(open(f, "rb"))
                    after_hm_dict[k] = aft
        return affine_dict, ori_heatmaps, after_hm_dict

    def compare_bfaf_heatmaps(self, before_heatmap_dict, after_heatmap_dict, reverse_transform = None, postfix= '', bg_mask_dict = None):
        '''
        Since the volume heatmaps of a val dataset is large, save the intermediate results to pickle
        This func read a pickle path, and get the pickle files of two heatmap dict.
        Experiment results for before-after conditions:
            - the heatmap similarity change.
            - the accuracy change.
            - time spent on heatmap generation (should be more or less the same, using t-test)
        :param
            pickle_path: path to save the heatmap and all intermediate results
            seg_root_path: path root to the brats dataset
            reverse_transform: used for consistency exp, to reverse transform the heatmaps and compare similarity with original hms
        heatmap_dict[bratsID] = (heatmap_dict of XAI methods, pred, labels_val, time_record_dict of XAI methods)
        :return:
        a df with xai method in the col/key, and each metrics in the row, including:

        '''
        # init saved variables
        heatmap_b_a = dict()
        time_b = [] # 2d array, [bratsID, xai method]
        time_a = []
        pred_b = []
        pred_a = []
        gt_b = []
        gt_a = []
        bratsID_lst = list(before_heatmap_dict.keys())

        # read data from before/after dict and save in list
        for k in before_heatmap_dict:
            heatmap_b_a[k] = (before_heatmap_dict[k][0], after_heatmap_dict[k][0])
            time_b.append(before_heatmap_dict[k][3]) # a list of time cost for xai methods
            time_a.append(after_heatmap_dict[k][3])
            pred_b.append(before_heatmap_dict[k][1])
            gt_b.append(before_heatmap_dict[k][2])
            pred_a.append(after_heatmap_dict[k][1])
            gt_a.append(after_heatmap_dict[k][2])

        # get accuracy before and after
        acc_b = accuracy_score(gt_b, pred_b)
        acc_a = accuracy_score(gt_a, pred_a)

        # init result to save
        # method_list = heatmap_b_a[0][0].keys()
        results = {k: dict() for k in self.method_list}
        if postfix:
            postfix = '_'+postfix

        # get average time cost, and before/after time difference
        time_b, time_a = np.array(time_b), np.array(time_a)
        for i in range(len(self.method_list)):
            xai = self.method_list[i]
            t, p = ttest_rel(time_b[:,i], time_a[:,i])
            results[xai]['time_mean_before'] = time_b[:,i].mean()
            results[xai]['time_std_before'] = time_b[:,i].std()
            results[xai]['time_mean_after'] = time_a[:,i].mean()
            results[xai]['time_std_after'] = time_a[:,i].std()
            results[xai]['time_diff_ttest'] = p

        # heatmap similarity
        sim_method = None
        for i in range(len(self.method_list)): # iterate via xai methods
            xai = self.method_list[i]
            dataset_sim = []
            for id in heatmap_b_a: # iterate via bratsID
                # sim: 2d list of [mod, sim_method_val]
                before_hm = heatmap_b_a[id][0][xai]
                if bg_mask_dict: # mask before hm with bg mask as gt, used for detect_bias()
                    before_hm = mask_heatmap(before_hm, bg_mask_dict[id])
                    # upsample before and after heatmap to mask size = original input size (240, 240, 155)
                    after_hm = Resize(spatial_size=bg_mask_dict[id])(after_hm)
                    assert before_hm.shape == after_hm.shape, 'ERROR! When masking bg hm, bg_hm.shape != after_hm.shape'
                after_hm = heatmap_b_a[id][1][xai]
                if reverse_transform: # used for consistency()
                    after_hm = reverse_transform(after_hm)
                sim, sim_method = self.compare_heatmap_similarity_all_mod(before_hm, after_hm)
                dataset_sim.append(sim) # 3d list of [bratsID, mod, sim_method_val]
            # average sim for dataset
            dataset_sim = np.array(dataset_sim)
            mean_allmod= np.mean(dataset_sim, axis = (0,1))
            std_allmod = np.std(dataset_sim, axis = (0,1))
            # average sim for each mod
            mean = np.mean(dataset_sim, axis = 0)
            std = np.std(dataset_sim, axis=0)
            for i in range(len(sim_method)):
                sim_m = sim_method[i]
                results[xai][sim_m+'_mean_allmod'] = mean_allmod[i]
                results[xai][sim_m+'_std_allmod'] = std_allmod[i]
                for j in range(len(self.modality)):
                    results[xai][sim_m+'_mean_'+self.modality[j] ] = mean[j][i]
                    results[xai][sim_m + '_std_' + self.modality[j]] = std[j][i]
        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        results_filename = os.path.join(pickle_path, 'results{}_{}.csv'.format(postfix, time_stamp))
        results_df = pd.DataFrame(data = results)
        results_df.to_csv(results_filename)
        acc_df = pd.DataFrame(data = [acc_b, acc_a], index = ['acc_before', 'acc_after'])
        acc_filename = os.path.join(pickle_path, 'acc{}_{}.csv'.format(postfix, time_stamp))
        print(acc_filename)
        acc_df.to_csv(acc_filename)
        return results, acc_df

    def compare_bfaf_heatmaps_pipeline(self, pickle_path):
        before_heatmap_dict, after_heatmap_dict = self.find_before_after_pickle(pickle_path)
        results, acc_df = self.compare_bfaf_heatmaps(before_heatmap_dict, after_heatmap_dict)
        return results, acc_df

    def compare_with_gt_dataset(self, save_dir, seg_path):
        """
        Compare the hm with gt segmentation maps
        :param save_dir:
        :param seg_path:
        :return:
        """
        # data_ids, df = get_data_ids(save_dir)
        # heatmap compare with seg map gt
        # before_file = glob.glob(os.path.join(save_dir, '*before.pkl'))
        # assert len(before_file) == 1, print('ERROR! multiple before files {} in {}'.format(before_file, pickle_path))
        # heatmap_dict = pickle.load(open(before_file[0], "rb"))
        gt_compare_col = None
        for m in self.method_list:
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
                post_hm = postprocess_heatmaps(hm, no_neg = True, rotate_axis=False)
                if post_hm.min() == post_hm.max():
                    print("Warning, for data {}, min {} == max {}".format(dataID, post_hm.min(), post_hm.max()))
                    continue
                # print(dataID, hm.shape, post_hm.shape)
                # new_hm_dict[dataID] = post_hm
            # hms, _ = get_heatmaps(save_dir, m, by_data=False, hm_as_array=False)
            # for bratsID in data_ids:
                seg = read_seg(seg_path, dataID)
                # seg = np.rot90(seg, k=3, axes=(0, 1))  # important, avoid bug of seg, saliency map mismatch
                # print(post_hm.shape) #(4, 240, 240, 155)
                gt_r, gt_compare_col = compare_with_gt(seg, post_hm) # [tumor_port + IoU, 4 mods]
                gt_results.append(gt_r)
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

class Randomize_Model(BaseXAIevaluator):
    def __init__(self, config):
        super().__init__(config)
        self.save_dir = None
        self.results, self.acc_df = None, None


    def randomize_model(self, computnormalmasks = True):
        '''
        Heatmap eval experiment
        '''
        # compute normal heatmaps
        output_heatmaps_b = None
        if computnormalmasks:
            # heatmap_dict[bratsID] = (heatmap, pred, labels_val, time_record list)
            output_heatmaps_b = generate_heatmap_dataset(
                self.valid_data_loader, self.model,  exp_condition = 'model_rand_before', **self.gnt_hm_ds_args)
            self.logger.info("Done with computing heatmaps with normal model.")
        # create reinitialized model
        randomized_model = self.load_model(reinit= True)
        randomized_model_heatmap_dict = generate_heatmap_dataset(
            self.valid_data_loader , randomized_model,exp_condition= 'model_rand_after', **self.gnt_hm_ds_args)
        # heatmap_dict[bratsID] = (heatmap, pred, labels_val, time_record)
        self.logger.info("Done with computing heatmaps with randomized MODEL.")
        # setup save dir
        save_dir = self.save_dir/ 'randomize_model'
        save_dir = self.save_heatmap_pickle(before= output_heatmaps_b,
                            after= randomized_model_heatmap_dict,
                            exp_dir = save_dir,
                            exp_name = self.exp_name)
        self.save_dir = save_dir
        return self.save_dir

    def pipeline(self):
        pickle_path = self.randomize_model(computnormalmasks= True)
        self.results, self.acc_df = self.compare_bfaf_heatmaps_pipeline(pickle_path)
        self.logger.info("randomize model results: \n{}\nAcc_df:\n{}".format(self.results, self.acc_df))
        return self.results, self.acc_df

class Randomize_Label(BaseXAIevaluator):
    def __init__(self, config):
        ill_model_path = self.config.xai.random_label_model
        self.ill_model = self.load_model(ill_model_path)
        super().__init__(config)
        self.save_dir = None

    def randomize_label(self, ill_model, normal_model = None):
        '''
        Heatmap eval experiment
        :param dataloader: the normal valdataloader
        :param ill_model: the model already trained with randomized labels. Trained w/ opt.task = RANDOM_IDH
        :param method_list:
        :param computnormalmasks:
        :param exp_name:
        :return:
        '''
        normal_heatmaps = None
        # if normal model is given, compute normal heatmaps
        if normal_model:
            normal_heatmaps = generate_heatmap_dataset(
                self.valid_data_loader, self.model, exp_condition = 'label_rand_before', **self.gnt_hm_ds_args)
            self.logger.info("Done with computing heatmaps with normal model.")
        # create heatmpas with ill trained model on randomized label
        ill_heatmaps = generate_heatmap_dataset(
            self.valid_data_loader, ill_model, exp_condition='model_rand_after', **self.gnt_hm_ds_args)
        self.logger.info("Done with computing heatmaps with randomized LABEL.")
        # setup save dir
        save_dir = '../exp_log/heatmap_exp/randomize_label/'
        save_dir = self.save_heatmap_pickle(before= normal_heatmaps,
                            after= ill_heatmaps,
                            exp_dir = save_dir,
                            exp_name = self.exp_name)
        return self.save_dir

class Acc_Drop(BaseXAIevaluator):
    def __init__(self, config, load_model = True, load_data = True, baseline_value_type = 'acc_drop'):
        """
        :param config:
        :param load_model:
        :param load_data:
        :param baseline_value_type:  acc_drop: random sample from unimportant values; acc_drop_bg: replace with background value, acc_drop_nb: neighbor value
        """
        "save_dir: the dir that saves all heatmaps"
        super().__init__(config, load_model = load_model, load_data = load_data)
        self.steps = 10 # num of steps to ablate the input
        # self.exp_name += "/acc_drop"
        # self.save_dir = save_dir
        self.baseline_value_type = baseline_value_type
        if load_data:
            self.image_key = config['image_key']


    def neighbor_baseline_value(self, input, mask, zero_background = True):
        '''
        create abalated input with baseline value of mean of the neighboor pixel (conditional value)
        for the important feature mask, dilate and find connected components,
        and for each connected component, find its neighbor pixels means that are non-overlapping with other important features
        :param input: including all modalities
        :param mask: masked areas to be removed from the input; including all modalities
        :return: unimportant_image: fill in value for the mask
                 modified_input: input and fill in value for the mask overlaid
        '''
        # input_shape = input.shape
        # print(input.shape, mask.shape, 'neigh shape') # (W,H, D)

        unimportant_image = []
        for mod in range(len(input)):
            mod_inp = input[mod]
            mod_mask = mask[mod]
            # print(mod_inp.shape, mod_mask.shape)
            if zero_background:
                blobs_labels, num_blobs = measure.label(mod_mask, background=0, return_num=True, connectivity = 3)
            else:
                blobs_labels, num_blobs = measure.label(mod_mask, return_num=True, connectivity = 3)

            baseline_blobs = np.zeros(shape=blobs_labels.shape)
            # print('num_blob', num_blobs)#, np.unique(blobs_labels))
            for blob_idx in range(1, num_blobs):
                # print('blob_idx', blob_idx)
                blob = blobs_labels == blob_idx
                # print(np.unique(blob), np.count_nonzero(blob), 'blob size')
                edge = 3
                selem = np.ones((edge, edge, edge))
                neighbor_size = 0
                while neighbor_size <= 10: # ensure the neighbor has enough pixels to calculate
                    neighbor_and_blob = binary_dilation(blob, mask = mod_mask == 0, structure = selem).astype(int)
                    neighbor_only_mask = np.logical_and(neighbor_and_blob == 1,  blob != 1)
                    neighbor_size = np.count_nonzero(neighbor_only_mask)
                    edge += 1
                    selem = np.ones((edge, edge, edge))
                    if np.count_nonzero(neighbor_and_blob)!= np.count_nonzero(neighbor_only_mask)+ np.count_nonzero(blob):
                        self.logger.info('neighbor_and_blob {} != neighbor_only_mask {} + blob {}'.format(np.count_nonzero(neighbor_and_blob), np.count_nonzero(neighbor_only_mask), np.count_nonzero(blob)))
                    # print(neighbor_size,  'neighbor_size')
                neighbor_only = mod_inp[neighbor_only_mask == 1] # a flatten arry of masked neighbor pixels
                neighbor_mean = neighbor_only.mean() #np.ma.masked_equal(neighbor_only, 0).mean()   # mean of nonzero values
                unimportant_blob =  np.where(blobs_labels == blob_idx,  neighbor_mean, 0)# replace the blob value with neighbor_mean value
                baseline_blobs += unimportant_blob
            unimportant_image.append(baseline_blobs.astype(mod_inp.dtype))
        depths = [img.shape[-1] for img in unimportant_image]
        depth_is_same = all(element == depths[0] for element in depths)
        if depth_is_same:
            unimportant_image = np.stack(unimportant_image)#.astype(input.dtype)
        return unimportant_image



    def modify_input_predict(self, input, heatmaps, data_id, modified_input_save_dir, hm_normalize = False, get_rid_neg = False, ABS=False, bl_repeat_num = 0):
        '''
        Utility func for acc_drop. To reduce putting the generated modified_inputs in memory, generate a batch for all quantile stpes,
        and pass the full batch to predict, and save the input and prediction. Then iterate to the next random baseline input.
        :param input: one input image of np array
        :param heatmaps: a heatmap mask to pertube the input area.
        :param hm_normalize: whether to normalize a heatmap. Normalize to [-1, 1] using normalize_scale()
        :return: modified_input_lst: a modified_input of the same size. 0 is modified according to the heatmap, while the rest is random ablated baseline
        with the same num of important features, and ablated and filled use the same unimportant feature value.
        '''
        assert input.shape == heatmaps.shape, 'input {} and heatmap {} have different shape'.format(input.shape, heatmaps.shape)
        if hm_normalize:
            heatmaps = normalize_scale(heatmaps)
            # heatmaps = scale_hms(heatmaps)
        if ABS:
            heatmaps = np.absolute(heatmaps)
        if get_rid_neg:
            #  set negative regions to 0, only evalaute positive ones
            heatmaps[heatmaps<0] = 0

        # prepare quantile/cutoff steps
        quantile_step = [0]
        interval = 1.0/self.steps
        accumuate = 0.0
        for i in range(self.steps): # accumulated pertubation precentage
            accumuate += interval
            quantile_step.append(float(format(accumuate, '.5f')))
        # self.logger.debug('quantile steps to delete input: {}'.format(quantile_step))

        # for each ablation step, get
        # prev_bl_input = np.zeros(input.shape) # save the random baseline for prev steps, and ablate on top of it.
        preds = []

        # permute the oringial heatmap to create a bl_repeat_num of random heatmaps as random baseline
        heatmaps_and_random_bl = [heatmaps]
        for rpt in range(bl_repeat_num):
            # random_hm = np.random.permutation(heatmaps.flatten()).reshape(heatmaps.shape)
            # random_hm = self.random_heatmap(heatmaps, input)
            random_hm = np.random.permutation(heatmaps.flatten()).reshape(heatmaps.shape)
            heatmaps_and_random_bl.append(random_hm)

        # began gradual feature removal
        quantile_dict = {}
        for q_idx, cutoff in enumerate(quantile_step[1:]):
            modified_input_batch = []
            for idx, hm in enumerate(heatmaps_and_random_bl):
                # construct modified input
                # print(idx, 'idx')
                if q_idx not in quantile_dict: # save the quantile so that random baseline has the same quantile to ablate
                    quantile = np.quantile(hm[np.nonzero(hm)], 1.0-cutoff)
                    quantile_dict[q_idx] = quantile
                else:
                    quantile = quantile_dict[q_idx]
                # print("Step index {}, Quantile {}, for heatmap idx {}".format(q_idx, quantile, idx ))

                # if quantile == heatmaps.min():
                #     return None, None
                # if q_idx < len(quantile_step[1:])-1: # at last step, quantile can be equal to hm.min()
                #     assert quantile != heatmaps.min(), print('Error! quantile == hm.min() at step {}, cutoff value {}'.format(q_idx, cutoff)) #  todo remove the requirement for GradCAM
                # get the hm mask
                mask = hm >= quantile
                mask_size = mask.sum()
                # print('mask sum', mask.sum(), 'unimportant size', unimportant.shape, alpha, lower_quantile, heatmaps.min(), heatmaps.max())
                if self.baseline_value_type == 'acc_drop_nb' and idx == 0:
                    unimportant_image  = self.neighbor_baseline_value(input, mask) # fill in value for the mask
                else:
                    if self.baseline_value_type == 'acc_drop' :
                        unimportant_size = 0
                        alpha = 0.04
                        # get signal from unimportant regions
                        while unimportant_size <= 0:
                            alpha += 0.01
                            assert  alpha < 1.0, print('Error on identifying unimportant regions!')
                            lower_quantile = np.quantile(hm, alpha) # get the lower 5% quantile values of heatmap (including 0) as unimportant
                            unimportant = input[hm <= lower_quantile]
                            unimportant_size = unimportant.size
                        # used to ablate both input and random baseline
                        unimportant_image = np.random.choice(unimportant, size=input.shape,
                                                             replace=True)
                    elif (self.baseline_value_type == 'acc_drop_bg') or (self.baseline_value_type == 'acc_drop_nb' and idx != 0):
                        #use the background mean value (0s) for each modality
                        # applicable to bg setting, and neighbor setting for random heatmap baseline, as the heatmaps are permuted and it is computational expensive to compute their mean of neighbors' pixels
                        # assert input[:,0,0,:].mean() == input[:, 0,-1,:].mean() ==input[:, -1,0,:].mean()== input[:, -1,-1,:].mean() == 0
                        unimportant_image = np.zeros(shape = input.shape, dtype=np.float32)
                # remove background value and replace with 0
                unimportant_image = np.where(input == 0, 0, unimportant_image)
                modified_input = np.where(mask, unimportant_image, input)
                # save the modified_input
                if idx == 0:
                    modified_input_fn = "{}-{}.pkl".format(data_id, str(cutoff))
                else:
                    modified_input_fn = "{}-bl{}-{}.pkl".format(data_id, idx, str(cutoff))
                    pickle.dump(modified_input, open(os.path.join(modified_input_save_dir, modified_input_fn), 'wb')) #TODO visualize the results

                # get randomized mask, obsolete as the random baseline is not consistant for each quantile removal
                # random_bl_input = []
                # for i in range(bl_repeat_num):
                #     # permuted_mask = np.random.permutation(mask.flatten()).reshape(input.shape) # todo: another way is to shuffle in a unit of 3D blocks
                #     # the permuted_mask should be for input values != 0
                #     permuted_mask = self.random_place_mask(input, mask)
                #     bl_modified_input = np.where(permuted_mask, unimportant_image, input)
                #     random_bl_input.append(bl_modified_input)

                # vis for debug
                # dateTimeObj = datetime.now()
                # time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
                # img_name = 'acc_drop_modifyinputhm-{}-{}-{}'.format(data_id, cutoff, time_stamp)
                # vis_dir = image3d_to_gif(save_dir = '/local-scratch/authorid/vis_brats/{}'.format(img_name),
                #                          heatmaps = heatmaps, mris = modified_input, keep_png = True)
                # vis for debug end

                # modified_input_batch = [modified_input] + random_bl_input
                modified_input_batch.append(modified_input)

            modified_input_batch = np.stack(modified_input_batch)
            # pass the modified input to model
            modified_input = torch.as_tensor(modified_input_batch)
            if bl_repeat_num == 0:
                modified_input = torch.unsqueeze(modified_input, 0)
            modified_input = modified_input.to(self.device)
            pred = self.model.forward(modified_input).cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)
            preds.append(pred) # 2D list. row: quantile steps. col: 0: modified input pred, rest: randomized baseline pred
            # if len(preds) ==2:
            #     break # todo
        preds = np.stack(preds) # [quantile_step, bl_num+1]
        # print(preds.shape)


        return preds, quantile_step

    def random_heatmap(self, heatmap, input):
        ''' Obsolete function
        Generate random heatmap by random ordering O (following: https://arxiv.org/pdf/1509.06321.pdf)
        while keeping the location the same. Therefore, the random baseline and original heatmap should have the same
        performance at init and last removal step.
        :return:
        '''
        rand_hm = np.zeros(shape = heatmap.shape)
        heatmap_mask = np.logical_and( (heatmap > 0), input != 0)
        for index in np.unique(heatmap_mask):
            mask = heatmap_mask == index
            if index == 1:
                rand_hm[mask] = np.random.permutation(heatmap[mask])#.flatten())
        print('random hm', rand_hm.min(), rand_hm.max())
        print((rand_hm == heatmap).sum() / heatmap.size )
        return rand_hm

    def acc_drop(self, save_dir, hm_normalize = True, get_rid_neg = True, bl_repeat_num = 15):
        '''
        Heatmap eval experiment:
        Occlude heatmap mask and see how quickly the acc curve drops
        :param dataloader:
        :param model:
        :param steps:
        :param method_list:
        :param get_baseline: draw a baseline by randomizing the same number of features of a method, and ablate it. repeat each xai method n times, and get acc_drop mean, and std.
        :return:
        drop_dict = {method: [acc drop for each steps]}
        auc_list = [roc for each method]
        '''
        data_ids, df = get_data_ids(save_dir)
        self.logger.info('Load {} datapoints for Acc Drop Experiment'.format(len(data_ids)))

        # prep variables to save results
        # pass the data to model once, get the original heatmaps
        # heatmap_dict[bratsID] = (heatmap, pred, labels_val, time_record, input)
        # heatmap_dict = generate_heatmap_dataset(self.valid_data_loader, self.model,  **self.gnt_hm_ds_args) #, exp_condition='acc_drop', save_input = True,
        # save the original heatmaps
        # self.save_experiment_results(heatmap_dict, folder_name = 'heamaps_for_trained_model_BRATS_IDH')

        # add original accuracy to drop_dict
        # pred_b = [heatmap_dict[i][1] for i in heatmap_dict]
        # gt     = [heatmap_dict[i][2] for i in heatmap_dict]
        # acc = accuracy_score(gt, pred_b)

        # init the record
        # csv_filename = Path('/local-scratch/authorid/trained_model/BRATS20_HGG/test') / 'cv_result_fold_{}.csv'.format(self.fold)
        try:
            seed = self.config['seed']
        except:
            seed = None
        if seed is not None:
            if self.config['machine'] == 'cc':
                csv_filename = Path('/scratch/authorid/results_brats_rerun/trained_models/test') / 'test_result_fold_1_seed_{}.csv'.format(self.seed)
            elif self.config['machine'] == 'ts':
                csv_filename = Path('/local-scratch/authorid/brats_rerun_20220502/test') / 'test_result_fold_1_seed_{}.csv'.format(self.seed)

        print('Test performance csv:', csv_filename)
        results = pd.read_csv(csv_filename, names=['dataID', 'gt', 'pred', 'softmax_0', 'softmax_1'])
        gt = results['gt']
        pred = results['pred']
        acc_init = accuracy_score(gt, pred)
        print('init acc', acc_init)
        # acc_init = save_dir_sanity_check(save_dir, self.method_list, self.valid_data_loader)
        # assert acc_init, self.logger.info("Sanity check failed!")
        # drop_dict = {m: [acc_init] for m in self.method_list}

        # get img array
        save_dir = Path(save_dir)
        # input_dir = save_dir / 'input'
        # data_ids = [k.name.strip('.pkl') for k in Path(input_dir).rglob('*.pkl')]
        modified_input_save_root = save_dir / '{}'.format(self.baseline_value_type)
        print('modified_input_save_root', modified_input_save_root, self.baseline_value_type)

        # get gt label
        gts = df[["Data_ID", "GT"]].drop_duplicates()
        gts = gts.set_index("Data_ID")
        gts = gts['GT'].to_dict()

        with torch.no_grad():
            # pertubate the top quantile heatmap ranking each step, and pass the occuluted input to model to get accuracy
            for m in self.method_list:
                modified_input_save_dir = modified_input_save_root / m
                modified_input_save_dir.mkdir(exist_ok=True, parents=True)
                csv_filename = modified_input_save_root / '{}_acc_drop_record.csv'.format(m)
                file_exists = os.path.isfile(csv_filename)
                # drop_dict = {q: [acc_init] for q in quantile_step}
                bl_drop_dict = {0: [acc_init for i in range(bl_repeat_num)]}
                if file_exists:
                    self.logger.info("{} file exists, pass".format(csv_filename))
                    continue
                # for quantile in quantile_step[1:]:
                # pred_result_dict = {i: [] for i in range(bl_repeat_num+1)} # i == 0 is the heatmap one, the rest is baseline
                gt_record = []
                pred_result_alldata = []

                hm_dict, _ = get_heatmaps(save_dir, m, by_data = False, hm_as_array=False)
                if len(hm_dict.keys()) == 0:
                    self.logger.info("{} has no heatmaps {}.\n".format(m, hm_dict))
                    continue
                # for d_id in tqdm(data_ids):
                # use input data from val_data_loader directly
                for idx, data in enumerate(tqdm(self.valid_data_loader)): # note: need to specify bs 1 in command line
                    input_array, target, d_id = data[self.image_key], data['gt'], data['bratsID'][0] #, data['seg']
                    # print(d_id)
                    if d_id not in hm_dict: # GuidedGradCAM, some data hm is all 0s, thus not saved, skip those data
                        self.logger.info("{} do not have hm {}".format(d_id, m))
                        continue
                    else:
                        heatmap = hm_dict[d_id]
                        # visualize input and heatmap for sanity check of correctness
                        # dateTimeObj = datetime.now()
                        # time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
                        # img_name = 'acc_drop_inputandhm_beforeprocess-{}-{}-{}'.format(d_id, m, time_stamp)
                        # vis_dir = image3d_to_gif(save_dir = '/local-scratch/authorid/vis_brats/{}'.format(img_name),
                        #                          mris = heatmap,  keep_png = True)
                        # visualize for debug end

                        # print('heatmap shape before', heatmap.shape) # (4, 128, 128, 128)
                        # print('pre process', heatmap.shape[1:], [int(i) for i in input_array.shape[2:]])
                        heatmap = postprocess_heatmaps(heatmap, img_size = tuple([int(i) for i in input_array.shape[2:]]), no_neg=True, rotate_axis = False)
                        # print('heatmap shape after', heatmap.shape) #  (4, 128, 128, 128)

                        # visualize input and heatmap for sanity check of correctness
                        # dateTimeObj = datetime.now()
                        # time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
                        # img_name = 'acc_drop_inputandhm_afterprocess-{}-{}-{}'.format(d_id, m, time_stamp)
                        # vis_dir = image3d_to_gif(save_dir = '/local-scratch/authorid/vis_brats/{}'.format(img_name),
                        #                          heatmaps = heatmap, mris = np.squeeze(input_array), keep_png = True)
                        # visualize for debug end
                        if heatmap.max() == 0:
                            continue # todo for gradcam
                        # input_array = pickle.load(open(input_dir / '{}.pkl'.format(d_id), "rb"))
                        input_array = input_array.cpu().detach().numpy()
                        self.logger.debug('\n method {}, id {}, image shape {}, heatmap range {}-{}'.format( m, d_id, input_array.shape, heatmap.min(), heatmap.max()))
                        preds, quantile_steps_lst = self.modify_input_predict(np.squeeze(input_array), heatmap, modified_input_save_dir = modified_input_save_dir, data_id = d_id,
                                                                              bl_repeat_num = bl_repeat_num,
                                                                              hm_normalize = False, get_rid_neg = True, ABS=False)

                        # if not(preds and quantile_steps_lst):
                        #     self.logger.info("max {},min {}.".format(heatmap.max(), heatmap.min()))
                        #     continue
                        # self.logger.info('Done {}, preds shape {}'.format(m, preds.shape))
                        # record the pred for each data point
                        pred_result_alldata.append(preds) # [quantile_step, bl_num+1]
                        # get the baseline results
                        gt = gts[d_id]
                        gt_record.append(gt)
                    # if len(pred_result_alldata) == 1:
                    #     break # todo
                # get the new acc
                pred_result_alldata = np.stack(pred_result_alldata)# [datapoints_num, quantile_step-1, bl_num+1]

                # acc_array = []

                # save the acc_array
                random_bl = ['baseline_{}'.format(i) for i in range(1, bl_repeat_num + 1)]
                fnames = [m] + random_bl

                df = pd.DataFrame(columns=fnames, index=quantile_steps_lst)
                # print("pred_result_alldata", pred_result_alldata.shape, quantile_steps_lst)
                # print('init df', df)
                for q, quantile in enumerate(quantile_steps_lst):
                    # acc_quantile = []
                    for i in range(bl_repeat_num+1):
                        if quantile == 0:
                            acc = acc_init
                        else:
                            # print(df)
                            acc = accuracy_score(gt_record, pred_result_alldata[:, q-1, i])
                        df.loc[quantile][fnames[i]] = acc
                        if i == 0:
                            self.logger.info("\nQuantile {}, {}, acc drop from {} to: {}\n".format(quantile, m, acc_init, acc ))
                self.logger.info('{}: {}'.format(m, df))
                df.to_csv(csv_filename)

                # delete saved folder # uncomment for test
                try:
                    shutil.rmtree(modified_input_save_dir)
                except OSError as e:
                    print("Error {}, cannot delete tmp folder {}".format(e, modified_input_save_dir))


                    # acc_array.append(acc_quantile)
                # acc_array = np.array(acc_array)  # acc for [quantile_step, bl_num+1]
                # print(acc_array)

                # df = pd.DataFrame(data = drop_dict, columns = fnames, index_label = 'quantile')
                # with open(csv_filename, 'a', newline='') as csv_file:
                #     csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
                #     if not file_exists:
                #         csv_writer.writeheader()
                #         for key, value in drop_dict.items():
                #             csv_writer.writerow({'quantile': key, '{}_acc_drop'.format(m): value})

        # calculate roc
        # for m in drop_dict:
        #     area = auc(quantile_step, drop_dict[m])
        #     auc_list.append(area)

        # save the results
        # save_dir = '../xaiexp_log/heatmap_exp/acc_drop/'
        # if self.exp_name:
        #     save_dir += self.exp_name
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # drop_dict
        # dateTimeObj = datetime.now()
        # time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        # drop_filename = os.path.join(modified_input_save_root, 'acc_drop_{}.csv'.format(time_stamp))
        # drop_dict['quantile_step'] = quantile_step
        # drop_df = pd.DataFrame(data = drop_dict)
        # drop_df.to_csv(drop_filename, index=False)
        # # auc_list
        # auc_df = pd.DataFrame(data = auc_list, index = self.method_list)
        # auc_filename = os.path.join(modified_input_save_root, 'auc_acc_drop_{}.csv'.format(time_stamp))
        # auc_df.to_csv(auc_filename)

        # return drop_df, auc_list, drop_filename, auc_filename
        return modified_input_save_root

    def plot_acc_drop(self, modified_input_save_root):
        '''
        plot the acc drop line for each xai method
        :param drop_dict:
        :param auc_list:
        :return:
        '''
        sns.set_style("whitegrid")

        auc_record = dict()
        modified_input_save_root = Path(modified_input_save_root)
        # get all subdir
        # method_list = [f.name for f in modified_input_save_root.iterdir() if f.is_dir()]
        # if len(method_list) == 0:
        method_list = [k.name.split('.')[0].split('_')[0] for k in Path(modified_input_save_root).rglob('*_acc_drop_record.csv')]
        print(modified_input_save_root, method_list)
        drop_dict = dict()
        df_lst = []
        # calculate roc
        for m in method_list:
            csv_filename = modified_input_save_root /'{}_acc_drop_record.csv'.format(m)
            csv = pd.read_csv(csv_filename, index_col=0)

            quantile_step = csv.index

            xai_acc_drop = csv[m]
            drop_dict[m] = xai_acc_drop
            #         print(drop_dict)

            # calculate baseline acc drop
            bl_col = [col for col in csv if col.startswith('baseline_')]
            bl_acc_drop = csv[bl_col].mean(axis=1).to_dict()
            bl_acc_drop_std = csv[bl_col].std(axis=1).to_dict()
            drop_dict['{}_baseline'.format(m)] = list(bl_acc_drop.values())
            drop_dict['{}_blstd'.format(m)] = list(bl_acc_drop_std.values())
            # auc
            area = auc(quantile_step, xai_acc_drop)
            auc_record[m] = area
            bl_auc = auc(quantile_step, list(bl_acc_drop.values()))
            auc_record['{}_baseline'.format(m)] = bl_auc

            # modify the col name to be combined
            csv.reset_index(level=0, inplace=True)
            df = csv.rename(columns={'index': 'quantile', m: 'baseline_0'})

            df['xai_label'] = '{}. AUC= {:.2f}. Its baseline AUC ={:.2f}'.format(m, area, bl_auc)
            #         print(df)

            df_long = pd.wide_to_long(df, ['baseline'], i='quantile', j='trialID',
                                      sep='_')  # df.melt('quantile',  var_name='cols',  value_name='vals')
            df_long = df_long.rename(columns={'baseline': 'accuracy'})
            df_long.reset_index(inplace=True)

            df_long['mORb'] = 'baseline'
            df_long.loc[df_long['trialID'] == 0, 'mORb'] = 'xaiMethod'
            #         print(df_long)

            df_lst.append(df_long)

        # combine all csvs
        methods_sorted_by_auc = dict(sorted(auc_record.items(), reverse=True, key=lambda item: item[1]))

        aggregate_df = pd.concat(df_lst)
        aggregate_df.to_csv(modified_input_save_root/'aggregate_df_{}.csv'.format(datetime.now().strftime("%Y%m%d_%H%M")))
        # put xai method in the sequence of pertubation /gradient based method. Not working, may need to reorder df
        name_dct = {}
        for idx, n in enumerate(aggregate_df['xai_label'].unique()):
            name = n.split('.')[0].lower()
            name_dct[str(name)] = n
        row_order = []
        for p in self.perturbation_method:
            p = p.lower()
            if p in name_dct:
                row_order.append(name_dct[p])
        for p in self.gradient_method:
            p = p.lower()
            if p in name_dct:
                row_order.append(name_dct[p])
        # print(row_order)
        # plot
        g1 = sns.relplot(
            data=aggregate_df, x='quantile', y='accuracy', style='mORb', hue='xai_label', markers=True,
            kind='line', col='xai_label', col_wrap=3, row_order = row_order
        )
        g1.set_titles("{col_name}")
        #     plt.xlim(0, None)

        #     plt.clf()
        #     g = sns.lineplot(data = aggregate_df,x = 'quantile', y='accuracy', style = 'mORb', hue = 'xai_label', markers=True, )
        #     sns.catplot(x = 'quantile', y='accuracy', col='xai_label', col_wrap = 4, kind='point',
        #                data = aggregate_df)

        # prev vis
        drop_dict['quantile'] = quantile_step
        drop_df = pd.DataFrame(data=drop_dict)
        #     print(drop_df)
        df = drop_df.melt('quantile', var_name='cols', value_name='vals')
        #     print(df)
        # print(auc_record)
        df = df.rename(columns={'quantile': 'Input perturbation percentage', 'cols': 'XAI method', 'vals': 'Accuracy'})
        g = sns.catplot(x='Input perturbation percentage', y="Accuracy", hue='XAI method',
                        hue_order=methods_sorted_by_auc.keys(), ci='sd', legend_out=True, data=df, kind='point')
        # customize legend to show AUC
        labels = ["{}, AUC = {:.2f}".format(k, v) for k, v in methods_sorted_by_auc.items()]
        for ax in g.axes.flat:
            leg = g.axes.flat[0].get_legend()
            if not leg is None: break
        # or legend may be on a figure
        if leg is None:
            leg = g._legend
        for t, l in zip(leg.texts, labels):
            t.set_text(l)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # g.ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

        print(methods_sorted_by_auc.keys(), methods_sorted_by_auc.values())

        # save figure
        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        fig_name = 'fig_acc_drop_{}.pdf'.format(time_stamp)
        g1.savefig(modified_input_save_root / fig_name)
        return aggregate_df, g1

    def pipeline(self, save_dir, **kwargs):
        modified_input_save_root = self.acc_drop(save_dir, **kwargs)
        # aggregate_df, graph = self.plot_acc_drop(modified_input_save_root)
        # return aggregate_df, graph, modified_input_save_root
    # def plot_acc_drop(self, modified_input_save_root):
    #     '''
    #     plot the acc drop line for each xai method
    #     :param drop_dict:
    #     :param auc_list:
    #     :return:
    #     '''
    #     auc_record = dict()
    #     modified_input_save_root = Path(modified_input_save_root)
    #     # get all subdir
    #     method_list = [f.name for f in modified_input_save_root.iterdir() if f.is_dir()]
    #     print(method_list)
    #     drop_dict = dict()
    #     # calculate roc
    #     for m in method_list:
    #         csv_filename = modified_input_save_root / m/ 'acc_drop_record_{}.csv'.format(m)
    #         csv = pd.read_csv(csv_filename)
    #         csv = csv.sort_values('quantile')
    #         quantile_step = csv['quantile']
    #         acc_drop = csv['acc_drop']
    #         drop_dict[m] = acc_drop
    #         area = auc(quantile_step, acc_drop)
    #         auc_record[m] = area
    #         # csv = csv.set_index('quantile')
    #         # csv = csv.rename(columns={'acc_drop': m})
    #         # dfs.append(csv)
    #     # if (type(drop_df) is str) and (type(auc_list)  is str ):
    #     #     p = Path(drop_df)
    #     #     drop_df = pd.read_csv(drop_df, index_col= False)
    #     #     auc_list = pd.read_csv(auc_list)
    #     #     save_path = p.parent
    #     #     print(drop_df, save_path)
    #     # drop_df = pd.concat(dfs, join='inner')
    #     drop_dict['quantile'] = quantile_step
    #     drop_df = pd.DataFrame(data=drop_dict)
    #     print(drop_df)
    #     df = drop_df.melt('quantile', var_name='cols',  value_name='vals')
    #     print(df)
    #     # print(auc_record)
    #     methods_sorted_by_auc = dict(sorted(auc_record.items(), reverse = True, key=lambda item: item[1]))
    #     df = df.rename(columns={'quantile': 'Input perturbation percentage', 'cols': 'XAI method', 'vals': 'Accuracy'})
    #     g = sns.catplot(x='Input perturbation percentage', y="Accuracy", hue='XAI method', hue_order = methods_sorted_by_auc.keys(), ci = 'sd', legend_out= True, data=df, kind='point')
    #     # customize legend to show AUC
    #     labels = ["{}, AUC = {:.2f}".format(k, v) for k,v in methods_sorted_by_auc.items()]
    #     for ax in g.axes.flat:
    #         leg = g.axes.flat[0].get_legend()
    #         if not leg is None: break
    #     # or legend may be on a figure
    #     if leg is None:
    #         leg = g._legend
    #     for t, l in zip(leg.texts, labels):
    #         t.set_text(l)
    #     # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #     # g.ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    #
    #     print(methods_sorted_by_auc.keys(), methods_sorted_by_auc.values())
    #     # save figure
    #     dateTimeObj = datetime.now()
    #     time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
    #     fig_name = 'fig_acc_drop_{}.pdf'.format(time_stamp)
    #     g.savefig(modified_input_save_root/fig_name)
    #     return g

    # def plot_acc_drop(self, drop_df, auc_list):
    #     '''
    #     plot the acc drop line for each xai method
    #     :param drop_dict:
    #     :param auc_list:
    #     :return:
    #     '''
    #     save_path = None
    #     if (type(drop_df)  is  str) and (type(auc_list)  is str ):
    #         p = Path(drop_df)
    #         drop_df = pd.read_csv(drop_df, index_col= False)
    #         auc_list = pd.read_csv(auc_list)
    #         save_path = p.parent
    #         print(drop_df, save_path)
    #     df = drop_df.melt('quantile_step', var_name='cols',  value_name='vals')
    #     df = df.rename(columns={'quantile_step': 'Input perturbation percentage', 'cols': 'XAI method', 'vals': 'Accuracy'})
    #     g = sns.catplot(x='Input perturbation percentage', y="Accuracy", hue='XAI method', data=df, kind='point')
    #     if save_path:
    #         dateTimeObj = datetime.now()
    #         time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
    #         fig_name = 'fig_acc_drop_{}.pdf'.format(time_stamp)
    #         g.savefig(save_path/fig_name)
    #     return g


    # def modify_input(self, input, heatmaps, cutoff, hm_normalize = True, get_rid_neg = True, ABS=False, bl_repeat_num = 0):
    #     '''
    #     Utility func for acc_drop.
    #     :param input: one input image of np array
    #     :param heatmaps: a heatmap mask to pertube the input area.
    #     :param hm_normalize: whether to normalize a heatmap. Normalize to [-1, 1] using normalize_scale()
    #     :return: modified_input_lst: a modified_input of the same size. 0 is modified according to the heatmap, while the rest is random ablated baseline
    #     with the same num of important features, and ablated and filled use the same unimportant feature value.
    #     '''
    #     assert input.shape == heatmaps.shape, 'input {} and heatmap {} have different shape'.format(input.shape, heatmaps.shape)
    #     if hm_normalize:
    #         heatmaps = normalize_scale(heatmaps)
    #         # heatmaps = scale_hms(heatmaps)
    #     if ABS:
    #         heatmaps = np.absolute(heatmaps)
    #     if get_rid_neg:
    #         # TODO?? set negative regions to 0
    #         heatmaps[heatmaps<0] = 0
    #     quantile = np.quantile(heatmaps[np.nonzero(heatmaps)], 1.0-cutoff)
    #     assert quantile != heatmaps.min(), print('Error! quantile == hm.min()')
    #     unimportant_size = 0
    #     alpha = 0.04
    #     # get signal from unimportant regions
    #     while unimportant_size <= 0:
    #         alpha += 0.01
    #         assert  alpha < 1.0, print('Error on identifying unimportant regions!')
    #         lower_quantile = np.quantile(heatmaps, alpha) # get the lower 5% quantile values of heatmap (including 0) as unimportant
    #         unimportant = input[heatmaps <= lower_quantile]
    #         # noises = np.std(unimportant)
    #         unimportant_size = unimportant.size
    #
    #         # assert unimportant.size > 0, print('Error! No unimportant region defined!')
    #     mask_size = 0
    #     for i in range(bl_repeat_num+1):
    #         if i ==0:
    #             mask = heatmaps >= quantile
    #             mask_size = mask.sum()
    #             print('mask sum', mask.sum(), 'unimportant size', unimportant.shape, alpha, lower_quantile, heatmaps.min(), heatmaps.max())
    #         else:
    #             mask = get_baseline_random_mask(mask_size, input.shape)#todo
    #         # noise_signal = np.random.normal(scale = noises, size = input.shape)
    #         # unimportant_noise_image = (np.random.choice(unimportant.flatten(), size = input.size)).reshape(input.shape)
    #         # unimportant_noise_image += noise_signal
    #         unimportant_image = np.random.choice(unimportant, size=input.shape, replace=True)
    #         # modified_input = input.copy()
    #         modified_input = np.where(mask, unimportant_image, input)
    #         modified_input_lst.append(modified_input)
    #     return modified_input_lst  #TODO visualize the results
    #
    #
    #
    # def acc_drop(self, save_dir, hm_normalize = True, get_rid_neg = True, bl_repeat_num = 15):
    #     '''
    #     Heatmap eval experiment:
    #     Occlude heatmap mask and see how quickly the acc curve drops
    #     :param dataloader:
    #     :param model:
    #     :param steps:
    #     :param method_list:
    #     :param get_baseline: draw a baseline by randomizing the same number of features of a method, and ablate it. repeat each xai method n times, and get acc_drop mean, and std.
    #     :return:
    #     drop_dict = {method: [acc drop for each steps]}
    #     auc_list = [roc for each method]
    #     '''
    #     data_ids, df = get_data_ids(save_dir)
    #     self.logger.info('Load {} datapoints for Acc Drop Experiment'.format(len(data_ids)))
    #
    #     # prep variables to save results
    #
    #     quantile_step = [0]
    #     interval = 1.0/self.steps
    #     accumuate = 0.0
    #     for i in range(self.steps): # accumulated pertubation precentage
    #         accumuate += interval
    #         quantile_step.append(float(format(accumuate, '.5f')))
    #     self.logger.info('quantile steps to delete input: {}'.format(quantile_step))
    #
    #     # pass the data to model once, get the original heatmaps
    #     # heatmap_dict[bratsID] = (heatmap, pred, labels_val, time_record, input)
    #     # heatmap_dict = generate_heatmap_dataset(self.valid_data_loader, self.model,  **self.gnt_hm_ds_args) #, exp_condition='acc_drop', save_input = True,
    #     # save the original heatmaps
    #     # self.save_experiment_results(heatmap_dict, folder_name = 'heamaps_for_trained_model_BRATS_IDH')
    #
    #     # add original accuracy to drop_dict
    #     # pred_b = [heatmap_dict[i][1] for i in heatmap_dict]
    #     # gt     = [heatmap_dict[i][2] for i in heatmap_dict]
    #     # acc = accuracy_score(gt, pred_b)
    #
    #     # init the record
    #     acc_init = save_dir_sanity_check(save_dir, self.method_list, self.valid_data_loader)
    #     assert acc_init, self.logger.info("Sanity check failed!")
    #     # drop_dict = {m: [acc_init] for m in self.method_list}
    #
    #     # get img array
    #     save_dir = Path(save_dir)
    #     input_dir = save_dir / 'input'
    #     # data_ids = [k.name.strip('.pkl') for k in Path(input_dir).rglob('*.pkl')]
    #     modified_input_save_root = save_dir / 'acc_drop'
    #
    #     # get gt label
    #     gts = df[["Data_ID", "GT"]].drop_duplicates()
    #     gts = gts.set_index("Data_ID")
    #     gts = gts['GT'].to_dict()
    #
    #     with torch.no_grad():
    #         # pertubate the top quantile heatmap ranking each step, and pass the occuluted input to model to get accuracy
    #         for m in self.method_list:
    #             modified_input_save_dir = modified_input_save_root / m
    #             modified_input_save_dir.mkdir(exist_ok=True, parents=True)
    #             csv_filename = modified_input_save_dir / '{}_acc_drop_record.csv'.format(m)
    #             file_exists = os.path.isfile(csv_filename)
    #             drop_dict = {q: [acc_init] for q in quantile_step}
    #             bl_drop_dict = {0: [acc_init for i in range(bl_repeat_num)]}
    #             if file_exists:
    #                 self.logger.info("{} file exists, pass".format(csv_filename))
    #                 continue
    #             for quantile in quantile_step[1:]:
    #                 pred_result_dict = {i: [] for i in range(bl_repeat_num+1)} # i == 0 is the heatmap one, the rest is baseline
    #                 gt_record = []
    #                 hm_dict, _ = get_heatmaps(save_dir, m, by_data = False, hm_as_array=False)
    #                 for d_id in data_ids:
    #                     heatmap = hm_dict[d_id]
    #                     input_array = pickle.load(open(input_dir / '{}.pkl'.format(d_id), "rb"))
    #                     input_array = input_array.numpy()
    #                     self.logger.debug('quantile {}, method {}, id {}, image shape {}'.format(quantile, m, d_id, input_array.shape))
    #                     modified_input_lst = self.modify_input(np.squeeze(input_array), heatmap, quantile, hm_normalize=hm_normalize, get_rid_neg = get_rid_neg, bl_repeat_num = bl_repeat_num)
    #                     for i, modified_input in enumerate(modified_input_lst):
    #                         if i == 0:
    #                             # save the modified_input
    #                             modified_input_fn = "{}-{}.pkl".format(d_id, str(quantile))
    #                             pickle.dump(modified_input, open(os.path.join(modified_input_save_dir, modified_input_fn), 'wb'))
    #                         # pass the modified input to model
    #                         modified_input = torch.as_tensor(modified_input)
    #                         modified_input = torch.unsqueeze(modified_input, 0).to(self.device)
    #                         pred = self.model.forward(modified_input).cpu().detach().numpy()
    #                         pred = int(np.argmax(pred))
    #                         # record the pred for each data point
    #                         pred_result_dict[i].append(pred)
    #                     # get the baseline results
    #                     gt = gts[d_id]
    #                     gt_record.append(gt)
    #                 for i, pred_result in pred_result_dict.items():
    #                     # get the new acc
    #                     acc = accuracy_score(gt_record, pred_result)
    #                     if i == 0:
    #                         self.logger.info("\nQuantile {}, {}, acc drop from {} to: {}\n".format(quantile, m, acc_init, acc ))
    #                     drop_dict[quantile].append(acc)
    #             # save the drop_dict
    #             random_bl = ['baseline_{}'.format(i) for i in range(1, bl_repeat_num+1)]
    #             fnames = ['quantile', m] + random_bl
    #             df = pd.Dataframe(data = drop_dict, columns = fnames, index_label = 'quantile')
    #             df.to_csv(csv_filename)
    #             # with open(csv_filename, 'a', newline='') as csv_file:
    #             #     csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
    #             #     if not file_exists:
    #             #         csv_writer.writeheader()
    #             #         for key, value in drop_dict.items():
    #             #             csv_writer.writerow({'quantile': key, '{}_acc_drop'.format(m): value})
    #
    #     # calculate roc
    #     # for m in drop_dict:
    #     #     area = auc(quantile_step, drop_dict[m])
    #     #     auc_list.append(area)
    #
    #     # save the results
    #     # save_dir = '../xaiexp_log/heatmap_exp/acc_drop/'
    #     # if self.exp_name:
    #     #     save_dir += self.exp_name
    #     # if not os.path.exists(save_dir):
    #     #     os.makedirs(save_dir)
    #     # drop_dict
    #     # dateTimeObj = datetime.now()
    #     # time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
    #     # drop_filename = os.path.join(modified_input_save_root, 'acc_drop_{}.csv'.format(time_stamp))
    #     # drop_dict['quantile_step'] = quantile_step
    #     # drop_df = pd.DataFrame(data = drop_dict)
    #     # drop_df.to_csv(drop_filename, index=False)
    #     # # auc_list
    #     # auc_df = pd.DataFrame(data = auc_list, index = self.method_list)
    #     # auc_filename = os.path.join(modified_input_save_root, 'auc_acc_drop_{}.csv'.format(time_stamp))
    #     # auc_df.to_csv(auc_filename)
    #
    #     # return drop_df, auc_list, drop_filename, auc_filename
    #     return modified_input_save_root
    def acc_drop_mrnet(self, save_dir, hm_normalize = True, get_rid_neg = True, bl_repeat_num = 15, model = None, valid_data_loader=None, test_record_csv = None):
        '''
        Heatmap eval experiment:
        Occlude heatmap mask and see how quickly the acc curve drops
        :param dataloader:
        :param model:
        :param steps:
        :param method_list:
        :param get_baseline: draw a baseline by randomizing the same number of features of a method, and ablate it. repeat each xai method n times, and get acc_drop mean, and std.
        :return:
        drop_dict = {method: [acc drop for each steps]}
        auc_list = [roc for each method]
        '''
        self.valid_data_loader = valid_data_loader
        self.model = model
        data_ids, df = get_data_ids(save_dir)
        self.logger.info('Load {} datapoints for Acc Drop Experiment'.format(len(data_ids)))

        # prep variables to save results
        # pass the data to model once, get the original heatmaps
        # heatmap_dict[bratsID] = (heatmap, pred, labels_val, time_record, input)
        # heatmap_dict = generate_heatmap_dataset(self.valid_data_loader, self.model,  **self.gnt_hm_ds_args) #, exp_condition='acc_drop', save_input = True,
        # save the original heatmaps
        # self.save_experiment_results(heatmap_dict, folder_name = 'heamaps_for_trained_model_BRATS_IDH')

        # add original accuracy to drop_dict
        # pred_b = [heatmap_dict[i][1] for i in heatmap_dict]
        # gt     = [heatmap_dict[i][2] for i in heatmap_dict]
        # acc = accuracy_score(gt, pred_b)

        # init the record
        # csv_filename = Path('/local-scratch/authorid/trained_model/BRATS20_HGG/test') / 'cv_result_fold_{}.csv'.format(self.fold)
        results = pd.read_csv(test_record_csv, names=['dataID', 'gt', 'pred'], header=0)
        fpr, tpr, threshold = roc_curve(results['gt'].to_list(), results['pred'].to_list())
        auc_init = auc(fpr, tpr)
        print('init acc', auc_init)
        # acc_init = save_dir_sanity_check(save_dir, self.method_list, self.valid_data_loader)
        # assert acc_init, self.logger.info("Sanity check failed!")
        # drop_dict = {m: [acc_init] for m in self.method_list}

        # get img array
        save_dir = Path(save_dir)
        # input_dir = save_dir / 'input'
        # data_ids = [k.name.strip('.pkl') for k in Path(input_dir).rglob('*.pkl')]
        modified_input_save_root = save_dir /  '{}'.format(self.baseline_value_type)

        # get gt label
        gts = df[["Data_ID", "GT"]].drop_duplicates()
        gts = gts.set_index("Data_ID")
        gts = gts['GT'].to_dict()

        with torch.no_grad():
            # pertubate the top quantile heatmap ranking each step, and pass the occuluted input to model to get accuracy
            for m in self.method_list:
                modified_input_save_dir = modified_input_save_root / m
                modified_input_save_dir.mkdir(exist_ok=True, parents=True)
                csv_filename = modified_input_save_root / '{}_acc_drop_record.csv'.format(m)
                file_exists = os.path.isfile(csv_filename)
                # drop_dict = {q: [acc_init] for q in quantile_step}
                bl_drop_dict = {0: [auc_init for i in range(bl_repeat_num)]}
                if file_exists:
                    self.logger.info("{} file exists, pass".format(csv_filename))
                    continue
                # for quantile in quantile_step[1:]:
                # pred_result_dict = {i: [] for i in range(bl_repeat_num+1)} # i == 0 is the heatmap one, the rest is baseline
                gt_record = []
                pred_result_alldata = []

                hm_dict, _ = get_heatmaps(save_dir, m, by_data = False, hm_as_array=False)
                print(sorted(hm_dict.keys()))

                # print('hm_dict !!!', hm_dict.keys())
                if len(hm_dict.keys()) == 0:
                    self.logger.info("{} has no heatmaps {}.\n".format(m, hm_dict))
                    continue
                # for d_id in tqdm(data_ids):
                # use input data from val_data_loader directly
                for idx, data in enumerate(tqdm(self.valid_data_loader)): # note: need to specify bs 1 in command line
                    # input_array, target, d_id = data[self.image_key], data['gt'], data['bratsID'][0]
                    input_array = data[0], data[1], data[2]
                    target, d_id = data[3], data[4][0]
                    # print(d_id, type(d_id), 'd_id') # 1130 <class 'str'> d_id
                    if d_id not in hm_dict: # GuidedGradCAM, some data hm is all 0s, thus not saved, skip those data
                        self.logger.info("{} do not have hm {}".format(d_id, m))
                        continue
                    else:
                        heatmap = hm_dict[d_id]
                        # print('\n pre process heatmap shape', len(heatmap), heatmap[0].shape) # 3 (25, 3, 224, 224)
                        # print([input_array[i].shape for i in range(len(input_array))]) # [torch.Size([1, 25, 3, 224, 224]), torch.Size([1, 27, 3, 224, 224]), torch.Size([1, 25, 3, 224, 224])]
                        heatmap = postprocess_heatmaps_mrnet(heatmap, no_neg=True, depth_first = False)
                        # print('\n postprocess hm shape', heatmap[0].shape) #  (224, 224, 25)
                        if heatmap[0].max() == 0 and heatmap[1].max() == 0 and heatmap[2].max() == 0:
                            continue # todo for gradcam
                        # input_array = pickle.load(open(input_dir / '{}.pkl'.format(d_id), "rb"))
                        np_input = []
                        for img in input_array:
                            np_img = img.cpu().detach().numpy()
                            np_input.append(np_img)
                        self.logger.debug('\nmethod {}, id {}, image shape {}'.format( m, d_id, np_input[0].shape))
                        preds, quantile_steps_lst = self.modify_input_predict_mrnet(np_input, heatmap, modified_input_save_dir = modified_input_save_dir, data_id = d_id,
                                                                              bl_repeat_num = bl_repeat_num)

                        # if not(preds and quantile_steps_lst):
                        #     self.logger.info("max {},min {}.".format(heatmap.max(), heatmap.min()))
                        #     continue
                        # self.logger.info('Done {}, preds shape {}'.format(m, preds.shape))
                        # record the pred for each data point
                        pred_result_alldata.append(preds) # [quantile_step, bl_num+1]
                        # get the baseline results
                        gt = gts[int(d_id)]
                        gt_record.append(gt)
                    # if len(pred_result_alldata) == 3:
                    #     break # todo
                # get the new acc
                pred_result_alldata = np.stack(pred_result_alldata)# [datapoints_num, quantile_step-1, bl_num+1]

                # acc_array = []

                # save the acc_array
                random_bl = ['baseline_{}'.format(i) for i in range(1, bl_repeat_num + 1)]
                fnames = [m] + random_bl

                df = pd.DataFrame(columns=fnames, index=quantile_steps_lst)
                # print("pred_result_alldata", pred_result_alldata.shape, quantile_steps_lst)
                # print('init df', df)
                for q, quantile in enumerate(quantile_steps_lst):
                    # acc_quantile = []
                    for i in range(bl_repeat_num+1):
                        if quantile == 0:
                            acc = auc_init
                        else:
                            # print(df)
                            # print('gt_record', len(gt_record), pred_result_alldata.shape, pred_result_alldata[:, q-1, i].shape) # gt_record 3 (3, 10, 16) (3,)
                            # print(gt_record, pred_result_alldata[:, q-1, i])
                            fpr, tpr, threshold = roc_curve(gt_record, pred_result_alldata[:, q-1, i])
                            # print('fpr', fpr, tpr)
                            acc = auc(fpr, tpr)
                            # acc = accuracy_score(gt_record, pred_result_alldata[:, q-1, i])
                        df.loc[quantile][fnames[i]] = acc
                        if i == 0:
                            self.logger.info("\nQuantile {}, {}, acc drop from {} to: {}\n".format(quantile, m, auc_init, acc ))
                self.logger.info('{}: {}'.format(m, df))
                df.to_csv(csv_filename)

                # delete saved folder # todo comment for test
                try:
                    shutil.rmtree(modified_input_save_dir)
                except OSError as e:
                    print("Error {}, cannot delete tmp folder {}".format(e, modified_input_save_dir))


                    # acc_array.append(acc_quantile)
                # acc_array = np.array(acc_array)  # acc for [quantile_step, bl_num+1]
                # print(acc_array)

                # df = pd.DataFrame(data = drop_dict, columns = fnames, index_label = 'quantile')
                # with open(csv_filename, 'a', newline='') as csv_file:
                #     csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
                #     if not file_exists:
                #         csv_writer.writeheader()
                #         for key, value in drop_dict.items():
                #             csv_writer.writerow({'quantile': key, '{}_acc_drop'.format(m): value})

        # calculate roc
        # for m in drop_dict:
        #     area = auc(quantile_step, drop_dict[m])
        #     auc_list.append(area)

        # save the results
        # save_dir = '../xaiexp_log/heatmap_exp/acc_drop/'
        # if self.exp_name:
        #     save_dir += self.exp_name
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # drop_dict
        # dateTimeObj = datetime.now()
        # time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        # drop_filename = os.path.join(modified_input_save_root, 'acc_drop_{}.csv'.format(time_stamp))
        # drop_dict['quantile_step'] = quantile_step
        # drop_df = pd.DataFrame(data = drop_dict)
        # drop_df.to_csv(drop_filename, index=False)
        # # auc_list
        # auc_df = pd.DataFrame(data = auc_list, index = self.method_list)
        # auc_filename = os.path.join(modified_input_save_root, 'auc_acc_drop_{}.csv'.format(time_stamp))
        # auc_df.to_csv(auc_filename)

        # return drop_df, auc_list, drop_filename, auc_filename
        return modified_input_save_root

    def modify_input_predict_mrnet(self, input, heatmaps, data_id, modified_input_save_dir, bl_repeat_num = 0):
        '''
        Utility func for acc_drop. To reduce putting the generated modified_inputs in memory, generate a batch for all quantile stpes,
        and pass the full batch to predict, and save the input and prediction. Then iterate to the next random baseline input.
        :param input: one input image of np array
        :param heatmaps: a heatmap mask to pertube the input area.
        :param hm_normalize: whether to normalize a heatmap. Normalize to [-1, 1] using normalize_scale()
        :return: modified_input_lst: a modified_input of the same size. 0 is modified according to the heatmap, while the rest is random ablated baseline
        with the same num of important features, and ablated and filled use the same unimportant feature value.
        '''
        # print(len(input[0]), len(input), len(heatmaps[0]),len(heatmaps), 'shape ')  # 1 3 25 3
        aligned_inputs = [] # list of modality inputs, each of (224, 224, depth)
        modality_mean = []
        for ipt in input:
            ipt3D = np.squeeze(ipt).mean(axis = 1).transpose(1,2,0) # (1, 25, 3, 224, 224) to (224, 224, 25)
            aligned_inputs.append(ipt3D)
            modality_mean.append(ipt3D.mean())
            # print(ipt3D.min(), ipt3D.max(), ipt3D.mean(), 'input min max')
        # print(heatmaps[0].shape, heatmaps[1].shape, heatmaps[2].shape, 'stacked_hm shape') # (224, 224, 25) (224, 224, 27) (224, 224, 25)
        # print(heatmaps[0].shape, stacked_hms.shape) # (224, 224, 25) (224, 224, 77)
        assert aligned_inputs[0].shape == heatmaps[0].shape, 'input {} and heatmap {} have different shape'.format(input.shape, heatmaps.shape)
        # if hm_normalize:
        #     heatmaps = normalize_scale(heatmaps)
        #     # heatmaps = scale_hms(heatmaps)
        # if ABS:
        #     heatmaps = np.absolute(heatmaps)
        # if get_rid_neg:
        #     # TODO?? set negative regions to 0
        #     heatmaps[heatmaps<0] = 0

        # test to visualize that the background is not 0s, so for acc_drop_bg use modality mean instead
        # pickle.dump(aligned_inputs,
        #             open(os.path.join('/local-scratch/authorid/brats_rerun_20220502/visualize/mrnet', '{}.pkl'.format(data_id)),
        #                  'wb'))
        # return

        # prepare quantile/cutoff steps
        quantile_step = [0]
        interval = 1.0/self.steps
        accumuate = 0.0
        for i in range(self.steps): # accumulated pertubation precentage
            accumuate += interval
            quantile_step.append(float(format(accumuate, '.5f')))
        # self.logger.debug('quantile steps to delete input: {}'.format(quantile_step))

        # for each ablation step, get
        # prev_bl_input = np.zeros(input.shape) # save the random baseline for prev steps, and ablate on top of it.
        preds = []

        # permute the oringial heatmap to create a bl_repeat_num of random heatmaps as random baseline
        heatmaps_and_random_bl = [heatmaps]
        for rpt in range(bl_repeat_num):
            # random_hm = np.random.permutation(heatmaps.flatten()).reshape(heatmaps.shape)
            # random_hm = self.random_heatmap(heatmaps, input)
            random_hm_lst = []
            for mod_hm in heatmaps:
                random_hm = np.random.permutation(mod_hm.flatten()).reshape(mod_hm.shape)
                random_hm_lst.append(random_hm)
            heatmaps_and_random_bl.append(random_hm_lst)

        quantile_dict = {}
        for q_idx, cutoff in enumerate(quantile_step[1:]):
            modified_input_batch = []
            for idx, hm in enumerate(heatmaps_and_random_bl):
                stacked_hms = np.dstack(hm)
                # construct modified input
                if q_idx not in quantile_dict:  # save the quantile so that random baseline has the same quantile to ablate
                    quantile = np.quantile(stacked_hms[np.nonzero(stacked_hms)], 1.0 - cutoff)
                    quantile_dict[q_idx] = quantile
                else:
                    quantile = quantile_dict[q_idx]
                # if quantile == heatmaps.min():
                #     return None, None
                # if q_idx < len(quantile_step[1:])-1: # at last step, quantile can be equal to hm.min()
                #     assert quantile != heatmaps.min(), print('Error! quantile == hm.min() at step {}, cutoff value {}'.format(q_idx, cutoff)) #  todo remove the requirement for GradCAM


                # get a list of mask for each modality
                modality_mask_lst = []
                for k in range(len(input)):
                    mask = hm[k] >= quantile
                    modality_mask_lst.append(mask)

                # for the following each acc_drop study, construct the modified_input_list
                modified_input_list = []
                if self.baseline_value_type == 'acc_drop_nb' and idx == 0:
                    unimportant_image = self.neighbor_baseline_value(aligned_inputs, modality_mask_lst, zero_background=False)  # fill in value for the mask
                    for nb_m in range(len(input)):
                        mask = modality_mask_lst[nb_m]
                        modified_input = np.where(mask, unimportant_image[nb_m], aligned_inputs[nb_m])
                        modified_input_list.append(self._img_3d_to_5d(modified_input))
                else:
                    if self.baseline_value_type == 'acc_drop':

                        unimportant_size = 0
                        unimportant_list = []
                        alpha = 0.04
                        # get signal from unimportant regions
                        while unimportant_size <= 0:
                            alpha += 0.01
                            assert  alpha < 1.0, print('Error on identifying unimportant regions!')
                            lower_quantile = np.quantile(stacked_hms, alpha) # get the lower 5% quantile values of heatmap (including 0) as unimportant

                            for i in range(len(input)):
                                unimportant = aligned_inputs[i][hm[i] <= lower_quantile]
                                unimportant_list.append(unimportant)
                                unimportant_size += unimportant.size
                        # unimportant_image_list = []
                        # mask_list = []

                        for i in range(len(input)):
                            if unimportant_list[i].shape[0] <=10:
                                unimportant_image = np.zeros(shape = aligned_inputs[i].shape).astype(aligned_inputs[i].dtype)
                                # print(unimportant_image.shape, '0 uni', unimportant_image[i].dtype, aligned_inputs[i].dtype)
                            else:
                                # used to ablate both input and random baseline
                                unimportant_image = np.random.choice(unimportant_list[i], size=aligned_inputs[i].shape,
                                                                 replace=True)
                                # print(unimportant_image.shape, 'not 0 uni', unimportant_image[i].dtype)
                            # get the hm mask
                            # mask = heatmaps[i] >= quantile
                            # mask_list.append(mask)
                            # mask_size = mask.sum()
                            # print('mask sum', mask.sum(), 'unimportant size', unimportant.shape, alpha, lower_quantile, heatmaps.min(), heatmaps.max())
                            modified_input = np.where(modality_mask_lst[i], unimportant_image, aligned_inputs[i])
                            modified_input_list.append(self._img_3d_to_5d(modified_input))
                            # print(modified_input_list[0].shape, 'modified_input_list') # (1, 25, 3, 224, 224) modified_input_list
                            # unimportant_image_list.append(unimportant_image)
                    elif (self.baseline_value_type == 'acc_drop_bg') or (self.baseline_value_type == 'acc_drop_nb' and idx != 0):
                        # Background value is not 0, so acc_drop_bg use modality mean instead
                        for md in range(len(input)):
                            mask = modality_mask_lst[md]
                            unimportant_image = np.ones(shape=aligned_inputs[md].shape, dtype=np.float32) * modality_mean[md]
                            modified_input = np.where(mask, unimportant_image, aligned_inputs[md])
                            modified_input_list.append(self._img_3d_to_5d(modified_input))

                # save the modified_input
                if idx == 0:
                    modified_input_fn = "{}-{}.pkl".format(data_id, str(cutoff))
                else:
                    modified_input_fn = "{}-bl_{}-{}.pkl".format(data_id, idx, str(cutoff))
                pickle.dump(modified_input_list,
                            open(os.path.join(modified_input_save_dir, modified_input_fn),
                                 'wb'))  # TODO visualize the results

                modified_input_batch.append(modified_input_list)

                # --- obsolete function ---
                # random_bl_input_repeat = []
                # # get randomized mask
                # for i in range(bl_repeat_num):
                #     random_bl_input_list = []
                #     for j in range(len(input)):
                #         mask = mask_list[j]
                #         # print('mask', mask.shape, aligned_inputs[j].shape) # (224, 224, 25) (224, 224, 25)
                #         unimportant_image = unimportant_image_list[j]
                #         permuted_mask = np.random.permutation(mask.flatten()).reshape(aligned_inputs[j].shape) # todo: another way is to shuffle in a unit of 3D blocks
                #         bl_modified_input = np.where(permuted_mask, unimportant_image, aligned_inputs[j])
                #         random_bl_input_list.append(self._img_3d_to_5d(bl_modified_input))
                #     # print(random_bl_input_list[0].shape, 'random_bs_input') # (1, 25, 3, 224, 224) random_bs_input
                #     random_bl_input_repeat.append(random_bl_input_list)
                # --- obsolete function ---


            # pass the modified input to model
            pred_batch = []  # pred for a batch
            for modified_input_list in modified_input_batch:
                new_modified_input_list = []
                for modified_input in modified_input_list:
                    modified_input = torch.as_tensor(modified_input)
                    modified_input = modified_input.to(self.device)
                    new_modified_input_list.append(modified_input)
                pred = self.model.forward(*new_modified_input_list)
                pred = torch.sigmoid(pred).cpu().detach().numpy()
                pred = np.squeeze(pred)
                # print(pred.shape, 'pred shape', pred) # () pred shape 0.024714189
                pred_batch.append(pred)
                # pred = np.argmax(pred, axis=1)
            preds.append(pred_batch) # 2D list. row: quantile steps. col: 0: modified input pred, rest: randomized baseline pred
            # if len(preds) ==2:
            #     break # todo
        preds = np.stack(preds) # [quantile_step, bl_num+1]
        # print(preds.shape)



        return preds, quantile_step

    def _img_3d_to_5d(self, img_3d):
        # from  (224, 224, 25) to (1, 25, 3, 224, 224)
        img_3d = img_3d.transpose(2, 0, 1)
        img_4d = np.stack((img_3d,) * 3, axis=1)
        img_5d = np.expand_dims(img_4d, axis=0)
        return img_5d


class Explain_Consistency(BaseXAIevaluator):
    def __init__(self, config):
        super().__init__(config)
    def consistency(self, dataloader, model, ori_heatmap_pickle = None):
        '''
        Heatmap eval experiment:
        1. At test time, get original heatmap
        2. Linear transform the input
        3. Get heatmap, and prediction
        4. Reverse the transform, and compare reversed heatmap with original heatmap, check if model performance also changed.
        '''
        # 1. get before original heatmaps
        save_dir = '../exp_log/heatmap_exp/consistency/'
        if self.exp_name == None: # make sure all pickles are saved in one directory. If no exp_name, will save in different dir with timestamps
            dateTimeObj = datetime.now()
            self.exp_name = dateTimeObj.strftime("%Y%m%d_%H%M")
        if ori_heatmap_pickle:
            # read saved pickle path
            ori_heatmaps = pickle.load(open(ori_heatmap_pickle, "rb"))
        else:
            ori_heatmaps = generate_heatmap_dataset(
                    dataloader, model, method_list =self.method_list, device = self.device, save_input = True, exp_condition = 'consistency_before')
            # save the file as pickle
            self.save_heatmap_pickle(exp_dir = save_dir,
                                exp_name = self.exp_name,
                                before = ori_heatmaps)
        # 2. Linear transform the input
        global fold
        global data_root
        input_size = (128, 128, 128)
        compose_before_transform = [LoadNiftid(keys=["image", "seg"]),
                                    Orientationd(keys=["image", "seg"], axcodes="RAS")]
        compose_after_transform =  [NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                                    Resized(keys=["image", "seg"], spatial_size=input_size),
                                    ToTensord(keys=["image", "seg"])]
        angle1 = 10
        angle2 = 30
        affine_dict = {'flip0': ([Flipd(spatial_axis = 0, keys=["image", "seg"])], Flip(spatial_axis= 0)),
                       'flip2': ([Flipd(spatial_axis=2, keys=["image", "seg"])], Flip(spatial_axis=2)),
                       'flip012': ([Flipd(spatial_axis=None, keys=["image", "seg"])], Flip(spatial_axis=None)),
                       'rotate10': ([Rotated(angle = angle1, keys=["image", "seg"])], Rotate(angle=-angle1)),
                       'rotate30': ([Rotated(angle=angle2, keys=["image", "seg"])], Rotate(angle=-angle2))
                       }
        after_hm_dict = dict()#{k: None for k in affine_dict}
        for k, t in affine_dict.items():
            cmps = Compose(compose_before_transform + t[0] + compose_after_transform)
            valloader = load_dataloader(fold, data_root, composer = cmps)
            after_heatmaps = generate_heatmap_dataset(
                    dataloader, model, method_list =self.method_list, save_input = True, exp_condition = 'consistency_before',device = self.device)
            # save the after heatmap (before inverse transform)
            after_hm_dict[k] = after_heatmaps
            # save the file as pickle
            self.save_heatmap_pickle(exp_dir = save_dir,
                                exp_name = self.exp_name,
                                after = after_heatmaps,
                                postfix= k ) # label for the transformed input
            # TODO visualize the transformed images, and (reverse-)transformed heatmaps before/after
        # save affine_dict
        saved_dir = save_dir + self.exp_name
        pickle.dump(affine_dict, open(os.path.join(saved_dir, 'affine_dict.pkl'), 'wb'))
        return save_dir, affine_dict, ori_heatmaps, after_hm_dict

    def consistency_pipeline(self, dataloader, model, ori_heatmap_pickle = None):
        '''
        Consistency pipeline, compute heatmaps from scratch data and model
        :param dataloader:
        :param model:
        :param method_list:
        :param exp_name:
        :param ori_heatmap_pickle:
        :return:
        '''
        save_dir, affine_dict, ori_heatmaps, after_hm_dict = consistency(dataloader, model, method_list=method_list, exp_name=self.exp_name, ori_heatmap_pickle=ori_heatmap_pickle)
        for k, t in affine_dict.items():
            rt = t[1]
            results, acc_df = compare_bfaf_heatmaps(ori_heatmaps, after_hm_dict[k], reverse_transform = rt, postfix=k)

    def consistency_pipeline_from_pickle(self, save_dir):
        affine_dict, ori_heatmaps, after_hm_dict = find_before_after_pickle_multiple(save_dir)
        for k, t in affine_dict.items():
            rt = t[1]
            print(k)
            results, acc_df = compare_bfaf_heatmaps(ori_heatmaps, after_hm_dict[k], reverse_transform = rt, postfix=k)


class Detect_Bias(BaseXAIevaluator):
    def __init__(self, config):
        super().__init__(config)
    def detect_bias(self, dataloader, biased_model, bgloader):
        '''
        Heatmap eval experiment:
        1st half: generate heatmaps
        Given a trained gt biased model, evaluate whether heatmaps methods can alert such a biased model
        1. generate heatmaps on the biased model with normal test set, and compare the heatmap:
        2. Portion of heatmap attribution inside/outside bg mask
        3. Similarity with (bg masks * heatmaps generated on pattern bg only)
        :return:
        '''
        # save heatmaps in the same directory
        save_dir = '../exp_log/heatmap_exp/bias/'
        if self.exp_name == None: # make sure all pickles are saved in one directory. If no exp_name, will save in different dir with timestamps
            dateTimeObj = datetime.now()
            self.exp_name = dateTimeObj.strftime("%Y%m%d_%H%M")
        # 1. generate heatmaps on the biased model with normal test set
        bias_heatmaps_dict = generate_heatmap_dataset(
            dataloader, biased_model, method_list=method_list, save_input=True, exp_condition='biased_model', device = self.device)
        # bias_heatmaps = get_heatmaps_from_dict(bias_heatmaps_dict)
        self.save_heatmap_pickle(after = bias_heatmaps_dict,
                            exp_dir=save_dir,
                            exp_name=self.exp_name)
        # 2. Portion of heatmap attribution inside/outside bg mask
        # heatmap_bg_portion(bias_heatmaps, bg_mask) # leave the calculate results func in another one using saved pickle
        # 3. Similarity with (bg masks * heatmaps generated on pattern bg only)
        bg_heatmaps_dict = generate_heatmap_dataset(
            bgloader, biased_model, method_list=method_list, save_input=True, exp_condition='biased_model')
        # mask out the brain part
        # bg_heatmaps = get_heatmaps_from_dict(bg_heatmaps_dict)
        # masked_bg_heatmaps = mask_heatmap_dataset(bg_heatmaps, bg_mask)
        # compare_bfaf_heatmaps(bias_heatmaps, masked_bg_heatmaps)
        self.save_heatmap_pickle(before = bg_heatmaps_dict,
                            exp_dir=save_dir,
                            exp_name=self.exp_name)
        return save_dir

    def detect_bias_eval(self, save_dir, mask_path):
        '''
        Heatmap eval experiment:
        2nd half: Calculate the saved heatmaps w/ gt
        :param pickle_path: saved save_dir from detect_bias()
        :param mask_path: mask[bratsID] = mask of [H,W,D], with downsized to fit model input size
        :return:
        '''
        # read saved pickle files
        bg_heatmaps_dict, bias_heatmaps = self.find_before_after_pickle(save_dir) # before: bg_heatmaps_dict,bias_heatmaps -after
        bg_mask = None # Todo: read bg mask files

        # 2. Portion of heatmap attribution inside/outside bg mask
        portion_dict = self.heatmap_bg_portion(bias_heatmaps, bg_mask)
        # save the portion_dict
        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d_%H%M")
        portion_filename =  os.path.join(save_dir, 'bias_bg_portion_{}.csv'.format(time_stamp))
        portion_dict.to_csv(portion_filename, index=False)

        self.compare_bfaf_heatmaps(bg_heatmaps_dict, bias_heatmaps, bg_mask_dict = bg_mask)



    def heatmap_bg_portion(self, heatmaps_dataset, bg_mask_dict):
        '''
        detect_bias() helper func
        Calculate the portion outside (mask = 1) / inside (mask=0) the background mask
        :param bias_heatmaps: results from generate_heatmap_dataset:
        heatmap_dict[bratsID] = (heatmap, pred, labels_val, time_record)
        :param bg_mask_dict: mask[bratsID] = mask of [H,W,D], with original input size
        :return:
        '''
        portion_dict = {}
        for k in heatmaps_dataset:
            heatmap_dict = heatmap_dict[k][0]
            bg_mask = bg_mask_dict[k]
            for m in heatmap_dict:
                hm = heatmap_dict[m]
                if m not in portion_dict:
                    portion_dict[m] = list()
                portion_dict[m].append(self.hm_mask_portion(hm, bg_mask))
        for m in portion_dict:
            portion_list = np.array(portion_dict[m]) # 2d array [channel, bratsID]
            mean, std = portion_list.mean(axis=1), portion_list.std(axis=1)
            print('Method: , heatmap_bg_portion: mean - {}, std -{}'.format(m, mean, std))
        return portion_dict

    def hm_mask_portion(self, hms, mask, ABS=False):
        '''
        detect_bias() helper func
        :param hms: [C, H, W, D]
        :param mask: outside (mask = 1) / inside (mask=0)
        :param ABS:
        :return:
        '''
        portion_list = [] # save the portion for 4 channels
        if hms.shape != mask.shape:
            hms = Resize(spatial_size=mask.shape)(hms)
        if ABS:
            hms = np.absolute(hms)
        print('Raw heatmap min-max range:\n{}, {}'.format(np.array(hms).min(), np.array(hms).max()))
        scl_hms = self.scale_hms(hms)
        print('Scaled [-1,1] heatmap min-max range:\n{}, {}'.format(scl_hms.min(), scl_hms.max()))
        for scl_hm in scl_hms:
            portion = scl_hm[mask == 1].sum() / scl_hm.sum()
            portion_list.append()
        return portion_list


    def get_heatmaps_from_dict(self, heatmap_dict):
        '''
        detect_bias() helper func
        Get heatmaps from the dict generated by generate_heatmap_dataset()
        :param heatmap_dict: heatmap_dict[bratsID] = (heatmap, pred, labels_val, time_record, images)
        :return: heatmaps[bratsID] = {'method1': heatmap, 'method2': heatmap of [C, H,W,D]}
        '''
        heatmaps = dict()
        for k in heatmap_dict:
            heatmaps[k] = heatmap_dict[k][0]
        return heatmaps

    def mask_heatmap(self, heatmap, mask):
        '''
        detect_bias() helper func
        single heatmap and mask calculation
        :param heatmap: original output size (128, 128, 128)
        :param mask:  mask of [H,W,D], with original input size
        :return:
        '''
        hm_rz = Resize(spatial_size=mask.shape)(hm)
        masked_hm = np.zeros(hm_rz.shape)
        masked_hm[mask!=0] = hm_rz[mask!=0]
        return masked_hm

    # def mask_heatmap_dataset(heatmap_dict, mask_dict):
    #     '''
    #     obsolete
    #     detect_bias() helper func
    #     Mask heatmaps in the heetmap_dict
    #     :param heetmap_dict: heatmaps[bratsID] = {'method1': heatmap, 'method2': heatmap of [C, H,W,D]
    #     :param mask: mask[bratsID] = mask of [H,W,D], with original input size
    #     :return: masked_hm_dict
    #     '''
    #     masked_hm_dict = {}
    #     for idx in heatmap_dict:
    #         id_dict = {}
    #         for m in heatmap_dict[idx]:
    #             id_dict[m] = mask_heatmap(heatmap_dict[idx][m], mask_dict[idx])
    #         masked_hm_dict[idx] = id_dict
    #     return masked_hm_dict

#
#         def controlled_exp(self, dataloader, biased_models_dir, bgloader, mask_path, method_list, exp_name=None, ori_heatmap_pickle=None):
#             # TODO load biased_models in dir
#             biased_models = []
#             for model in biased_models:
#                 save_dir = detect_bias(dataloader, model, bgloader, method_list=method_list, exp_name=self.exp_name)
#                 # detect_bias_eval(save_dir, mask_path)
# #


# if __name__ == '__main__':
#     # global method_list
#     logging.basicConfig(level=logging.INFO)
#     # get the command line parameters
#     parser = build_parser()
#     opt = parser.parse_args()
#
#     # data_root = 'ts'
#     data_root = None
#     model_file = None
#     if opt.data_root == 'cc': # need to specify the string in ""
#         data_root = '/scratch/authorid/dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/'
#         model_file = '/scratch/authorid/BRATS_IDH/log/whole_balanced_0918/fold_1_epoch_46.pth'
#     elif opt.data_root == 'ts':
#         data_root = '/local-scratch/authorid/dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/'
#         model_file = '/local-scratch/authorid/BRATS_IDH/log/heatmap_model/fold_1_epoch_0.pth'
#     else:
#         logging.warning('>> Error, no data_root specified!')
#     fold =opt.fold
#     computnormalmasks = opt.normal_masks

    # valloader = load_dataloader(fold, data_root)
    # logging.info('\nDONE loading data')
    # model = load_model(model_path= model_file, model_architecture=Resnet3D)
    # logging.info('\nDONE loading model')

    # exp_name = 'test'

    # test randomize_model
    # pickle_path = randomize_model(valloader, model, method_list, computnormalmasks, exp_name = exp_name)
    # print(pickle_path)

    # test compare_bfaf_heatmaps
    # pickle_path = '../exp_log/heatmap_exp/randomize_model/test'
    # results, acc_df =   compare_bfaf_heatmaps_pipeline(pickle_path)
    # logging.info(results)
    # logging.info(acc_df)

    # test compare_with_gt_dataset
    # seg_path = os.path.join(data_root, 'all')
    # df, metric_allmod = compare_with_gt_dataset(pickle_path, seg_path= seg_path)
    # logging.info(df)
    # logging.info(metric_allmod)

    # test acc drop
    # drop_df, auc_list, drop_filename, auc_filename = acc_drop(valloader, model, steps=10, method_list=method_list, exp_name= exp_name)
    # print(drop_filename)
    # print(auc_filename)
    # # drop_filename = '../exp_log/heatmap_exp/acc_drop/test/acc_drop_20201208_1143.csv'
    # # auc_filename = '../exp_log/heatmap_exp/acc_drop/test/auc_acc_drop_20201208_1143.csv'
    # plot_acc_drop(drop_filename, auc_filename)

    # test consistancy
    # consistency_pipeline(valloader, model, method_list, exp_name = exp_name)
    # pickle_path = '../exp_log/heatmap_exp/consistency/test/'
    # consistency_pipeline_from_pickle(pickle_path)

