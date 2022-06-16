from validate import test
from pathlib import Path
from .heatmap_eval import BaseXAIevaluator
from .heatmap_utlis import postprocess_heatmaps, heatmap_to_binary
import numpy as np
import logging
import pickle, os, glob
import csv
from sklearn.metrics import jaccard_score
from tqdm import tqdm
def tumorsyn_get_test_performance(config, saveinput = False):
    save_path = config['trainer']['save_dir']
    fold = config['data_loader']['args']['fold']
    val_dataset = config['data_loader']['args']['val_dataset']
    input_save_dir = config['data_loader']['args']['input_save_dir']
    # each fold has a seperate fold to save test data, and load for later get_hm
    input_save_dir = Path(input_save_dir) / "tumor4mod_{}".format(fold)  # todo
    test_input_save_dir = Path(input_save_dir) / '{}'.format({True: 'val', False: 'test'}[val_dataset])
    test_input_save_dir.mkdir(parents=True, exist_ok=True)
    val_sec_gt_align_prob = float(int(fold) / 10)
    print('===Fold = {}, val_sec_gt_align_prob = {} ==='.format(fold, val_sec_gt_align_prob))
    image_label = 'tumor' # todo change to tumor
    if saveinput:
        saveinput = test_input_save_dir
    test(config, image_label = image_label, timestamp = True,
         csv_save_dir=test_input_save_dir,
         saveinput = saveinput,
         val_first_gt_align_prob=1,
         val_sec_gt_align_prob=val_sec_gt_align_prob,
         combine_with_brain = [1,1,1,1])
    if not val_dataset :
        # Test_T1c, first modality
        first_input_save_dir = Path(input_save_dir) / "first"
        first_input_save_dir.mkdir(parents=True, exist_ok=True)
        if saveinput:
            saveinput = first_input_save_dir
        test(config, image_label = image_label, timestamp = True,
             csv_save_dir=first_input_save_dir,
             saveinput=saveinput,
             val_first_gt_align_prob=1,
             val_sec_gt_align_prob=0,
             combine_with_brain = [1,1,1,1])
        # Test_Flair, 2nd modality
        second_input_save_dir = Path(input_save_dir) / "second"
        second_input_save_dir.mkdir(parents=True, exist_ok=True)
        if saveinput:
            saveinput = second_input_save_dir
        test(config, image_label = image_label, timestamp = True,
             csv_save_dir=second_input_save_dir,
             saveinput=saveinput,
             val_first_gt_align_prob=0,
             val_sec_gt_align_prob=1,
             combine_with_brain = [1,1,1,1])
        # baseline
        second_input_save_dir = Path(input_save_dir) / "baseline"
        second_input_save_dir.mkdir(parents=True, exist_ok=True)
        if saveinput:
            saveinput = second_input_save_dir
        test(config, image_label = image_label, timestamp = True,
             csv_save_dir=second_input_save_dir,
             saveinput=saveinput,
             val_first_gt_align_prob=0,
             val_sec_gt_align_prob=0,
             combine_with_brain = [1,1,1,1])
        # 2nd col mod
        # second_input_save_dir = Path(input_save_dir) / "mod2"
        # second_input_save_dir.mkdir(parents=True, exist_ok=True)
        # test(config, image_label = image_label, timestamp = True,
        #      csv_save_dir=second_input_save_dir,
        #      val_first_gt_align_prob=1,
        #      val_sec_gt_align_prob=1,
        #      combine_with_brain = [0,1,0,0])
        # # 4th mod
        # second_input_save_dir = Path(input_save_dir) / "mod4"
        # second_input_save_dir.mkdir(parents=True, exist_ok=True)
        # test(config, image_label = image_label, timestamp = True,
        #      csv_save_dir=second_input_save_dir,
        #      val_first_gt_align_prob=1,
        #      val_sec_gt_align_prob=1,
        #      combine_with_brain = [0,0,0,1])

def tumorsyn_get_hm(config, machine = 'ts',
                    saved_dir = "authorid/trained_model/tumorsyn/7/test/get_hm_input"
                    # saved_dir="/local-scratch/authorid/trained_model/tumorsyn/7/test/20210528_1919"
                    ):
    fold = config['data_loader']['args']['fold']
    if machine == 'solar':
        root = '/project/labname-lab/'

    else:
        root = '/local-scratch/'
    saved_dir = root + saved_dir

    # don't use different random test images for different fold models, as we want to keep the test img fixed as well.
    # fold_specific_test_input = {
    #     3: '20211113_0629',
    #     # 6: '20211113_1033',
    #     # 7: '20211113_0952',
    #     # 8: '',
    #     # 9: '20211113_0947'
    # }
    # # designate the saved_dir for test input
    # saved_dir = root+'authorid/trained_model/tumorsyn/tumor4mod_{}/test/{}_get_hm_input'.format(fold, fold_specific_test_input[fold])


    # setup xai evaluator, load model and data
    xai_eval = BaseXAIevaluator(config)
    model = xai_eval.get_model()
    # print(model.features.denseblock4.denselayer16.layers.conv2)
    # print(model)
    # print(model.state_dict().keys())
    dataloader = xai_eval.get_dataloader()
    val_loader = dataloader.get_val_loader_presaved(saved_dir, shuffle= False)
    print(len(val_loader))
    result_dir = xai_eval.generate_save_heatmap(val_loader, model, 'tumorsyn_get_hm_fold_{}'.format(fold))
    logging.info(result_dir)

def tumorsyn_results(gt_attr = [0,0,1,0], ABS= False, iou= False,
                     input_dir = "/local-scratch/authorid/trained_model/tumorsyn/7/test/20210528_1919",
                     hm_dir = "/local-scratch/authorid/tumorsyn/log/tumorsyn/gethm_fold_7/tumorsyn_get_hm_fold_7_fold_7/heatmap"):
#     method_list = config['xai']['method_list']
    method_list = ["Occlusion", "FeatureAblation", "KernelShap", "ShapleyValueSampling",
            "Gradient", "GuidedBackProp", "GuidedGradCAM",
                   "DeepLift", "InputXGradient", "Deconvolution",
                   "SmoothGrad", "IntegratedGradients","GradientShap", "FeaturePermutation", "Lime", "GradCAM"]
    fnames = ['dataID',  'IoU', 'XAI','pred_label', "absolute_hm"]
    for method in method_list:
        # get all hms
        hms1 = glob.glob(os.path.join(hm_dir, '*-{}.pkl'.format(method)))
        hms2 = glob.glob(os.path.join(hm_dir, '*-{}-*.pkl'.format(method)))
        hms = hms1 + hms2
        print(method, len(hms1), len(hms2), len(hms))
        csv_dir = Path(hm_dir).parent/'portion'
        csv_dir.mkdir(parents= True, exist_ok = True)
        csv_filename = csv_dir /'{}-{}.csv'.format(method, len(hms))
        file_exists = os.path.isfile(csv_filename)
        if file_exists:
            continue
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
            csv_writer.writeheader()
            for hm_fn in tqdm(hms):
                dataid = Path(hm_fn).name.split('.')[0].split('-')[0]
                hm = pickle.load(open(hm_fn, "rb"))
                hm = postprocess_heatmaps(hm, img_size = (256, 256), no_neg = True, rotate_axis= False) # get positive values only

    #             print(dataid)
                # get gt maps
                input_data = pickle.load(open(os.path.join(input_dir, "{}.pkl".format(dataid)), "rb"))
                seg = input_data['seg'] #* np.array(gt_attr)
                if not isinstance(seg,(np.ndarray)):
                    seg = seg.detach().cpu().numpy()
                seg = (seg>0).astype(int)
    #             print(seg[0].sum(), seg[1].sum(), seg[2].sum(),seg[3].sum())
                # masked segmentation with gt moadlity importance
                for i in range(seg.shape[0]):
                    seg[i] = seg[i] * gt_attr[i]
    #             print(seg[0].sum(), seg[1].sum(), seg[2].sum(),seg[3].sum())
                if iou:
                    b_hm = heatmap_to_binary(hm, criteria='t', cutoff=0.5, ABS=ABS)
                    score = jaccard_score(seg.reshape(-1), b_hm.reshape(-1))
                else: # use tumor portion for positive value inside seg / total
                    score = hm[seg == 1].sum() /hm.sum()
                fn_label = Path(hm_fn).name.split('.')[0].split('-')
                if len(fn_label) == 3:
                    pred_label = fn_label[-1]
                else:
                    pred_label = "correct"
                csv_record = {'dataID':dataid, 'IoU': score, 'XAI': method, 'pred_label': pred_label, 'absolute_hm': ABS}
                logging.info(csv_record)
                csv_writer.writerow(csv_record)

