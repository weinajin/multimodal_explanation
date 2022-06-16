import numpy as np
import torch
import torch.nn as nn
import logging
import pickle
import os
import time
import csv
from pathlib import Path
import gc
from monai.data import write_nifti, NiftiSaver
# from skimage.measure import regionprops
from skimage.segmentation import slic

from datetime import datetime
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import (
    Saliency,
    GradientShap,
    GuidedBackprop,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    GuidedGradCam,
    LayerActivation,
    InputXGradient,
    Deconvolution,
    FeatureAblation,
    Occlusion,
    NoiseTunnel,
    ShapleyValueSampling,
    KernelShap,
    Lime,
    FeaturePermutation,
    LayerGradCam
)

# data import
# from data import get_data

# model specific import
# from CNN3 import GeneNet
# from resnet3d import Resnet3D

# record memory usage
import psutil
import resource

import scipy.ndimage as ndimage

'''
Post-hoc heatmap map on the input 4 MR modalities.
It serves as baseline of showing the current XAI methods.
'''



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device_ids = [0]
# device = torch.device('cpu')

# def load_dataloader(fold, data_root, composer = None):
#     '''
#     #     Obsolete function
#
#     load validation data
#     :return:
#     '''
#     logging.info('\nLoading the validation dataset\n')
#     val_csv = os.path.join(data_root, 'IDH', 'val_fold_{}.csv'.format(fold))
#     logging.info(val_csv)
#     dataset = 'BRATS_IDH'
#     valloader = get_data(name=dataset, data_root=data_root, over_sample = False, roi = False,
#                        batch_size=1, fold=fold, val_only=True, composer = composer)
#     return valloader


# def load_model(model_path, model_architecture, reinit= False):
#     '''
#     Obsolete function
#     load trained model given path, and model architecture
#     :param:
#     model_path: the path of the saved model
#     model_architecture: the model architecture
#     :return:
#     '''
#     net = model_architecture(in_channels=4, num_class=2).to(device)
#     # move to GPU
#     model = nn.DataParallel(net, device_ids=device_ids).to(device)
#     if reinit:
#         model.eval()
#         return model
#     model.load_state_dict(torch.load(model_path))
#     # logging.info(model.state_dict().keys())
#     # final_layer = list(model.ModuleDict())
#     # logging.info(final_layer)
#     model.eval()
#     return model

# def get_sliency_constructors(model,
#                              conv_layer=None):
#     """
#     Obsolete func
#     Returns functions to compute saliency masks for methods in saliency package.
#     Adapoted from:
#     https://github.com/adebayoj/sanity_checks_saliency/blob/master/notebooks/mlp_mnist_cascading_randomization.ipynb
#     Args:
#         target: tensor corresponding to the logit output of the network.
#         input_tensor: tensor coressponding to the input data.
#         gradcam: Boolean to indicate whether to compute gradcam saliency maps.
#         conv_layer_gradcam: tensor corresponding to activations from a conv layer,
#                             from the trained model. Authors recommend last layer.
#     Returns:
#         saliency_constructor: dictionary where key is name of method, and value is
#                               function to each saliency method.
#
#         neuron_selector: tensor to indicate which specific output to explain.
#     """
#     # target layer for gradcam
#     conv_layer = model.module.features.residualblock2.conv
#     # saliency map methods
#     gradient = Saliency(model).attribute
#     gradient_shap = GradientShap(model).attribute
#     guided_backprop = GuidedBackprop(model).attribute
#     guided_gradcam = GuidedGradCam(model, conv_layer).attribute
#     integrated_gradients = IntegratedGradients(model).attribute
#
#     saliency_funcs = {'Gradient': Saliency(model).attribute,
#                         'GradientShap': GradientShap(model).attribute,
#                         'GuidedBackProp': GuidedBackprop(model).attribute,
#                         'GuidedGradCAM': guided_gradcam,
#                         'IntegratedGradients': integrated_gradients}
#     return saliency_funcs
#

# def generate_mask(input, window_size, overlap):
#     """
#     Obsolete function. use superpixel instead
#      feature_mask defines a mask for the input, grouping features which correspond to the same interpretable feature.
#      feature_mask should contain the same number of tensors as inputs.
#     :param shape:
#     :param window_size:
#     :param stride:
#     :return:
#     """
#     window_size = np.array(window_size)
#     overlap = np.array(overlap)
#     masks = []
#     bg_mask = np.where(input.numpy() == 0, 1, 0)
#     brain_mask = np.where(input.numpy() != 0, 1, 0)
#     for i in range(input.shape[1]):
#         img3d = np.squeeze(brain_mask)[i]
#         prop  = regionprops(img3d)
#         bbox = prop[0].bbox # (min_row, min_col, max_row, max_col)
#         print(bbox)
#         j = 0
#         start, end = np.array(bbox[:3]), np.array(bbox[3:])
#         print(start, end)
#         while j >=0:
#             window_end = np.add(start ,window_size)
#             print(window_end)
#             start = np.subtract(window_end ,overlap)
#             if window_end[0] > end[0] and window_end[1] > end[1] and window_end[2] > end[2]:
#                 print(window_end)
#                 break
#
#         mask = None
#         masks.append(bg_mask[i])
#         masks.append(mask)
#     masks = torch.from_numpy(np.stack(masks))
#     return masks

def get_process_memory():
    # https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms, mi.shared

def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes)+"B"
    elif abs(bytes) < 1e6:
        return str(round(bytes/1e3,2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"

def superpixel_mask(img, mod_specific = True):
    """
    assume dim 0 of img is the batch_size
    SLIC - K-Means based image segmentation
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html
    :param input: Input image, which can be 2D or 3D, and grayscale or multichannel (see multichannel parameter).
    Input image must either be NaN-free or the NaN’s must be masked out
    Input: should be the same size as a single batch size=1 input, dim 0 = 1
    :return:
    """
    bs_segments = []
    # for img in bs_img:
    # get a single batch image
    for i in range(img.shape[0]): # iterate over batch
        img_single_batch = img[i].cpu().numpy()  # from [bs, C, H, W, D] to [C, H, W, D]
        # channel_last_img = np.swapaxes(img_single_batch, 0,3) # [H, W, D, C]
        # print(channel_last_img.shape)
        segments = []
        prev_n_label = 0
        for c in range(img_single_batch.shape[0]):
            single_modality = img_single_batch[c]
            bg = np.where(single_modality == 0, 0, 1)  # mask = 0 is background

            # print(single_modality.shape)
            n_segments = 50
            if mod_specific:
                segments_slic = slic(single_modality, n_segments= n_segments, compactness=0.1, sigma=2, multichannel=False,
                                     start_label=1, #min_size_factor=0.8, max_size_factor=3, enforce_connectivity = True,
                                     mask = bg)  # [H, W, D]
                # add mask label indicies, so that not to be correlated with other modalities.
                if c>0:
                    # print(prev_n_label, np.unique(segments_slic))
                    segments_slic = np.where(segments_slic == 0, 0, segments_slic+prev_n_label)
                prev_n_label += len(np.unique(segments_slic)) - 1
                segments.append(segments_slic)
            else: # generate the same mask for all mods, suitable for KernalSHAP, Feature Permutation
                # <<-- correct the all modality the same
                if c == 1:
                    segments_slic = slic(single_modality, n_segments=n_segments, compactness=0.1, sigma=2,
                                         multichannel=False,
                                         start_label=1,
                                         # min_size_factor=0.8, max_size_factor=3, enforce_connectivity = True,
                                         mask=bg)  # [H, W, D]
                    # channel_last_img = np.transpose(img_single_batch, (1,2,3,0)) # [H, W, D, C]
                    # segments_slic = slic(channel_last_img, n_segments=50, compactness=10, sigma=1, multichannel= True,
                    #                      start_label=0) #[H, W, D]
                    # segments_slic = slic(channel_last_img, n_segments=50, compactness=1, sigma=1.5, multichannel=True,
                    #                      start_label=0, min_size_factor=0.8, max_size_factor=3)  # [H, W, D]
                    segments = np.stack([segments_slic]*img.shape[1])   # [C, H, W, D]
                # correct the all modality the same -->>
        if mod_specific:
            segments = np.stack(segments)  # [C, H, W, D]
        segments = np.expand_dims(segments,0) # [1, C, H, W, D]
    # print(img_single_batch.shape, channel_last_img.shape, segments_slic.shape, segments.shape)
    # print(segments.shape,'segment.shape')
        bs_segments.append(segments)
    bs_segments = np.concatenate(bs_segments)
    # print(bs_segments.shape)
    return torch.from_numpy(segments)

def generate_hm_filename(data_id, xai_name, target=None, pred=None, gt=None):
    gt = int(gt)
    if target:
        if type(target) != type(gt):
            target = target.cpu().numpy()
    if (pred == gt) or ((pred>0.5) == gt):
        hm_filename = '{}-{}.pkl'.format(data_id, xai_name)
    else:  # if pred wrong:
        if target:
            if target == pred:  # if passed target = pred
                target_postfix = "P"+ str(pred)  # xaimethod_bratsID_target(gt).pkl
            else: # hm for gt, AI not predicted
                target_postfix = "T" + str(gt)
        else: # if target is None, a single tensor
            target_postfix = "P" + str(pred)
        hm_filename = '{}-{}-{}.pkl'.format(data_id, xai_name, target_postfix)
    # print(pred == gt, pred, gt, hm_filename, type(target))
    return hm_filename

def generate_heatmap(input, model, target, dataIDs, method_list, device, last_layer = None, gt = None, pred = None, save_dir = None): #, save= False, postfix = None):
    '''
        Generate heatmpas according to the data, model, and method.
    Since CPU OOM issue, generate heatmap and save as:
    save_dir (pass as resume arg)  / exp_label (the experiment + fold #) /
        - heatmap / xaimethod_bratsID.pkl
        - input / bratsID.pkl
        - record_hm_generator.csv file: writerow - bratsID, xaimethod, pred, gt, time

    If gt is given, and it's different from prediction (target), generate additional hm for gt (the not predicted one)

    Main function to include the 10+ post-hoc XAI methods to generate heatmaps for individual input
    Given a instance and model, generate saliency map
    put data and model in cpu, to avoid OOM.
    :param:
        method_list: a name list of the heatmap method
    :return attributions: saliency map of the instance on different modalities
    '''
    # device = "cpu"
    input.requires_grad = True
    model.zero_grad()
    input = input.to(device)#cpu().detach()
    target = target.to(device)
    model.to(device)
    # print(model.state_dict().keys())
    # Parameters used for some functions
    input_dim = len(input.shape)
    baseline = torch.zeros(input.shape)
    mean_bs = torch.mean(input, axis = (1,2,3))
    img_dim = [i for i in range(input_dim)][2:]
    # used for Occlusion baselines and sliding window
    input_mean = torch.mean(input, dim=img_dim)
    input_std = torch.std(input, dim=img_dim)
    # generate fill in value of shape input_dim
    fill = []
    for b in range(input.shape[0]):
        fill_c = []
        for c in range(input.shape[1]):
            fill_in_value = torch.normal(mean = input_mean[b][c], std=input_std[b][c], size = input.shape[2:]) # generate random fill-in for each data in batch
            fill_c.append(fill_in_value)
        fill_c = torch.stack(fill_c)
        fill.append(fill_c)
    fill = torch.stack(fill)
    fill = fill.to(device)
    window_size = 11
    stride_step = 5
    # sliding_window_shapes =  tuple([tuple([window_size for i in range(input_dim-2)]) for i in range(input.shape[1])])
    sliding_window_shapes = [window_size for i in range(input_dim - 2)]
    sliding_window_shapes.insert(0, 1)
    sliding_window_shapes = tuple(sliding_window_shapes)
    strides = [stride_step for i in range(input_dim - 2)]
    strides.insert(0, 1)
    strides = tuple(strides)
    # superpixel mask for perturbation methods
    feature_mask = superpixel_mask(input, mod_specific = True)
    mod_unify_feature_mask = superpixel_mask(input, mod_specific = False) #  features are grouped across modalities
    # print('main', feature_mask.shape)


    # print(model.state_dict().keys())
    # Select algorithm to instantiate and apply
    # target layer for gradcam
    conv_layer = getattr(model, last_layer)   # GeneNet model last conv layer
    # conv_layer = model.features.denseblock4.denselayer16.layers.conv2 # tumorsyn model densenet
    # print(conv_layer)
    # Setup XAI methods
    saliency_funcs = {'Gradient': Saliency(model).attribute,
                        'GradientShap': GradientShap(model).attribute,
                        'GuidedBackProp': GuidedBackprop(model).attribute,
                        'GuidedGradCAM': GuidedGradCam(model, conv_layer).attribute,
                        'GradCAM': LayerGradCam(model, conv_layer).attribute,
                        'IntegratedGradients': IntegratedGradients(model).attribute,
                      'DeepLift':DeepLift(model).attribute,
                      'DeepLiftShap': DeepLiftShap(model).attribute,
                      'InputXGradient': InputXGradient(model).attribute,
                      'Deconvolution': Deconvolution(model).attribute,
                      'Occlusion': Occlusion(model).attribute,
                      'SmoothGrad': NoiseTunnel(Saliency(model)).attribute,
                      'ShapleyValueSampling': ShapleyValueSampling(model).attribute,
                      'ShapleyValueSamplingModUnify': ShapleyValueSampling(model).attribute,
                      # 'LimeBase': LimeBase(model).attribute,
                      'KernelShap': KernelShap(model).attribute,
                      'FeatureAblation': FeatureAblation(model).attribute,
                      'FeaturePermutation': FeaturePermutation(model).attribute,
                      'Lime': Lime(model, interpretable_model = SkLearnLinearRegression()).attribute
                      }

    # set up params for each saliency method
    gen_feed_dict = {'inputs': input, 'target': target}  # need to set batch_size =1 in config file
    feed_dict_w_baseline = dict({"baselines": baseline}, **gen_feed_dict)
    pertubation_args = dict({"feature_mask": feature_mask, "perturbations_per_eval": 5}, **feed_dict_w_baseline)
    saliency_params = {'Gradient': dict({'abs': False}, **gen_feed_dict), # get pos & neg value of gradients
                       'GradientShap':feed_dict_w_baseline,
                       'GuidedBackProp':  gen_feed_dict,
                       'GuidedGradCAM': gen_feed_dict,
                       'GradCAM': gen_feed_dict,
                       'IntegratedGradients': gen_feed_dict,
                       'DeepLift': feed_dict_w_baseline,
                       'DeepLiftShap': feed_dict_w_baseline,
                       'InputXGradient': gen_feed_dict,
                       'Deconvolution': gen_feed_dict,
                       'Occlusion': dict({'baselines': fill, 'sliding_window_shapes': sliding_window_shapes, 'strides': strides, "perturbations_per_eval": 5 }, **gen_feed_dict),
                       'SmoothGrad': dict({'nt_type': 'smoothgrad', 'nt_samples':10 } , **gen_feed_dict),
                       'ShapleyValueSampling': pertubation_args,
                       "ShapleyValueSamplingModUnify": dict({"feature_mask": mod_unify_feature_mask, "perturbations_per_eval": 5}, **gen_feed_dict),
                       # 'LimeBase': gen_feed_dict,
                       'KernelShap': dict({"n_samples": 300, "feature_mask": mod_unify_feature_mask, "return_input_shape": True}, **feed_dict_w_baseline),
                       'FeatureAblation': pertubation_args,
                       'FeaturePermutation': dict({"feature_mask": mod_unify_feature_mask, "perturbations_per_eval": 5} , **gen_feed_dict),
                       'Lime': dict({"feature_mask": feature_mask, "return_input_shape": True, "n_samples": 500}, **gen_feed_dict)
                       }
    assert saliency_funcs.keys() == saliency_params.keys()
    # output_heatmaps = []
    # time_record = []
    batch_size = len(gt)

    # prepare dir to save the input and generated hms
    hm_dir = save_dir / 'heatmap'
    hm_dir.mkdir(parents=True, exist_ok=True)
    input_dir = save_dir /'input'
    input_dir.mkdir(parents=True, exist_ok=True)
    # create csv file
    csv_filename = save_dir / 'record_{}.csv'.format(str(save_dir.name))
    fnames = ['Data_ID', 'XAI_Method', 'HM_target', 'Prediction', 'GT', 'Predicted_Correct',
              'Time_spent', 'File_name', 'max_mem', 'rss', 'vms', 'share_mem']
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
        if not file_exists:
            csv_writer.writeheader()



    for m_idx, name in enumerate(method_list):
        if name in saliency_funcs:
            with open(csv_filename, 'a', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
                # check if the hm: heatmap / xaimethod_bratsID.pkl exist
                hms_exist = []
                for idx, d_id in enumerate(dataIDs):
                    hm_filename = generate_hm_filename(d_id, name, target[idx], pred[idx], gt[idx])
                    exist = Path(hm_dir/ hm_filename).is_file() #check_file_exist(hm_dir, hm_filename)
                    hms_exist.append(exist)
                # must meet all files exist
                if all(hms_exist):
                    logging.info("HM {} [{}] exist. Pass".format(dataIDs, name))
                    continue
                else:
                    # generate and save hm and input
                    params = saliency_params[name]
                    start_time = time.time()
                    print(name, 'input size:', input.shape)
                    attribution = saliency_funcs[name](**params)
                    time_cost = (time.time() - start_time) / batch_size
                    logging.info('\nXAI method: {} costed time: {}\n'.format(name, time_cost))
                    # time_record.append(time_cost)
                    # confirm the shape matchs with input
                    if name != "GradCAM":
                        assert attribution.shape == input.shape, "Heatmap original size doesn't match with the input"
                    # Generate hms
                    rss_before, vms_before, shared_before = get_process_memory()
                    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    attribution = attribution.cpu().detach().numpy().squeeze() # [bs, channel, H, W, D]
                    rss_after, vms_after, shared_after = get_process_memory()
                    delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
                    rss = rss_after - rss_before
                    vms = vms_after - vms_before
                    share_mem = shared_after - shared_before # in byte
                    print(delta_mem, format_bytes(rss), format_bytes(vms), format_bytes(share_mem))

                    # save the generated hms
                    for idx, d_id in enumerate(dataIDs):
                        hm_filename = generate_hm_filename(d_id, name, target[idx], pred[idx], gt[idx])
                        # save heatmap as pkl file

                        if len(attribution.shape) == len(input.shape): # batch_size > 1
                            # print('attribution shape bs >1', attribution.shape, attribution[idx].shape, input.shape)
                            attr_to_save = attribution[idx]
                        else:
                            attr_to_save = attribution
                            # print('attribution shape', attribution.shape, attribution[idx].shape, input.shape)
                        # when max == 0,
                        # sanity check that the hm is valid one, not all 0
                        # if np.absolute(attr_to_save).max() == 0:
                        all_zeros = not np.any(attr_to_save)
                        if all_zeros:
                            logging.info("\nWARNING: XAI method {} are all zero for data {}, hm shape {}.".format(name, d_id, attr_to_save.shape))
                            continue
                        else:
                            pickle.dump(attr_to_save, open(os.path.join(hm_dir, hm_filename), 'wb'))
                        # save record
                        csv_record = {'Data_ID': d_id, 'XAI_Method': name, 'HM_target': target.cpu().numpy()[idx] , 'Prediction': pred[idx],\
                                      'GT': gt[idx], 'Predicted_Correct': int(gt[idx]==pred[idx]), 'Time_spent': time_cost , 'File_name': hm_filename, 'max_mem': delta_mem, 'rss': rss, 'vms': vms, 'share_mem': share_mem}
                        logging.info(csv_record)
                        csv_writer.writerow(csv_record)
                        # save input
                        input_filename = '{}.pkl'.format(d_id)
                        if Path(input_dir/input_filename).is_file():
                            pass
                        else:
                            logging.info('save input {} : {}\n'.format(d_id, input[idx].shape))
                            pickle.dump(input[idx], open(os.path.join(input_dir, input_filename), 'wb'))
                    del attribution
                    gc.collect()

                    # logging.info('Attribution shape: {}'.format(attribution.shape))
                    # resample to original size of d_size, h_size, w_size = 155, 240, 240
                    # ori_size = input.shape[2:]
                    # attribution = Resize(spatial_size=ori_size)(attribution) # calculate sim don't need to resize to ori size, only needed when compare gt seg, and for vis
                    # output_heatmaps.append(attribution)
                # save
                # if save:
                #     channel_dict = {0: 't1', 1: 't1ce', 2: 't2', 3: 'flair'}
                #     dateTimeObj = datetime.now()
                #     file_name = postfix + '_' + name + '_' + dateTimeObj.strftime("%Y%m%d_%H%M")
                #     output_dir = '../heatmap/' + file_name
                #     output_postfix = 'heatmap'
                #     # saver = NiftiSaver(output_dir='../heatmap/'+file_name, output_postfix='heatmap', resample = True)
                #     for channel in range(attribution.shape[0]):
                #         # print(attribution[channel].shape, channel_dict[channel])
                #         filename = create_file_basename(output_postfix, channel_dict[channel], output_dir, data_root_dir=output_dir)
                #         # print(filename)
                #         write_nifti(attribution[channel], file_name=filename) # save as HWD
                #         # saver.save(attribution[channel], {'filename_or_obj':channel_dict[channel], 'spatial_shape':ori_size})

        else:
            print('{} not in saliency function list {}'.format(name, saliency_params.keys()))
    gc.collect()

    # return np.stack(output_heatmaps), time_record


def generate_heatmap_dataset(dataloader, model, save_dir, method_list, device,  image_key = 'image', last_layer = None, task=None):#, exp_condition = None, save_input = False):
    '''
    If gt is given, and it's different from prediction (target), generate additional hm for gt (the not predicted one) (by changing the target fet to generate_heatmap func
    '''
    # heatmap_dict = dict()
    # save = None
    # if exp_condition != None:
    #     save = True
    #     postfix = exp_condition

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader): #
            if task == "MRNet":
                # for modality fusion at feature level
                images = batch_data[0], batch_data[1], batch_data[2]
                labels_val, dataIDs = batch_data[3], batch_data[4]
                images_val = []
                for img in images:
                    images_val.append(img.to(device))

                model.to(device)
                pred = model.forward(*images_val)
                pred = torch.sigmoid(pred)
                # print(pred, labels_val, dataIDs)
                generate_heatmap_featurefusion(
                    images_val, model, save_dir= save_dir, target = None, pred = pred.cpu().detach().numpy(),  gt = labels_val,  dataIDs = dataIDs, method_list = method_list, last_layer = last_layer, device= device) # save = save, postfix = postfix,

            # for brats dataset, modality fusion at input channel level
            else:
                images, labels_val, dataIDs = batch_data[image_key], batch_data['gt'].numpy(), batch_data['bratsID']
                images_val = images.to(device)
                model.to(device)
                pred = model.forward(images_val)#.cpu().detach().numpy()
                pred =torch.argmax(pred, axis=1)
                # postfix = str(bratsID[0])+'_gt'+ str(int(labels_val[0])) + '_pred' + str(pred)
                # generate heatmaps one image at a time
                # heatmap shape = [method_list number, batch_size, C, H, W, D], time_record = [time spend for a batch for each method]
                # heatmap, time_record = \

                # generate heatmap for right and wrong prediction.
                # when pred is right, check if file exist first;
                # when pred is wrong, generate a new heatmap for target = gt
                generate_heatmap(
                    images_val, model, save_dir= save_dir, target = pred,  pred = pred.cpu().detach().numpy(),  gt = labels_val,  dataIDs = dataIDs, method_list = method_list, last_layer = last_layer, device= device) # save = save, postfix = postfix,
                generate_heatmap(
                    images_val, model, save_dir= save_dir, target = torch.tensor(labels_val), pred=pred.cpu().detach().numpy(), gt=labels_val, dataIDs = dataIDs, method_list=method_list, device= device,
                    last_layer=last_layer)
            # if i >3:
            #     break # todo tmp for test


            # bs = len(batch_data[image_key])
            # print(heatmap.shape)
            # print(heatmap[:, 1].shape)
            # single_img_time_record = [t/bs for t in time_record]

            # save each data point in a batch
            # for i in range(bs):
            #     for m_id, m in enumerate(method_list):
            #         # save heatmap as pkl file
            #         print('hm', heatmap[m_id, i].shape)
            #         pickle.dump(heatmap[m_id, i], open(os.path.join(hm_dir, '{}_{}.pkl'.format(m, bratsID[i])), 'wb'))
            #
            #         record = [bratsID[i], m, pred[i].cpu().detach().numpy(), labels_val[i], single_img_time_record ]
            #         csv_writer.writerow(record)
            #     # save input
            #     print(images[i].shape)
            #     pickle.dump(images[i], open(os.path.join(input_dir, '{}.pkl'.format(bratsID[i])), 'wb'))



                    # if save_input:
                    # heatmap_dict[bratsID[i]] = (heatmap[:, i], pred[i].cpu().detach().numpy(), labels_val[i], single_img_time_record , images[i])
                # else:
                #     heatmap_dict[bratsID[i]] = (heatmap[i], pred[i], labels_val[i], time_record[i])
            # if len(heatmap_dict) >1:
            #     break
    # return heatmap_dict

def create_file_basename(
    postfix: str,
    input_file_name: str,
    folder_path: str,
    data_root_dir: str = "",
    ) -> str:
    """
    Utility function to create the path to the output file based on the input
    filename (file name extension is not added by this function).
    When `data_root_dir` is not specified, the output file name is:

        `folder_path/input_file_name (no ext.) /input_file_name (no ext.)[_postfix]`

    otherwise the relative path with respect to `data_root_dir` will be inserted.

    Args:
        postfix: output name's postfix
        input_file_name: path to the input image file.
        folder_path: path for the output file
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. This is used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names.
    """

    # get the filename and directory
    filedir, filename = os.path.split(input_file_name)
    # remove extension
    filename, ext = os.path.splitext(filename)
    if ext == ".gz":
        filename, ext = os.path.splitext(filename)
    # use data_root_dir to find relative path to file
    filedir_rel_path = ""
    if data_root_dir and filedir:
        filedir_rel_path = os.path.relpath(filedir, data_root_dir)

    # sub-folder path will be original name without the extension
    # subfolder_path = os.path.join(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if postfix:
        # add the sub-folder plus the postfix name to become the file basename in the output path
        output = os.path.join(folder_path, filename + "_" + postfix)
    else:
        output = os.path.join(folder_path, filename)
    return os.path.abspath(output)

def pipeline():
    logging.basicConfig(level=logging.INFO)
    data_root = 'ts'
    if data_root == 'cc':
        data_root = '/scratch/authorid/dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/'
        model_file = '/scratch/authorid/BRATS_IDH/log/whole_balanced_0918/fold_1_epoch_46.pth'
    elif data_root == 'ts':
        data_root = '/local-scratch/authorid/dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/'
        # model_file = '/local-scratch/authorid/BRATS_IDH/log/heatmap_model/fold_1_epoch_0.pth'
        model_file = '/local-scratch/authorid/BRATS_IDH/log/0122_pattern_10epoch/fold_1_epoch_9.pth'
    else:
        logging.warning('>> Error, no data_root specified!')
    fold =1
    valloader = load_dataloader(fold, data_root)
    logging.info('\nDONE loading data')

    logging.info('\nLoading the model and move to GPU\n')
    model = load_model(model_path= model_file, model_architecture=GeneNet)
    logging.info('\nDONE Loading the model\n')
    method_list = ['Gradient','GradientShap', 'GuidedBackProp', 'GuidedGradCAM', 'IntegratedGradients',\
                   'DeepLift', 'DeepLiftShap', 'InputXGradient', 'Deconvolution', 'Occlusion',\
                   'SmoothGrad'
                   ]
    # TODO: bugs in 'GradientShap',, 'IntegratedGradients' GPU OOM
    heatmap = None
    with torch.no_grad():
        for data in valloader:
            images_val, labels_val, bratsID = data['image'], data['gt'].numpy(), data['bratsID']
            images_val = images_val.to(device)
            # logging.info(images_val.shape)
            pred = model.forward(images_val).cpu().detach().numpy()
            pred = int(np.argmax(pred))
            postfix = str(bratsID[0])+'_gt'+ str(int(labels_val[0])) + '_pred' + str(pred)
            heatmap = generate_heatmap(images_val, model, postfix = postfix, target = pred, method_list = method_list)
            break
    return heatmap



def generate_heatmap_featurefusion(input, model, target, dataIDs, method_list, device, last_layer = None, gt = None, pred = None, save_dir = None): #, save= False, postfix = None):
    '''
    Variation of generate_heatmap (modality fused at input level).
    Applicable to MRNet where modality fused at feature level.
    Generate heatmpas according to the data, model, and method.
    Since CPU OOM issue, generate heatmap and save as:
    save_dir (pass as resume arg)  / exp_label (the experiment + fold #) /
        - heatmap / xaimethod_bratsID.pkl
        - input / bratsID.pkl
        - record_hm_generator.csv file: writerow - bratsID, xaimethod, pred, gt, time

    If gt is given, and it's different from prediction (target), generate additional hm for gt (the not predicted one)

    Main function to include the 10+ post-hoc XAI methods to generate heatmaps for individual input
    Given a instance and model, generate saliency map
    put data and model in cpu, to avoid OOM.
    :param:
        method_list: a name list of the heatmap method
    :return attributions: saliency map of the instance on different modalities
    '''
    dataIDs = dataIDs[0]
    # print(pred, 'pred') #[[0.01884678]] pred
    pred = pred[0][0]
    gt = gt[0][0]
    # print(dataIDs, pred, gt)
    # device = "cpu"
    new_input = []
    squeezed_input = [] # list of modality, each mod is of [D, W, H]
    for i, img in enumerate(input):
        img.requires_grad = True
        new_input.append(img.to(device))
        single_mod_vol = img[:,:,0,:,:] #torch.Size([25, 224, 224])
        squeezed_input.append(torch.squeeze(single_mod_vol))
    # superpixel mask for perturbation methods
    feature_mask, feature_mask_batch = superpixel_mask_featurefusion(squeezed_input)
    # print("feature_mask,", feature_mask[0].shape) # torch.Size([25, 3, 224, 224])
    assert torch.all(feature_mask[0][:, 0, :, :] == feature_mask[0][:, 1, :, :]) and torch.all(
        feature_mask[0][:, 1, :, :] == feature_mask[0][:, 2, :, :]), logging.info(
        "feature_mask 3 channeals are not equal values")

    input = tuple(new_input)
    model.zero_grad()
    if target:
        target = target.to(device)
    model.to(device)
    # print(model.state_dict().keys())


    # generate baseline value of shape input_dim, for occlusion and other methods
    baseline_tuple = []
    for i in range(len(input)): # 3 modalities torch.Size([1, 44, 3, 224, 224])
        assert input[i].shape[0] == 1, logging.info('input batch size is not 1, {}'.format(input[i].shape))
        modality_img = torch.squeeze(input[i]) # batch b = 1, torch.Size([44, 3, 224, 224])
        assert torch.all(modality_img[:, 0, :, :] == modality_img[:, 1, :, :]) and torch.all(modality_img[:, 1, :, :]  == modality_img[:, 2, :, :]), logging.info("modality image 3 channeals are not equal values")
        modality_3d_img_size = (modality_img.shape[0], modality_img.shape[2], modality_img.shape[3])
        modality_mean = torch.mean(modality_img)
        modality_std = torch.std(modality_img)
        # print("mean std", modality_mean, modality_std)
        modality_baseline = torch.normal(mean=modality_mean, std=modality_std,
                                     size=modality_3d_img_size)  # generate random fill-in for each data in batch
        baseline_normal = ndimage.gaussian_filter(modality_baseline, sigma=[int(modality_img.shape[0]/4), 20, 20])
        # numpy (44, 224, 224) to (44, 3, 224, 224)
        baseline_normal = np.stack((baseline_normal,)*3, axis=1)
        # baseline_normal = np.expand_dims(baseline_normal, axis = 0) # baseline should be the same shape as input. (1, 44, 3, 224, 224)
        baseline_normal = torch.from_numpy(baseline_normal).to(device)
        baseline_tuple.append(baseline_normal)
    # print('fill baseline', baseline_normal.shape)
    baseline_tuple = tuple(baseline_tuple)
    baseline_req_gd = []
    for i in range(len(input)):
        bs = torch.zeros(input[i].shape)
        bs.requires_grad = True
        baseline_req_gd.append(bs)
    baseline_req_gd = tuple(baseline_req_gd)
        # stride and window shape must be provided for each input tensor
    window_size = 11
    depth_window_size = 5
    stride_step = 5
    tuple_sliding_window_shapes = []
    tuple_strides = []
    for img in input: # torch.Size([1, 25, 3, 224, 224])
        # sliding_window_shapes = [window_size for i in range(input_dim - 2)]
        # sliding_window_shapes.insert(0, 1)
        sliding_window_shapes = tuple([depth_window_size, 3, window_size, window_size])
        tuple_sliding_window_shapes.append(sliding_window_shapes)
        # strides = [stride_step for i in range(input_dim - 2)]
        # strides.insert(0, 1)
        strides = tuple([depth_window_size, 3, stride_step, stride_step])
        tuple_strides.append(strides)
    sliding_window_shapes = tuple(tuple_sliding_window_shapes)
    strides = tuple(tuple_strides)
    # print('input shape', input[0].shape, sliding_window_shapes, strides)

    # print(model)
    # print(model.state_dict().keys())
    # Select algorithm to instantiate and apply
    # target layer for gradcam
    # conv_layer = getattr(model, last_layer)
    # print(getattr(model, "model.axial_net.features.10"))
    # conv_layer = model.features.denseblock4.denselayer16.layers.conv2 # todo change with differetn model
    conv_layer = None #axial_net.features.3   #model.axial_net.features.10
    # print(axial_conv_layer)
    # Setup XAI methods
    saliency_funcs = {'Gradient': Saliency(model).attribute,
                        'GradientShap': GradientShap(model).attribute,
                        'GuidedBackProp': GuidedBackprop(model).attribute,
                        'GuidedGradCAM': GuidedGradCam(model, conv_layer).attribute, #todo
                        'GradCAM': LayerGradCam(model, conv_layer).attribute,
                        'IntegratedGradients': IntegratedGradients(model).attribute,
                      'DeepLift':DeepLift(model).attribute,
                      'DeepLiftShap': DeepLiftShap(model).attribute,
                      'InputXGradient': InputXGradient(model).attribute,
                      'Deconvolution': Deconvolution(model).attribute,
                      'Occlusion': Occlusion(model).attribute,
                      'SmoothGrad': NoiseTunnel(Saliency(model)).attribute,
                      'ShapleyValueSampling': ShapleyValueSampling(model).attribute,
                      # 'ShapleyValueSamplingModUnify': ShapleyValueSampling(model).attribute,
                      # 'LimeBase': LimeBase(model).attribute,
                      'KernelShap': KernelShap(model).attribute,
                      'FeatureAblation': FeatureAblation(model).attribute,
                      'FeaturePermutation': FeaturePermutation(model).attribute,
                      'Lime': Lime(model, interpretable_model = SkLearnLinearRegression()).attribute
                      }

    # set up params for each saliency method
    squeezed_inputs = tuple([torch.squeeze(ipt) for ipt in input])
    if target:
        gen_feed_dict = {'inputs': input, 'target': pred}  # need to set batch_size =1 in config file
    else:
        gen_feed_dict = {'inputs': input}
    feed_dict_w_baseline = dict({"baselines": baseline_tuple}, **gen_feed_dict)
    pertubation_args = dict({"feature_mask": feature_mask}, **gen_feed_dict) #**feed_dict_w_baseline)
    saliency_params = {'Gradient': dict({'abs': False}, **gen_feed_dict), # get pos & neg value of gradients
                       'GradientShap':{'inputs':input, 'n_samples':1,'baselines':tuple([torch.zeros(ipt.shape) for ipt in input])}, #feed_dict_w_baseline,
                       'GuidedBackProp':  gen_feed_dict,
                       'GuidedGradCAM': gen_feed_dict,
                       'GradCAM': gen_feed_dict,
                       'IntegratedGradients': {'inputs': squeezed_inputs},
                       'DeepLift': {'inputs': input, 'baselines': baseline_req_gd}, #gen_feed_dict, #feed_dict_w_baseline,
                       'DeepLiftShap': feed_dict_w_baseline,
                       'InputXGradient': gen_feed_dict,
                       'Deconvolution': gen_feed_dict,
                       'Occlusion': dict({'baselines': baseline_tuple, 'sliding_window_shapes': sliding_window_shapes, 'strides': strides}, **gen_feed_dict),
                       'SmoothGrad': dict({'nt_type': 'smoothgrad', 'nt_samples':10 , 'inputs': squeezed_inputs}),
                       'ShapleyValueSampling': pertubation_args,
                       # "ShapleyValueSamplingModUnify": dict({"feature_mask": mod_unify_feature_mask, "perturbations_per_eval": 5}, **gen_feed_dict),
                       # 'LimeBase': gen_feed_dict,
                       'KernelShap': dict({"n_samples": 300, "feature_mask": feature_mask, "return_input_shape": True}, **feed_dict_w_baseline),
                       'FeatureAblation': pertubation_args,
                       'FeaturePermutation': dict({"feature_mask": feature_mask_batch, "perturbations_per_eval": 5} , **gen_feed_dict),
                       'Lime': dict({"feature_mask": feature_mask, "return_input_shape": True, "n_samples": 500}, **gen_feed_dict)
                       }
    assert saliency_funcs.keys() == saliency_params.keys()
    # output_heatmaps = []
    # time_record = []
    batch_size = 1# len(gt)

    # prepare dir to save the input and generated hms
    hm_dir = save_dir / 'heatmap'
    hm_dir.mkdir(parents=True, exist_ok=True)
    input_dir = save_dir /'input'
    input_dir.mkdir(parents=True, exist_ok=True)
    # create csv file
    csv_filename = save_dir / 'record_{}.csv'.format(str(save_dir.name))
    fnames = ['Data_ID', 'XAI_Method', 'HM_target', 'Prediction', 'GT', 'Predicted_Correct',
              'Time_spent', 'File_name', 'max_mem', 'rss', 'vms', 'share_mem']
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
        if not file_exists:
            csv_writer.writeheader()



    for m_idx, name in enumerate(method_list):
        if name in saliency_funcs:
            with open(csv_filename, 'a', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fnames)
                # check if the hm: heatmap / xaimethod_bratsID.pkl exist
                hms_exist = []
                # for idx, d_id in enumerate(dataIDs):
                hm_filename = generate_hm_filename(dataIDs, name, target=None, pred = pred, gt = gt)
                exist = Path(hm_dir/ hm_filename).is_file() #check_file_exist(hm_dir, hm_filename)
                hms_exist.append(exist)
                # print(hm_filename)
                # must meet all files exist
                if all(hms_exist):
                    logging.info("HM {} [{}] exist. Pass".format(dataIDs, name))
                    continue
                else:
                    # generate and save hm and input
                    params = saliency_params[name]
                    start_time = time.time()
                    # print(name, 'input size:', input.shape)
                    attribution = saliency_funcs[name](**params)
                    # print(len(attribution), attribution[0].shape) # 3 torch.Size([1, 25, 3, 224, 224])
                    time_cost = (time.time() - start_time) / batch_size
                    logging.info('\nXAI method: {} costed time: {}\n'.format(name, time_cost))
                    # time_record.append(time_cost)
                    # confirm the shape matchs with input
                    if name != "GradCAM" and name != "SmoothGrad" and name!= "IntegratedGradients":
                        assert attribution[0].shape == input[0].shape, "Heatmap original size doesn't match with the input"
                    # Generate hms
                    rss_before, vms_before, shared_before = get_process_memory()
                    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    detached_attribution = []
                    three_mod_not_all_zeros = []
                    for a in attribution:
                        reshaped_a = a.cpu().detach().numpy().squeeze() #  (25, 3, 224, 224) 3 is not the same.
                        # print(reshaped_a[0][0].shape, reshaped_a[0][0] == reshaped_a[0][1]) #
                        # print('reshaped a', reshaped_a.shape)
                        detached_attribution.append(reshaped_a)
                        not_all_zeros = np.any(reshaped_a)
                        three_mod_not_all_zeros.append(not_all_zeros)
                    # print(three_mod_not_all_zeros, )
                    rss_after, vms_after, shared_after = get_process_memory()
                    delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
                    rss = rss_after - rss_before
                    vms = vms_after - vms_before
                    share_mem = shared_after - shared_before # in byte
                    print(delta_mem, format_bytes(rss), format_bytes(vms), format_bytes(share_mem))

                    # save the generated hms
                    # for idx, d_id in enumerate(dataIDs):
                    # hm_filename = generate_hm_filename(dataIDs, name, target=None, pred=pred, gt=gt)
                        # save heatmap as pkl file

                        # if len(attribution.shape) == len(input.shape): # batch_size > 1
                        #     # print('attribution shape bs >1', attribution.shape, attribution[idx].shape, input.shape)
                        #     attr_to_save = attribution[idx]
                        # else:
                        #     attr_to_save = attribution
                            # print('attribution shape', attribution.shape, attribution[idx].shape, input.shape)
                        # when max == 0,
                        # sanity check that the hm is valid one, not all 0
                        # if np.absolute(attr_to_save).max() == 0:

                    if not np.any(three_mod_not_all_zeros):
                        logging.info("\nWARNING: XAI method {} are all zero for data {}, hm shape {}.".format(name, dataIDs, detached_attribution[0].shape))
                        continue
                    else:
                        pickle.dump(detached_attribution, open(os.path.join(hm_dir, hm_filename), 'wb'))
                        # save record
                        gt = int(gt)
                        csv_record = {'Data_ID': dataIDs, 'XAI_Method': name, 'Prediction': pred,\
                                      'GT': gt, 'Predicted_Correct': int(((pred>0.5) == gt)), 'Time_spent': time_cost , 'File_name': hm_filename, 'max_mem': delta_mem, 'rss': rss, 'vms': vms, 'share_mem': share_mem}
                        logging.info(csv_record)
                        csv_writer.writerow(csv_record)
                        # save input
                        # input_filename = '{}.pkl'.format(dataIDs)
                        # if Path(input_dir/input_filename).is_file():
                        #     pass
                        # else:
                        #     logging.info('save input {} : {}\n'.format(dataIDs, input.shape))
                        #     pickle.dump(input, open(os.path.join(input_dir, input_filename), 'wb'))
                    del attribution
                    gc.collect()

                    # logging.info('Attribution shape: {}'.format(attribution.shape))
                    # resample to original size of d_size, h_size, w_size = 155, 240, 240
                    # ori_size = input.shape[2:]
                    # attribution = Resize(spatial_size=ori_size)(attribution) # calculate sim don't need to resize to ori size, only needed when compare gt seg, and for vis
                    # output_heatmaps.append(attribution)
                # save
                # if save:
                #     channel_dict = {0: 't1', 1: 't1ce', 2: 't2', 3: 'flair'}
                #     dateTimeObj = datetime.now()
                #     file_name = postfix + '_' + name + '_' + dateTimeObj.strftime("%Y%m%d_%H%M")
                #     output_dir = '../heatmap/' + file_name
                #     output_postfix = 'heatmap'
                #     # saver = NiftiSaver(output_dir='../heatmap/'+file_name, output_postfix='heatmap', resample = True)
                #     for channel in range(attribution.shape[0]):
                #         # print(attribution[channel].shape, channel_dict[channel])
                #         filename = create_file_basename(output_postfix, channel_dict[channel], output_dir, data_root_dir=output_dir)
                #         # print(filename)
                #         write_nifti(attribution[channel], file_name=filename) # save as HWD
                #         # saver.save(attribution[channel], {'filename_or_obj':channel_dict[channel], 'spatial_shape':ori_size})

        else:
            print('{} not in saliency function list {}'.format(name, saliency_params.keys()))
    gc.collect()

    # return np.stack(output_heatmaps), time_record

def superpixel_mask_featurefusion(modality_list):
    """
    Variation of superpixel_mask function. Applicable to MRNet
    3D image level, generate feature mask, by 2D slice stack
    SLIC - K-Means based image segmentation
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html
    :param input: Input image, which can be 2D or 3D, and grayscale or multichannel (see multichannel parameter).
    Input image must either be NaN-free or the NaN’s must be masked out
    Input: list of modaltieis, each modality is 3D torch image of [D, W, H]
    :return:
    """
    mod_segments = []
    prev_n_label = 0
    feature_mask_batch = [] # shape[0] == 1
    for c, single_modality in enumerate(modality_list):
        single_modality = single_modality.cpu().numpy()
        segments = []
        bg = np.where(single_modality == 0, 0, 1)  # mask = 0 is background
        # print(single_modality.shape)
        n_segments = 30
        segments_slic = slic(single_modality, n_segments= n_segments, compactness=0.1, sigma=2, multichannel=False,
                             start_label=1, #min_size_factor=0.8, max_size_factor=3, enforce_connectivity = True,
                             mask = bg)  # [H, W, D]
        # add mask label indicies, so that not to be correlated with other modalities.
        if c>0:
            # print("prev_n_label", prev_n_label, np.unique(segments_slic))
            segments_slic = np.where(segments_slic == 0, 0, segments_slic+prev_n_label)
        prev_n_label += len(np.unique(segments_slic)) - 1
        segments_slic = np.stack((segments_slic,) * 3, axis=1) # numpy (44, 224, 224) to (44, 3, 224, 224)
        segments_slic_batch = np.expand_dims(segments_slic, axis=0) # (1, 44, 3, 224, 224)
        mod_segments.append(torch.from_numpy(segments_slic))
        feature_mask_batch.append(torch.from_numpy(segments_slic_batch))
            # segments = np.stack(segments)  # [C, H, W, D]
        # segments = np.expand_dims(segments,0) # [1, C, H, W, D]
    # print(img_single_batch.shape, channel_last_img.shape, segments_slic.shape, segments.shape)
    # print(segments.shape,'segment.shape')
    #     bs_segments.append(segments)
    # bs_segments = np.concatenate(bs_segments)
    # print(bs_segments.shape)
    return tuple(mod_segments), tuple(feature_mask_batch)

if __name__ == "__main__":
    pipeline()
