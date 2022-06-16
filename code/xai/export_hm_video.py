

import numpy as np
from matplotlib import pyplot as plt
# from  skimage import draw
import sys
import os
from pathlib import Path
import pandas as pd
from xai.heatmap_utlis import *
import pickle
# machine = 'solar'
machine = 'ts'

if machine == 'solar':
    root = '/project/labname-lab/'
    save_dir = Path(root + 'authorid/BRATS_IDH/log/BRATS_HGG/0429_114257_fold_1/get_hm_fold_1')
else:
    root = '/local-scratch/'
    # save_dir = Path(root+'authorid/trained_model/BRATS20_HGG/heatmaps/get_hm_fold_1')
    save_dir = Path(root+"authorid/trained_model/BRATS20_HGG/heatmaps/fold_1/get_hm_fold_1/heatmap/")
# save_dir = Path('/local-scratch/authorid/BRATS_IDH/log/BRATS_HGG/0503_185306_fold_1/get_hm_fold_1')
# save_dir = Path('/local-scratch/authorid/BRATS_IDH/log/BRATS_HGG/0504_143959_fold_1/get_hm_fold_1')

method_list = ["Occlusion", "FeatureAblation", "KernelShap", "ShapleyValueSampling",
            "Gradient", "GuidedBackProp", "GuidedGradCAM",
                   "DeepLift", "InputXGradient", "Deconvolution",
                   "SmoothGrad", "IntegratedGradients","GradientShap", "FeaturePermutation", "Lime", "GradCAM"]

method_list = ['FeatureAblation']
path = root+"authorid/dld_data/brats2020/MICCAI_BraTS2020_TrainingData/all"

# select MRIs for doctor user study
# csv = pd.read_csv(root+"authorid/trained_model/BRATS20_HGG/test/cv_result_fold_1.csv")
csv = pd.read_csv(root+'authorid/dld_data/brats2020/MICCAI_BraTS2020_TrainingData/Grade/test_fold_1.csv')
# hgg_r = csv.loc[(csv['gt']==1) & (csv['pred']==1)]
# hgg_w = csv.loc[(csv['gt']==1) & (csv['pred']==0)]
# lgg_w = csv.loc[(csv['gt']==0) & (csv['pred']==1)]
# lgg_r = csv.loc[(csv['gt']==0) & (csv['pred']==0)]
# print(hgg_r.shape, hgg_w.shape, lgg_r.shape, lgg_w.shape)


video_list =  ['BraTS20_Training_070']#list(csv["BraTS_2020_subject_ID"])
# video_list.remove('BraTS20_Training_001')
# video_list.remove('BraTS20_Training_003')
# video_list.remove('BraTS20_Training_004')
# video_list.remove('BraTS20_Training_005')
# video_list.remove('BraTS20_Training_134')
# video_list.remove('BraTS20_Training_270')
# video_list.remove('BraTS20_Training_277')
# video_list.remove('BraTS20_Training_070')
# video_list.remove('BraTS20_Training_331')
# video_list.remove('BraTS20_Training_339')
# video_list.remove('BraTS20_Training_343')
# video_list.remove('BraTS20_Training_346')
# video_list.remove('BraTS20_Training_356')
# video_list.remove('BraTS20_Training_359')
# video_list.remove('BraTS20_Training_362')
#
# video_list.remove('BraTS20_Training_019')
# video_list.remove('BraTS20_Training_025')
# video_list.remove('BraTS20_Training_033')
# video_list.remove('BraTS20_Training_042')
# video_list.remove('BraTS20_Training_048')
# video_list.remove('BraTS20_Training_050')

# video_list = video_list[::-1]
# video_list = ['BraTS20_Training_070', 'BraTS20_Training_042', 'BraTS20_Training_064']+['BraTS20_Training_277', 'BraTS20_Training_260', 'BraTS20_Training_308'] +  list(lgg_w["dataID"]) + list(hgg_w["dataID"])
# video_list = ['BraTS20_Training_070', 'BraTS20_Training_270',  'BraTS20_Training_277', 'BraTS20_Training_134']
print(len(video_list), video_list)


# save mri to video
# for i, bratsId in enumerate(video_list):
#     mri = load_mri(path, bratsId, get_seg = False)
#     mri_array = np.array(mri)
#     new_hm_dict = {}
#     kwags_dict = {"mri_lst": mri,
#                   "heatmap_lst": np.expand_dims(np.zeros(mri_array.shape), 0),
#                   "bratsid": bratsId,
#                   "hm_names": [],
#                   "figsize": (100, 10),
#                   "mri_only": True
#                  }
# #     MultipleMRIViewer(**kwags_dict)
#     generate_mrivideo("MRI-{}".format(bratsId), subfolder = None, exist_ok= True, ffmpeg = False, dir = root+'authorid/trained_model/BRATS20_HGG/heatmaps/get_hm_fold_1/videos',  **kwags_dict)
#     print("Video exported: MRI-{}".format(bratsId))

save_hm = True


if save_hm:
    # save to video
    for i, bratsId in enumerate(video_list):
        mri = load_mri(path, bratsId, get_seg = False)
        for method in method_list:
            hm_fns = Path(save_dir).rglob("{}-{}*.pkl".format(bratsId, method))
            for hm_fn in hm_fns:
                hm = pickle.load(open(hm_fn, "rb"))
                hm_name = Path(hm_fn).name.split('.')[0]
                print(hm_name)
                if hm_name=="BraTS20_Training_270-SmoothGrad-T0":
                    continue

                post_hm = postprocess_heatmaps(hm, no_neg = True)

                print(method, mri[0].shape, hm.shape, post_hm.shape)
                kwags_dict = {"mri_lst": mri,
                          "heatmap_lst": np.expand_dims(post_hm, 0),
                          "bratsid": bratsId,
                          "hm_names": [method],
                          "figsize": (100, 10),
                        "title_prefix": method
                         }
        #         MultipleMRIViewer(**kwags_dict)
                generate_mrivideo(hm_name, subfolder = None, ffmpeg=True, dir = root+'authorid/trained_model/BRATS20_HGG/heatmaps/get_hm_fold_1/videos', **kwags_dict)
                print("Video exported: {}".format(hm_name))
# if save_hm:
#     # save to video
#     for i, bratsId in enumerate(video_list):
#         mri = load_mri(path, bratsId, get_seg = False)
#         hm_dict, _, data_record = get_heatmaps(save_dir, bratsId, by_data= True, hm_as_array= False, return_mri= False)
#         print(len(hm_dict.keys()))#, hm_dict.keys())
#         new_hm_dict = {}
#         for method, hm in hm_dict.items():
#             if method in method_list:
#                 post_hm = postprocess_heatmaps(hm)
#                 print(method, hm.shape, post_hm.shape)
#                 new_hm_dict[method] = post_hm
#         for j, method in enumerate(new_hm_dict):
#             kwags_dict = {"mri_lst": mri,
#                       "heatmap_lst": np.expand_dims(new_hm_dict[method], 0),
#                       "bratsid": bratsId,
#                       "hm_names": [method],
#                       "figsize": (100, 10),
#                     "title_prefix": method
#                      }
#     #         MultipleMRIViewer(**kwags_dict)
#             generate_mrivideo("{}-{}".format(method, bratsId), subfolder = None, ffmpeg=False, dir = root+'authorid/trained_model/BRATS20_HGG/heatmaps/get_hm_fold_1/videos', **kwags_dict)
#             print("Video exported: {}-{}".format(method, bratsId))

