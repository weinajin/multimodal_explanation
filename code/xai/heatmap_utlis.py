# load seg map, make sure they're aligned with heatmap. Remove neg value in heatmap for fair comparison.

import nibabel
import os
import logging
import glob
import numpy as np
import pickle
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import  segmentation

# from matplotlib.colors import Normalize
import matplotlib.cm as cm
import torch
from captum.attr import LayerAttribution
from numpy import ndarray
import warnings
from typing import Union


import scipy.ndimage as ndimage
from sklearn.metrics import jaccard_score
from monai.transforms import Resize

# for calculate heatmap similarity
from scipy.stats import spearmanr as spr
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
from skimage.metrics import structural_similarity as ssim
# https://scikit-image.org/docs/stable/api/skimage.metrics.html?highlight=structural_similarity#skimage.metrics.structural_similarity
from scipy.stats import pearsonr
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
from skimage.feature import hog
from skimage.morphology import convex_hull_image, binary_dilation
# https://scikit-image.org/docs/stable/api/skimage.feature.html?highlight=feature#skimage.feature.hog
# from sklearn.metrics import normalized_mutual_info_score
import ipywidgets as widgets
from pathlib import Path
import pandas as pd

import subprocess
import imageio
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix
import csv
import itertools, math
import copy
modality = ['t1', 't1ce', 't2', 'flair']
def build_parser():
    '''
    Obsolete function
    :return:
    '''
    parser = ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, help="cross validation fold number, [1-5]")
    parser.add_argument('--model_file', type=str,
                        default='/local-scratch/authorid/BRATS_IDH/log/model_0920/fold_1_epoch_46.pth',
                        help='path of the model')
    parser.add_argument('--dataset', type=str, default='BRATS_IDH', help='Input dataset name')
    parser.add_argument(
        '--data_root',
        type=str,
        default='cc',
        help='option: cc or ts, path of the brast dataset and labels')
    # parser.add_argument(
    #     '--log',
    #     # dest='logging',
    #     action='store_true')
    # option about heatmap
    parser.add_argument(
        '--normal_masks',
        # dest='normal_masks',
        action='store_true')
    return parser

class ImageSliceViewer3D:
    """
    Code base: https://github.com/mohakpatel/ImageSliceViewer3D
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks.

    User can interactively change the slice plane selection for the image and
    the slice plane being viewed.

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html

    """

    def __init__(self, mri_lst, heatmap_lst, bratsid, alpha=.5, figsize=(10, 10), cmap='gray', heatmap_cmap='bwr', modality = ['t1', 't1ce', 't2', 'flair']):
        import ipywidgets as widgets

        self.mri_lst = mri_lst
        self.heatmap_lst = heatmap_lst
        self.figsize = figsize
        self.cmap = cmap
        if np.array(heatmap_lst).min() == 0.0: # for hm with [0,1] range
            self.heatmap_cmap = "Reds"
        else:
            self.heatmap_cmap = heatmap_cmap
        self.alpha = alpha
        self.hm_on = False
        self.mri_on = True
        self.bratsid = bratsid
        self.modality = modality
        # Call to select slice plane
        widgets.interact(self.view_selection, view=widgets.RadioButtons(
            options=['Axial', 'Saggital', 'Coronal'], value='Axial',
            description='MRI view:', disabled=False,
            style={'description_width': 'initial'}))

        #         # Call to adjust heatmap alpha
        #         widgets.interact(self.heatmap_alpha, alpha=widgets.IntSlider(min=0, max=1, step=0.1, continuous_update=False,
        #             description='Adjust heatmap opaque'))

        # Call to turn on/off heatmap
        widgets.interact(self.heatmap_switch, switch=widgets.Checkbox(
            value=True,
            description='Show heatmap',
            disabled=False))
        # Call to turn on/off mri
        widgets.interact(self.mri_switch, switch=widgets.Checkbox(
            value=True,
            description='Show MRI',
            disabled=False))

    def heatmap_switch(self, switch):
        self.hm_on = switch

    def mri_switch(self, switch):
        self.mri_on = switch
        if self.hm_on == False and self.mri_on == False:
            self.mri_on = True

    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"Saggital": [1, 2, 0], "Coronal": [2, 0, 1], "Axial": [0, 1, 2]}
        for i in range(len(self.modality)):
            mri = np.transpose(self.mri_lst[i], orient[view])
            hm = np.transpose(self.heatmap_lst[i], orient[view])
            self.mri_lst[i] = mri
            self.heatmap_lst[i] = hm
            maxZ = mri.shape[2] - 1

        # Call to view a slice within the selected slice plane
        widgets.interact(self.plot_slice,
                         z=widgets.IntSlider(min=0, max=maxZ, step=1, continuous_update=True,
                                             description='Slices:'))

    #     def heatmap_alpha(self, alpha):
    #         widgets.interact(self.plot_slice,
    #             alpha=widgets.IntSlider(min=0, max=1, step=0.1, continuous_update=False,
    #             description='Adjust heatmap opaque'))

    def plot_slice(self, z):

        columns = 2
        rows = 2
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        for i in range(1, columns * rows + 1):
            mri = self.mri_lst[i - 1]
            hm = self.heatmap_lst[i - 1]
            subplot = self.fig.add_subplot(rows, columns, i)
            if self.hm_on == False and self.mri_on == False:
                self.mri_on = True
            if self.mri_on:
                subplot.imshow(mri[:, :, z], cmap=plt.get_cmap(self.cmap),
                               vmin=np.min(mri), vmax=np.max(mri))
            if self.hm_on:
                #             hm_mask = np.ma.masked_array(mri*hm > 0, hm)
                subplot.imshow(hm[:, :, z], cmap=plt.get_cmap(self.heatmap_cmap),
                               alpha=self.alpha, vmin=-1, vmax=1)
            #             plt.imshow(hm_mask[:,:,z], cmap='gray', alpha=0.5)
            subplot.set_title(self.modality[i - 1].upper() + ' ' + self.bratsid)

class MultipleMRIViewer(ImageSliceViewer3D):
    def __init__(self, mri_lst, heatmap_lst, bratsid, hm_names, seg = None, alpha=.5, figsize=(20, 10), cmap='gray', heatmap_cmap='bwr', modality = ['t1', 't1ce', 't2', 'flair'], title_prefix= "", outlier_perc = 1, mri_only= False):
        # super().__init__(mri_lst, heatmap_lst, bratsid, alpha=alpha, figsize=figsize, cmap=cmap, heatmap_cmap=heatmap_cmap)
        self.mri_lst = mri_lst
        self.mri_only = mri_only
        self.heatmap_lst = heatmap_lst
        self.seg = seg
        self.contours = []
        # draw contours for each seg map
        if self.seg is not None:
            for i in np.unique(self.seg):
                if i != 0:
                    contour = segmentation.find_boundaries(seg==i)
                    self.contours.append(contour)
        self.figsize = figsize
        self.cmap = cmap
        if np.array(heatmap_lst).min() == 0.0: # for hm with [0,1] range
            self.heatmap_cmap = "bwr"
        else:
            self.heatmap_cmap = heatmap_cmap
        self.outlier_perc = outlier_perc # heatmap normalization parameter
        # self.c_cmap = copy.copy(cm.get_cmap(heatmap_cmap))
        # self.c_cmap.set_under('k', alpha=0)
        #         self.v = [np.min(volume), np.max(volume)]
        self.alpha = alpha
        self.hm_on = False
        self.mri_on = True
        self.bratsid = bratsid
        self.modality = modality
        self.title_prefix = title_prefix
        if not self.mri_only:
            assert heatmap_lst.shape[0] == len(hm_names), "ERROR! Heatmap number not aligned!"
        self.nm_hm = len(hm_names)
        self.hm_names = hm_names
        if self.nm_hm ==1:
            self.hm_mix_vis = True
        else:
            self.hm_mix_vis = False
        # Call to select slice plane
        widgets.interact(self.view_selection, view=widgets.RadioButtons(
            options=['Axial', 'Saggital', 'Coronal'], value='Axial',
            description='MRI view:', disabled=False,
            style={'description_width': 'initial'}))

        #         # Call to adjust heatmap alpha
        #         widgets.interact(self.heatmap_alpha, alpha=widgets.IntSlider(min=0, max=1, step=0.1, continuous_update=False,
        #             description='Adjust heatmap opaque'))

        # Call to turn on/off heatmap
        widgets.interact(self.heatmap_switch, switch=widgets.Checkbox(
            value=True,
            description='Show heatmap',
            disabled=False))
        # Call to turn on/off mri
        widgets.interact(self.mri_switch, switch=widgets.Checkbox(
            value=True,
            description='Show MRI',
            disabled=False))


    def view_selection(self, view):

        # Transpose the volume to orient according to the slice plane selection
        orient = {"Saggital": [1, 2, 0], "Coronal": [2, 0, 1], "Axial": [0, 1, 2]}
        for i in range(len(self.modality)):
            mri = np.transpose(self.mri_lst[i], orient[view])
            self.mri_lst[i] = mri
            maxZ = mri.shape[2] - 1
        if not self.mri_only:
            heatmap_lst = []
            for m in range(len(self.hm_names)):
                hms = []
                for i in range(len(self.modality)):
                    hm = np.transpose(self.heatmap_lst[m, i], orient[view])
                    hms.append(hm)
                heatmap_lst.append(np.stack(hms))

            self.heatmap_lst = np.stack(heatmap_lst)

        # Call to view a slice within the selected slice plane
        widgets.interact(self.plot_slice,
                         z=widgets.IntSlider(min=0, max=maxZ, step=1, continuous_update=True,
                                             description='Slices:'))
    # def map_alpha(self, img_array, z):
    #     # return an array of (H, W, RGB channel)
    #     abs_array = np.absolute(img_array[:,:, z])
    #     abs_scaled = scale01(abs_array)
    #     # abs_scaled = np.stack([abs_scaled, abs_scaled,abs_scaled], axis = -1 )
    #     return abs_scaled

    def plot_slice(self, z):
        """
        plot MRI on first row, plot MRI + hm on sequential rows, with the hm and mri can be turn on/off.
        MRI for each col is the same, row = modality number
        """
        fontsize = self.figsize[0]
        method_names = ['MRI'] + self.hm_names
        columns = len(self.modality)
        if self.hm_mix_vis:  # add extra row below to show hm alone
            rows =3
        elif not self.mri_only:
            rows = self.nm_hm + 1
        else:
            rows = 1
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=(self.figsize[0], int(self.figsize[0]/columns * (rows+0.4))))
        # print((self.figsize[0], int(self.figsize[0]/columns * (rows+0.4))), rows, columns)
        ax_placeholder = self.fig.add_subplot(rows + 1, 1, rows + 1)
        ax_placeholder.axis("off")
        splots = [ax_placeholder]
        for i in range(1, columns * rows + 1):
            subplot = self.fig.add_subplot(rows, columns, i)
            mri = self.mri_lst[(i - 1) % 4]

            if i <= columns:


                subplot.imshow(mri[:, :, z], cmap=plt.get_cmap(self.cmap),
                               vmin=np.min(mri), vmax=np.max(mri))
                subplot.set_title("{}  {}\n".format(method_names[(i-1)//columns], self.modality[(i-1)%columns].upper() ), y=0.85, color = 'white', fontsize=fontsize)

            else:
                if self.mri_only:
                    break
                if self.hm_mix_vis:
                    hm = self.heatmap_lst[0]
                    scaled_hm = hm #normalize_scale(hm, self.outlier_perc)
                    scaled_hm_mod = scaled_hm[(i - 1) % columns]
                    if i > columns *2 :
                        subplot.imshow(mri[:, :, z], cmap=plt.get_cmap(self.cmap),
                                       vmin=np.min(mri), vmax=np.max(mri))
                        subplot.imshow(scaled_hm_mod[:, :, z],  # cmap= plt.get_cmap(self.heatmap_cmap),  alpha=self.alpha,
                                       cmap=plt.get_cmap(self.heatmap_cmap), alpha=np.absolute(scaled_hm_mod[:, :, z]),
                                       vmin=-1, vmax=1)
                    else:
                        subplot.imshow(scaled_hm_mod[:, :, z],
                                       # cmap= plt.get_cmap(self.heatmap_cmap),  alpha=self.alpha,
                                       cmap=plt.get_cmap(self.heatmap_cmap), #alpha=np.absolute(scaled_hm_mod[:, :, z]),
                                       vmin=-1, vmax=1)
                                   # vmin=np.min(self.heatmap_lst[0]),
                                   # vmax=np.max(self.heatmap_lst[0]))
                else:
                    hm = self.heatmap_lst[(i - 1) // columns - 1] # select all mod for that method
                    scaled_hm = hm #normalize_scale(hm, self.outlier_perc) # normalize the full modality
                    scaled_hm_mod = scaled_hm[(i - 1) % columns] # select single modality to subplot
                    if self.hm_on == False and self.mri_on == False:
                        self.mri_on = True
                    if self.mri_on :#or (self.hm_mix_vis and ((i-1)//columns) %2 ==1):
                        subplot.imshow(mri[:, :, z], cmap=plt.get_cmap(self.cmap),
                                       vmin=np.min(mri), vmax=np.max(mri))
                    if self.hm_on:# or self.hm_mix_vis:
                        # three_chnl_hm = np.expand_dims(hm[:, :, z], axis=-1)
                        # viz.visualize_image_attr(three_chnl_hm, mri[:, :, z], method="alpha_scaling")
                        #             hm_mask = np.ma.masked_array(mri*hm > 0, hm)
                        # alpha_mapping = self.map_alpha(scaled_hm, z) # mapping abs(hm) to  alpha value:  0 (transparent) and 1 (opaque)
                        # alpha_mapping = Normalize(vmin = 0, vmax=np.max(np.absolute(self.heatmap_lst[(i-1)//columns-1])))(np.absolute(hm))
                        # print(alpha_mapping.shape, mri[:, :, z].shape) #(128, 128)
                        subplot.imshow(scaled_hm_mod[:, :, z], #cmap= plt.get_cmap(self.heatmap_cmap),  alpha=self.alpha,
                                       cmap=plt.get_cmap(self.heatmap_cmap), alpha=np.absolute(scaled_hm_mod[:,:, z]),
                                       vmin= -1, vmax = 1)
                                       # vmin=np.min(self.heatmap_lst[(i-1)//columns-1]), vmax=np.max(self.heatmap_lst[(i-1)//columns-1]))
                        if len(self.contours) >0:
                            c_cmap = cm.cool
                            c_cmap.set_under('k', alpha=0)
                            for c in self.contours:
                                subplot.imshow(c[:,:, z], cmap=c_cmap, interpolation='none', clim=[0.98, 1])
                    #             plt.imshow(hm_mask[:,:,z], cmap='gray', alpha=0.5)
                hm_title_color = 'white' if self.mri_on else 'black'
                if self.hm_mix_vis:
                    pass
                else:
                    subplot.set_title("{}: {}  {}\n".format((i-1)//4,  self.modality[(i-1)%columns].upper(), method_names[(i-1)//columns] ), y=0.85, color = hm_title_color, fontsize=fontsize)
            splots.append(subplot)
        # self.fig.tight_layout()
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0,
                            hspace=0)

        # remove the x and y ticks
        for i, ax in enumerate(splots):
            ax.set_xticks([])
            ax.set_yticks([])
            # if i %4 == 3+:
            #     ax.get_shared_y_axes().join(*row)
            #     row = []
        # colorbar= 'Red: most important. Blue: least important'

        self.fig.suptitle("{}   {}".format(self.title_prefix, self.bratsid), y = 1 + fontsize*0.0005, fontsize = fontsize)
        if not self.mri_only:
            # colorbar
            norm = mpl.colors.Normalize(vmin=-1, vmax=1)
            # axis_separator = make_axes_locatable(ax)
            # colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
            divider = make_axes_locatable(ax_placeholder)
            cax = divider.new_vertical(size="7%", pad=15, pack_start=True)
            self.fig.add_axes(cax)
            # subplot = self.fig.add_subplot(rows+1, columns,rows+1*columns)
            cbar = self.fig.colorbar(mpl.cm.ScalarMappable( cmap=plt.get_cmap(self.heatmap_cmap) , norm = norm),
                              orientation="horizontal",  ticks=[-1,  0,  1], cax = cax
                              )
                              # cax=colorbar_axis,
            #                   label = "Feature Importance Legend (Red (1): most important. Blue (-1): least important)")
            # set colorbar tick color
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label("Feature Importance Colormap \n\nBlue -1: least important. Red 1: most important.", size=fontsize)
            # set colorbar edgecolor
            cbar.outline.set_edgecolor('white')

    def get_figure(self):
        return self.fig

# save 3D image to video
# kwags_dict = {"mri_lst": input_array,
#               "heatmap_lst": hm_array,
#               "bratsid": brastId,
#               "hm_names": method_list,
#               "figsize": (30, 10),
#               "title_prefix": title_prefix,
#               "heatmap_cmap":'Oranges'
#              }

def generate_mrivideo(video_name, exist_ok = True, subfolder = None, ffmpeg = True, dir = '/local-scratch/authorid/trained_model/BRATS20_HGG/heatmaps/get_hm_fold_1/videos', png_to_video=False,  **kwargs):
    if subfolder:
        folder = Path(dir)/ subfolder  # use bratsid as subfolder name
    else:
        folder = Path(dir)
    img_folder = folder/video_name
    img_folder.mkdir(parents= True, exist_ok=True)
    folder.mkdir(parents= True, exist_ok=True)
    video_file = folder / '{}.mp4'.format(video_name)
    file_exists = os.path.isfile(video_file)
    if not png_to_video:
        viewer = MultipleMRIViewer(**kwargs)
        depth= kwargs["mri_lst"][0].shape[-1]
        print('depth is {}'.format(depth))
        if file_exists:
            if exist_ok:
                print("{} file exists, pass".format(video_file))
                return
            else: # cover prior video file
                os.remove(video_file)

        for z in range(depth):
            img_file = img_folder /"hm{0:03d}.png".format(z)
            img_file_exists = os.path.isfile(img_file)
            if img_file_exists:
                continue
            viewer.plot_slice(z)
            figure = viewer.get_figure()
            figure.savefig(img_file)
            plt.close('all')
    # os.chdir(folder)
    if ffmpeg:
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', img_folder/'hm%03d.png', '-r', '30', '-pix_fmt', 'yuv420p', #'-noautoscale',
            video_file
        ])
    # for file_name in folder.rglob('*.png'):#glob.glob(folder/"*.png"):
    #     os.remove(file_name) # todo keep png images and select for thrumbnail image in the survey

# generate_mrivideo("aaa", **kwags_dict)


def plot_paper_figure_4img(dataid, hm_dir, input_dir, figsize=10):
    # get all dataid in the dir, if wrong prediction, print extra label
    hm_dict, _, _ = get_heatmaps(hm_dir, dataid, hm_as_array=False, return_mri=False, get_pred_label=True)
    input_data = pickle.load(open(os.path.join(input_dir, "{}.pkl".format(dataid)), "rb"))
    image = input_data['image']
    gt = input_data['gt']
    #     hms = [p for p in Path(hm_dir).rglob('*{}*.pkl'.format(dataid))]
    hms = glob.glob(os.path.join(hm_dir, '*{}*.pkl'.format(dataid)))  # only read csv in current dir, not subdir
    hms = [Path(p) for p in hms]
    if len(hms) == 0:
        return
    n_row = 1 + len(hms)
    n_col = len(modality_dict)
    # show input image in the first row, and each xai in rest rows
    fig = plt.figure(figsize=(figsize, int(figsize / n_col * (n_row + 0.4))))
    splots = []
    #     print(hms)
    for i in range(1, n_col * n_row + 1):

        subplot = fig.add_subplot(n_row, n_col, i)
        if i <= n_col:
            mod_name = list(modality_dict.keys())[i - 1]
            #             print(mod_name, modality_dict[mod_name])
            subplot.imshow(image[modality_dict[mod_name]], cmap='gray')

            subplot.set_title("{}".format(mod_name))
        else:
            row = (i - 1) // 4 - 1
            col = (i - 1) % 4
            title = hms[row].name
            hm = pickle.load(open(hms[row], "rb"))
            #             print(type(hm) =='numpy.ndarray')
            if not isinstance(hm, (np.ndarray)):
                print(type(hm))
                hm = hm.detach().cpu().numpy()
            hm = postprocess_heatmaps(hm, img_size=(256, 256), rotate_axis=False)
            mod_name = list(modality_dict.keys())[(i - 1) % n_col]
            #             print(row, col,mod_name, hm.shape)
            subplot.imshow(hm[modality_dict[mod_name]], vmin=-1, vmax=1, cmap='bwr')
            subplot.set_title("{}".format(hms[row].name))
        splots.append(subplot)
    for i, ax in enumerate(splots):
        ax.set_xticks([])
        ax.set_yticks([])

def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def normalize_scale(attr: ndarray, outlier_perc = 1):
    # https://github.com/pytorch/captum/blob/7d21f58371cae981a1707625f85246ad559cea9b/captum/attr/_utils/visualization.py#L41
    scale_factor = _cumulative_sum_threshold(np.abs(attr), 100 - outlier_perc)
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)

def get_data_ids(save_dir):
    '''
    Obsolete method. Some folder don't save input in "input" folder
    Get data ids, and saved csv file

    :param save_dir:
    :return:
    '''
    save_dir = Path(save_dir)
    # read csv
    csv_filename = save_dir / 'record_{}.csv'.format(str(save_dir.name))
    df = pd.read_csv(csv_filename)
    # read data ids
    input_dir = save_dir /'input'
    data_ids = [k.name.strip('.pkl') for k in Path(input_dir).rglob('*.pkl')]

    return data_ids, df

def sanity_check_saved_heatmaps(save_dir, method_list):
    "Ensure each datapoint has heatmaps in the method_list"
    data_ids, df = get_data_ids(save_dir)
    method_list = set(method_list)
    missed_hm = False
    for d in data_ids:
        _, m_lst, _, _ = get_heatmaps(save_dir, d)
        if set(method_list).issubset(set(m_lst)): # the tobe evaluated method_list is a subset of m_lst
            pass
        else:
            logging.info("Data {} missing hm: {}".format(\
                d, set(method_list)-set(m_lst)))
            missed_hm = True
    if missed_hm:
        return False
    return True

def save_dir_sanity_check(save_dir, method_list = None, valid_data_loader = None):
    # check_heatmap_exist
    if method_list:
        result = sanity_check_saved_heatmaps(save_dir, method_list)
        if result:
            logging.info("Pass: Saved heatmaps have all required XAI methods.")
        else:
            logging.info("Fail: Saved heatmaps missing XAI methods.")
    # check_dataloader_complete(self, save_dir):
    data_ids, df = get_data_ids(save_dir)
    if valid_data_loader:
        if len(valid_data_loader)  == len(data_ids):
            logging.info("Pass: Validation Data is complete!")
        else:
            logging.info("Fail: input folder data number: {}. Val data loader: {}. Missing data!".format(len(data_ids), len(valid_data_loader)))
    # check if val data accuracy metric exists
    acc = df[['Data_ID', 'Predicted_Correct']]
    acc = acc.drop_duplicates()
    if set(acc['Data_ID']) == set(data_ids):
        logging.info("Pass: Saved CSV contains all data points and have unique prediction results")
        accuracy = acc['Predicted_Correct'].value_counts()
        val_acc = accuracy[1] / len(acc)
        return val_acc
    else:
        logging.info("Fail! {} in accuracy results. Totol validation data {}".format(len(acc), len(data_ids)))
    return


def get_heatmaps(save_dir, data_or_method_name, by_data= True, hm_as_array= True, return_mri = True, get_pred_label = True):
    '''
    Main function to read saved results and get heatmaps. Default to load heatmap of predicted class.
    :param by_data: default, load all methods for a datapoint; False: load by xai method
    :param hm_as_array: if True, return hms as [method, C, H, W, D], else, return a dict {method: hm  [C, H, W, D]}
    :return:
    '''
    save_dir = Path(save_dir)
    hm_dir = save_dir / 'heatmap'
    input_dir = save_dir / 'input'
    if by_data:
        # get the heatmap as [method, C, H, W, D] for a data point, and return the method list and array
        data_id = data_or_method_name
    else:
        method_name = data_or_method_name
        # data_ids = [k.name.strip('.pkl') for k in Path(input_dir).rglob('*.pkl')]

    csv_filename = save_dir / 'record_{}.csv'.format(str(save_dir.name))
    # print(csv_filename)
    try:
        record = pd.read_csv(csv_filename)
        if by_data:
            data_record = record[record['Data_ID']== data_id]
        else:
            data_record = record[record['XAI_Method'] == method_name]
    except:
        print(csv_filename, 'failed to open!')
        data_record = None

    if hm_as_array:
        hm_array = []
        name_list = []
    else:
        hm_dict = dict()

    for k in Path(hm_dir).rglob('*{}*.pkl'.format(data_or_method_name)): # get all files contains method or data_id
        fn_segments_list = k.name.split('.')[0].split('-')
        data_id = fn_segments_list[0]
        method_name = fn_segments_list[1]
        # to prevent method name duplication, eg: Gradient appears in 4 XAI method names
        if (by_data and data_id == data_or_method_name) or ((not by_data) and method_name == data_or_method_name):
            if len(fn_segments_list) == 3: # in case prediction is wrong, and has postfix of prediction / gt hms
                if get_pred_label: # only get hm for the predicted class, will ignore the True label with T
                    label = 'T'
                else:
                    label = 'P'
                if fn_segments_list[2][0] == label:
                    continue
            logging.debug('Heatmap loaded: {} in {}'.format(k.name, k))
            hm = pickle.load(open(k, "rb"))
            if by_data:
                name = method_name
            else:
                name = data_id
            # name = lambda by_data: method_name if by_data else data_id
            if hm_as_array:
                hm_array.append(hm)
                name_list.append(name)
            else:
                hm_dict[name] = hm
            # print(hm.shape, method_name)
    if not by_data:
        if hm_as_array:
            return hm_array, name_list, data_record
        else:
            return hm_dict, data_record
    input_array = None
    if return_mri:
        input_array = pickle.load(open(input_dir / '{}.pkl'.format(data_id), "rb"))
        # if input_array.is_cuda:
        input_array = input_array.cpu().detach().numpy()
        # else:
        #     input_array = input_array.cpu().numpy()
    if hm_as_array:
        hm_array = np.stack(hm_array)
        return hm_array, name_list, input_array, data_record
    else:
        return hm_dict, input_array, data_record


def postprocess_heatmaps(hm_array, img_size = (240, 240, 155), num_modality = 4, no_neg = False, rotate_axis = True):
    '''
    Heatmap post-processing pipeline:
    Upsample to original size of the input
    Normalize (modality-wide) to -1, 1
    (get rid of -1 for LIME)
    Gaussian smooth
    :param hm_array: [C, H, W, D] of numpy array
    :param img_size: [H, W, D] = (240, 240, 155)
    :return: numpy array
    '''
    # Upsample to original size of the input
    # print(hm_array.shape)
    if hm_array.ndim == len(img_size): # For GradCAM input (2,2,2), copy the channel to get 4 mods
        hm_cp = [hm_array] * num_modality
        hm_array = np.stack(hm_cp)
    # print(hm_array.shape)
    hm_tensor = torch.from_numpy(np.expand_dims(hm_array, axis=0))
    # print(hm_tensor.shape)
    # print('in postprocess', hm_array.shape[1:] , img_size, np.array_equal(np.array([int(i) for i in hm_array.shape[1:]]) , np.array(img_size)))
    if np.array_equal(np.array([int(i) for i in hm_array.shape[1:]]) , np.array(img_size)):
        upsampled_hm_tensor = hm_tensor
    else:
        print("postprocess_heatmaps: upsample needed")
        upsampled_hm_tensor = LayerAttribution.interpolate(layer_attribution=hm_tensor,
                                                           interpolate_dims=img_size)  # layer_attribution (torch.Tensor) [batch, C, H, W, D]
        # print(upsampled_hm.shape)

    upsampled_hm = upsampled_hm_tensor.numpy()
    upsampled_hm = np.squeeze(upsampled_hm)

    # Gaussian smooth
    hm = []
    for mod in upsampled_hm:
        if rotate_axis:
            mod = np.rot90(mod, k = 3, axes = (0,1))
        if len(img_size) == 3:
            hm.append(ndimage.gaussian_filter(mod, sigma= [1, 1, 0]))
        elif len(img_size) ==2:
            hm.append(ndimage.gaussian_filter(mod, sigma= [3, 3]))
    hm = np.stack(hm)
    # Normalize (modality-wide) to -1, 1
    hm = normalize_scale(hm)
    # print(hm.shape)
    if no_neg:
        hm[hm<=0] = 0
    return hm






def load_mri(path, bratsID, get_seg = True, rotate_axis = True):
    '''
    load mri and seg
    :param path:
    :param bratsID:
    :return:
    '''
    mri_lst = []
    bg_mask = []
    for m in modality:
        mri_path = os.path.join(path, bratsID, bratsID + '_{}.nii.gz'.format(m.lower()))
        mri = nibabel.load(mri_path).get_fdata() # [H,W,D]
        if rotate_axis:
            mri = np.rot90(mri, k=3, axes=(0, 1))
        mri_lst.append(mri)
    if not get_seg:
        return mri_lst
    seg = read_seg(path, bratsID)
    if rotate_axis:
        seg = np.rot90(seg, k=3, axes=(0, 1))
    return mri_lst, seg

def get_bgmask(mri_lst):
    bg_masks = []
    bg_mask_max_brain_slice = []
    skull_masks = []
    edge = 20
    skull_thick = 8
    selem = np.ones((edge,edge))
    selem_skull = np.ones((skull_thick, skull_thick))
    for m in mri_lst:
        bg_slices = []
        skull_slices = []
        bg = np.where(m == 0, 0, 1) # mask = 0 is background
        # mask on axial slice (axis = 2)
        for i in range(bg.shape[2]):
            brain_mask = convex_hull_image(bg[:,:,i]) # ndarray
            mask = binary_dilation(brain_mask, selem = selem)
            bg_slices.append(mask)

            skull_part = binary_dilation(brain_mask, selem = selem_skull)
            skull = np.where( (skull_part==1) & (brain_mask ==0), 1, 0 )
            skull_slices.append(skull)
        # convert slices into 3d masks
        masks = np.dstack(bg_slices)
        bg_masks.append(masks)
        n_slice = np.argmax(np.sum(masks, axis=(0,1)))   # get the axial with the least bg portion
        max_brain_slice = masks[:, :, n_slice]
        bg_mask_max_brain_slice.append(max_brain_slice)
        skull_mask = np.dstack(skull_slices)
        skull_masks.append(skull_mask)
    return np.stack(bg_masks), np.stack(skull_masks), np.stack(bg_mask_max_brain_slice)

def get_bgmask_from_id(path, bratsID):
    '''
    Obsolete function
    '''
    mri_lst = load_mri(path, bratsID, get_seg = False)
    bg_masks, skull_masks = get_bgmask(mri_lst)
    return bg_masks, skull_masks

def read_bgmask_from_id(bgmask_path, bratsID, slice_wise):
    '''
    :param path: a path, with bratsID as folder name, and each pkl name with bratsID_bgmask2d/bgmask4d/skull4d
    :param bratsID:
    :param slice_wise:
    :return:
    '''
    skull4d_file = os.path.join(bgmask_path, bratsID, '{}_skullMask.pkl'.format(bratsID))
    skull4d = pickle.load(open(skull4d_file, "rb"))
    if slice_wise:
        bg_mask4d_file = os.path.join(bgmask_path, bratsID, '{}_bgMask.pkl'.format(bratsID))
        bg_mask4d = pickle.load(open(bg_mask4d_file, "rb"))
        return bg_mask4d, skull4d
    bg_mask2d_file = os.path.join(bgmask_path, bratsID, '{}_maxBrainSlice.pkl'.format(bratsID))
    bg_mask2d = pickle.load(open(bg_mask2d_file, "rb"))
    return bg_mask2d, skull4d



## get the axial with the least bg portion
# n_slice = np.argmax(np.sum(bg_masks[i], axis=(0,1)))
def save_bgmask(bg_masks, skull_masks, bg_mask_max_brain_slice, bgmask_path, bratsID):
    dir = os.path.join(bgmask_path, bratsID)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # save np array as pickles
    pickle.dump(bg_masks, open(os.path.join(bgmask_path, bratsID, '{}_bgMask.pkl'.format(bratsID)), 'wb'))
    pickle.dump(skull_masks, open(os.path.join(bgmask_path, bratsID, '{}_skullMask.pkl'.format(bratsID)), 'wb'))
    pickle.dump(bg_mask_max_brain_slice, open(os.path.join(bgmask_path, bratsID, '{}_maxBrainSlice.pkl'.format(bratsID)), 'wb'))


def get_and_save_bgmask(mri_path, bgmask_path):
    '''
    Generate and save background masks given brats data path
    :param mri_path:
    :param bgmask_path:
    :return:
    '''
    bratsIDs = [f for f in os.listdir(mri_path) if os.path.isdir(os.path.join(mri_path, f))]
    for bratsID in bratsIDs:
        mri_lst = load_mri(mri_path, bratsID, get_seg=False)
        bg_masks, skull_masks, bg_mask_max_brain_slice = get_bgmask(mri_lst)
        save_bgmask(bg_masks, skull_masks, bg_mask_max_brain_slice, bgmask_path, bratsID)


def load_hm(path, bratsID, non_neg=False):
    '''
    obsolete, split it to load_hm and load_mri, since hm and mri may be in different folder
    Given a path of bratsID, load it with 4 MR modalities, and 4 heatmaps for each modality.
    Used for Vis to overlay heatmap with MRI
    Load MRI and its saved heatmaps
    non_neg: get rid of negative value of heatmap
    '''

    hm_lst = []
    hm_smooth_lst = []
    for m in modality:
        hm_path = os.path.join(path, bratsID, '{}_heatmap.nii'.format(m.lower()))
        hm = nibabel.load(hm_path).get_fdata()
        hm = hm[::-1, ::-1, :]
        if non_neg:
            hm[hm < 0] = 0
        hm_smooth = ndimage.gaussian_filter(hm, sigma=2)

        hm_lst.append(hm)
        hm_smooth_lst.append(hm_smooth)
    # seg_path = os.path.join(path, bratsID, bratsID + '_seg.nii.gz')
    # seg = nibabel.load(seg_path).get_fdata()
    return hm_lst, hm_smooth_lst


def load_data_vis(path, bratsID, non_neg=False):
    '''
    obsolete, split it to load_hm and load_mri, since hm and mri may be in different folder
    Given a path of bratsID, load it with 4 MR modalities, and 4 heatmaps for each modality.
    Used for Vis to overlay heatmap with MRI
    Load MRI and its saved heatmaps
    non_neg: get rid of negative value of heatmap
    '''

    mri_lst = []
    hm_lst = []
    hm_smooth_lst = []
    for m in modality:
        hm_path = os.path.join(path, bratsID, '{}_heatmap.nii'.format(m.lower()))
        hm = nibabel.load(hm_path).get_fdata()
        mri_path = os.path.join(path, bratsID, bratsID + '_{}.nii.gz'.format(m.lower()))
        mri = nibabel.load(mri_path).get_fdata()

        hm = hm[::-1, ::-1, :]
        if non_neg:
            hm[hm < 0] = 0
        hm_smooth = ndimage.gaussian_filter(hm, sigma=2)

        hm_lst.append(hm)
        mri_lst.append(mri)
        hm_smooth_lst.append(hm_smooth)
    # seg_path = os.path.join(path, bratsID, bratsID + '_seg.nii.gz')
    # seg = nibabel.load(seg_path).get_fdata()
    seg = read_seg(path, bratsID)
    return mri_lst, hm_lst, hm_smooth_lst, seg

def read_seg(path, bratsID):
    seg_path = os.path.join(path, bratsID, bratsID + '_seg.nii.gz')
    seg = nibabel.load(seg_path).get_fdata()
    return seg




# post processing

# utility func.
def scale01(hm):
    return (hm - hm.min()) / (hm.max() - hm.min())


def scale01_hms(hms):
    '''
    Scale the heatmaps in 4 modality to 0-1.
    Scale base on single modality and scale 4 modalties all to [0,1] is not correct, since the model may pay different attention to different modalities.
    Thus use scale 4 mod together. This may be sensitive to extereme value, but extreme values are meaningful
    https://developers.google.com/machine-learning/data-prep/transform/normalization
    '''
    mn = min([hm.min() for hm in hms])
    mx = max([hm.max() for hm in hms])
    scaled_hms = []
    for hm in hms:
        scl = (hm - mn) / (mx - mn)
        scaled_hms.append(scl)
    return np.array(scaled_hms)


def scale_hms(hms):
    '''
    when hm contains negative value, scale to -1, 1, whichever side (pos/neg) is the largest range set to abs(1)
    '''

    neg_value = np.array([(hm < 0) for hm in hms]).sum()
    if neg_value == 0:
        logging.debug("All positive value, scale to [0,1]")
        # if no neg value
        return scale01_hms(hms)
    else:
        logging.debug("Has negative value, scale to [-1,1]")
        scaled_hms = []
        mn = min([np.abs(hm.min()) for hm in hms])
        mx = max([hm.max() for hm in hms])
        scale_val = max(mn, mx)
        for hm in hms:
            hm_scale = np.copy(hm)
            hm_scale /= scale_val
            scaled_hms.append(hm_scale)
    return np.array(scaled_hms)


def vis_scale_dstb(hm_before, hm_after, log=False):
    '''Visualize the distribution shift of hm before/after the scaling'''

    bins = np.linspace(min(hm_before.min(), hm_after.min()), max(hm_before.max(), hm_after.max()), 50)
    plt.hist([hm_before, hm_after], bins, label=['Before', 'After'])
    plt.legend()
    if log:
        plt.yscale('log', nonposy='clip')
    plt.show()
    print('min & max \nBefore: {}, {} \nAfter: {}, {}'.format(hm_before.min(), hm_before.max(), hm_after.min(),
                                                              hm_after.max()))


def heatmap_to_binary(hms, criteria, cutoff, ABS=False, smooth=False):
    '''
    -1. get rid of negative value (if any),
    1. normalize to 0,1 if all pos, or [-1,1] if contains neg,  smooth if needed
    2. according to criteria, select the top value and set to 1, set the rest to 0
    criteria:
    - threshold, set value >= cutoff to 1
    - quantile, set value >= quantile cutoff to 1
    Input:
    hms: hms of 4 mods in a list
    ABS: whether to keep the absolute value when doing cutoff
    smooth: gassian smooth of the heatmaps
    return a binary heatmap
    '''
    #     print(hm.min(), hm.max())
    #     if (hm < 0).sum() >0 :
    #         print('Setting negative value to 0')
    #         hm[hm < 0] = 0
    #     hm = scale01(hm)
    #     print(hm.min(), hm.max())

    # post-processing, scale, smooth
    if hms.max() > 1 or hms.min() < -1:
        logging.info("normalize heatmap")
        hms = normalize_scale(hms)
    # scaled_hms = scale_hms(hms)  # (4, 240, 240, 155)
    if ABS:
        hms = np.absolute(hms)
    bn_hms = []
    if smooth:
        hms = ndimage.gaussian_filter(hms, sigma=1)  # using 1d conv filter ??

    if criteria == "t":  # threshold
        print('Using {} of cutoff value {}'.format(criteria, cutoff))
        for hm in hms:
            hm[hm >= cutoff] = 1
            hm[hm < cutoff] = 0
            bn_hms.append(hm)
    elif criteria == 'q':  # quantile
        # compute quantile based on non zero value
        hms = np.array(hms)
        quantile = np.quantile(hms[np.nonzero(hms)], cutoff)
        assert quantile != hms.min(), print('Error! quantile == hm.min()')
        print('Using {} of cutoff value {}'.format(criteria, quantile))
        for hm in hms:
            hm[hm >= quantile] = 1
            hm[hm < quantile] = 0
            bn_hms.append(hm)
    elif criteria == 'all_positive':
        hms[hms >= 0.0] = 1
        hms[hms < 0.0] = 0
        bn_hms = hms
    else:
        print('No criteria specified!')
    #     assert hm.min()==0 and hm.max()==1, 'Error! hm.min()!=0 or hm.max()!=1'
    #     assert (np.unique(hm) == np.array([0., 1.])).all(), 'Error! Unique value is not 0 and 1'
    return np.array(bn_hms)


# def generate_binary_heatmap_across_modality(hms, criteria="q", cutoff = 0.5, smooth= False):
#     '''
#     No longer used since heatmap_to_binary deals with all mods
#     '''
#     binary_hms = list()
#     for hm in hms:
#         binary_hm = heatmap_to_binary(hm, criteria=criteria, cutoff = cutoff, smooth= smooth)
#         binary_hms.append(binary_hm)
#     return binary_hms


def IoU(seg, bn_hms):
    '''
    Compute IoU of hm of 4 modalities with segmentation map of different tumor subpart.
    Input: bn_hms: binary heatmaps of 4 modalties
    Seg array([0., 1., 2., 4.])
    0 - bg
    1 - necrotic and non-enhancing tumor core, NCR/NET
    2 - peritumoral edema, ED
    4 - GD-enhancing tumor, ET
    1 & 4 - TC (Tumor core)
    1 & 4 & 2 - WT (Whole tumor)
    Return: iou_score dict, each value include list of iou for each modality
    '''
    seg = np.stack([seg]*len(bn_hms)) # from (240, 240, 155) to (4, 240, 240, 155)
    iou_score = {'wt': list(), 'tc': list(), 'et': list(), 'ed': list(), 'ncr': list()}
    wt = (np.logical_or(np.logical_or(seg == 1, seg == 4), seg == 2)).astype(int).reshape(-1)
    tc = (np.logical_or(seg == 1, seg == 4)).astype(int).reshape(-1)
    et = (seg == 4).astype(int).reshape(-1)
    ed = (seg == 2).astype(int).reshape(-1)
    ncr = (seg == 1).astype(int).reshape(-1)
    hms = bn_hms.astype(int).reshape(-1)
    iou_score['wt'].append([jaccard_score(wt, hms)]*len(bn_hms))
    iou_score['tc'].append([jaccard_score(tc, hms)]*len(bn_hms))
    iou_score['et'].append([jaccard_score(et, hms)]*len(bn_hms))
    iou_score['ed'].append([jaccard_score(ed, hms)]*len(bn_hms))
    iou_score['ncr'].append([jaccard_score(ncr, hms)]*len(bn_hms)) # extend IoU score to a list of length |modality|, to accomdate old code
    print('IoU', iou_score['wt'])
    return iou_score

def IoU_wt(seg, bn_hms):
    '''
    Since IoU function contains bugs when calling jaccard_score, manually calculate IoU score.
    Compute IoU of hm of 4 modalities with segmentation map of different tumor subpart.
    Input: bn_hms: binary heatmaps of 4 modalties
    Seg array([0., 1., 2., 4.])
    0 - bg
    1 - necrotic and non-enhancing tumor core, NCR/NET
    2 - peritumoral edema, ED
    4 - GD-enhancing tumor, ET
    1 & 4 - TC (Tumor core)
    1 & 4 & 2 - WT (Whole tumor)
    Return: iou_score dict, each value include list of iou for each modality
    '''
    seg = np.stack([seg]*len(bn_hms)) # from (240, 240, 155) to (4, 240, 240, 155)
    iou_score = {'wt': list()}
    wt = (seg>0).astype(int).reshape(-1)
    hms = bn_hms.astype(int).reshape(-1)
    intersection = np.logical_and(hms == 1, wt == 1).sum()
    union = np.logical_or(hms == 1, wt == 1).sum()
    iou = intersection/union
    print(iou, intersection, union)
    iou_score['wt'].append([iou]*len(bn_hms)) # extend IoU score to a list of length |modality|, to accomdate old code
    print('IoU', iou_score['wt'])
    assert iou >=0.0
    assert iou <= 1.0
    return iou_score

def tumor_portion(seg, hms, ABS=False):
    '''
    Compute how much portion the heatmap is paying attention to the tumor region.
    Tumor region has two definitions:
    1 & 4 - TC (Tumor core)
    1 & 4 & 2 - WT (Whole tumor)
    Input:
    heatmap: may be binary hm, or with soft probabilities (assume raw not normalized)
    ABS: convert hms to absolute value and then calculate
    :return:
    tumor_region_precetage={'tc_portion': [list of portion for 4 mods], 'wt_portion': [list of portion for 4 mods]}
    '''
    wt = (np.logical_or(np.logical_or(seg == 1, seg == 4), seg == 2)).astype(int)
    tc = (np.logical_or(seg == 1, seg == 4)).astype(int)
    tumor_region_precetage = {'wt_portion': list(), 'tc_portion': list()}
    # print(seg.shape, hms.shape, 'shape') # (240, 240, 155) (4, 240, 240, 155) shape
    if ABS:
        hms = np.absolute(hms)
    if np.unique(hms).shape[0] == 2:
        if (np.unique(hms) == np.array([0., 1.])).all():  # binary hm
            for hm in hms:
                hm = hm.astype(int)
                wt_portion = np.logical_and(wt == 1, hm == 1).sum() / hm.sum()
                tc_portion = np.logical_and(tc == 1, hm == 1).sum() / hm.sum()
                tumor_region_precetage['wt_portion'].append(wt_portion)
                tumor_region_precetage['tc_portion'].append(tc_portion)
    else:
        # print('Raw heatmap min-max range:\n{}, {}'.format(np.array(hms).min(), np.array(hms).max()))
        if hms.max() > 1 or hms.min() < -1:
            logging.info("normalize heatmap")
            hms = normalize_scale(hms)
            print('Feature portion: Scaled [-1,1] heatmap min-max range:\n{}, {}'.format(scl_hms.min(), scl_hms.max()))
        wt_portion = hms[np.stack([wt]*len(hms)) == 1 ].sum() / hms.sum()
        tc_portion = hms[np.stack([tc]*len(hms)) == 1 ].sum() / hms.sum()
        tumor_region_precetage['wt_portion'].append([wt_portion]*len(hms))
        tumor_region_precetage['tc_portion'].append([tc_portion]*len(hms))
        print('wt_portion', wt_portion)
    return tumor_region_precetage


def compare_with_gt(seg, hms):
    '''
    pipeline of comparing heatmap with gt seg map:
    1. post-processing hm:
        resize to the same size
        normalize
    2. use normalized hm to compute tumor_portion
    3. generate binary hm, compute IoU
    :param seg: np array
    :param hm: direct output size from model, size of input [C, H, W, D]
    :return: gt_result: 2D array: [tumor_port + IoU, 4 mods]
    '''
    if seg.shape != hms.shape[1:]:
        # ori_size = (240, 240, 155) # TODO check input size after LoadNifti, and visualize to see if match with seg and mri
    # resize needed when compare gt seg, and for vis
        logging.info("Resize hm from {} to {}".format(hm.shape[1:], seg.shape))
        hms = Resize(spatial_size=[hms.shape[0]]+seg.shape)(hms)
    # hm_nm = scale_hms(hm_resized)
    if hms.max()>1 or hms.min()<-1:
        logging.info("normalize heatmap")
        hms = normalize_scale(hms)


    # tumor portion
    tumor_region_precetage = tumor_portion(seg, hms= hms, ABS=False)
    # IoU
    # bn_hms = heatmap_to_binary(hm_nm, 't', 1e-5, True, False) # TODO: hyperparameter search for optional threshold for each method
    # bn_hms = heatmap_to_binary(hms, 'all_positive', 0.0) #using it would be smaller than using upper 50% quantile, as there are more noisy heatmap signals in the denominator
    bn_hms = heatmap_to_binary(hms, 'q', 0.5)
    # bn_hms = heatmap_to_binary(hms, 't', 0.1)
    iou_score = IoU_wt(seg, bn_hms)
    gt_compare_col = list(tumor_region_precetage.keys()) + list(iou_score.keys())
    gt_result = list(tumor_region_precetage.values()) + list(iou_score.values())

    return gt_result, gt_compare_col


### heatmap similarity calculation
def corr_hog(hm1, hm2):
    '''
    rank correlation between histogram of gradients
    Input: hm of HWD
    For 3D heatmaps, compute HOG on each 2D slice along the axial view.
    '''
    assert hm1.shape == hm2.shape
    assert len(hm1.shape)==3
    corr = list()
    for d in range(hm1.shape[2]):
        hog1 = hog(hm1[:,:,d], pixels_per_cell=(16, 16))
        hog2 = hog(hm2[:,:,d], pixels_per_cell=(16, 16))
        rank_corr_hog, _ = spr(hog1, hog2)
        corr.append(rank_corr_hog)
    corr = np.nanmean(np.array(corr))
    return corr



def compare_heatmap_similarity(hm1, hm2):
    '''

    :param hm1: heatmap of a MRI image, consist of 1 mod only, HWD
    :param hm2:
    :return: similarity metrics of :
    -
    '''
    ss = ssim(hm1.flatten(), hm2.flatten(), gaussian_weights=True,
              multichannel=False)  # compare single hm of HWD. need a for loop for 4 modalities
    sp, _ = spr(hm1.flatten(), hm2.flatten())
    hog = corr_hog(hm1, hm2)
    # mi = normalized_mutual_info_score(hm1.flatten(), hm2.flatten())
    # mi = sitk.MattesMutualInformationImageToImageMetric(hm1, hm2)
    mi = None # TODO
    sim_dict = {'ssmi': ss, 'spearmanr': sp, 'mi': mi, 'hog': hog}
    sim_array = [ss, sp, hog]#, mi]
    sim_methods = ['ssmi','spearmanr',  'hog']#,'mi']
    return sim_array, sim_methods

def compare_heatmap_similarity_all_mod(hms1, hms2):
    '''

    :param hms1:
    :param hms2:
    :return: a 2D array with col = sim_dict, and row = channel  [channel, sim_method_val]
    '''
    result = []
    sim_methods = None
    for channel in range(hms1.shape[0]):
        sim_array, sim_methods = compare_heatmap_similarity(hms1[channel], hms2[channel])
        result.append(sim_array)
        # for sim_method in sim_dict: #sim_array, col_name
        #     result[sim_method+'_'+channel] =sim_dict[sim_method]
    return result, sim_methods

# def compare_heatmap_similarity_dataset():
#     '''
#     Given a dataloader, cal compare_heatmap_similarity for each data and compute sim for each images
#     :return:
#     '''
#     return

# bratsID_lst = [\
#                'BraTS19_TCIA03_474_1',
#                'BraTS19_TCIA01_412_1'
# ]
# idx = 0
# mri_lst, hm_lst, hm_smooth_lst, seg = load_data(path, bratsID_lst[idx], False)
# binary_hms = heatmap_to_binary(hm_lst, 't', 0.1, True, False) #'q', 0.98,

#####################
#     visualize 3D image as gif
#####################

def image3d_to_gif(save_dir=None, mris = None, heatmaps = None, seg = None, mi_gt = None, keep_png = False, modalities = modality, figwidth = 30, fontsize = 40, show_img = False, cmap = 'autumn'):
    ''' backbone code of convert data to image to gif
    :param save_dir: the save_dir with subdir name (may contain exepriemnt label) as filename of gif.
    :param mri: shape of [modality, H, W, D]
    :param heatmap:  shape of [modality, H, W, D]
    :param seg:  shape of [modality, H, W, D] or [H, W, D]
    :return:
    '''
    if heatmaps is not None:
        assert len(heatmaps.shape) == 4
        nb_modality = heatmaps.shape[0]
        nb_slice = heatmaps.shape[-1]
        if torch.is_tensor(heatmaps):
            heatmaps = heatmaps.cpu().detach().numpy()
    if mris is not None:
        assert len(mris.shape) == 4
        print(mris.shape, 'mri shape')
        if torch.is_tensor(mris):
            mris = mris.cpu().detach().numpy()
        nb_modality = mris.shape[0]
        nb_slice = mris.shape[-1]
    if seg is not None:
        assert len(seg.shape) == 3 or len(seg.shape) == 4
        if heatmaps is not None:
            assert seg.shape == heatmaps.shape or seg.shape == heatmaps.shape[1:]
        elif mris is not None:
            assert seg.shape == mris.shape or seg.shape == mris.shape[1:]
        contour = segmentation.find_boundaries(seg>0)
    if (heatmaps is not None) and (mris is not None):
        assert heatmaps.shape == mris.shape, 'heatmap and mri does not have the same shape'
    for slc in range(nb_slice):
        rows = 1
        columns = nb_modality
        fig = plt.figure(figsize=(figwidth, int(figwidth/columns * (rows))))
    #     fig.suptitle("{}".format(d_id), fontsize=14)

        ax_placeholder = fig.add_subplot(rows + 1, 1, rows + 1)
        ax_placeholder.axis("off")
        splots = [ax_placeholder]
        c_cmap = cm.winter #cm.gnuplot
        c_cmap.set_under('k', alpha=0)
        b_cmap = cm.binary
        b_cmap.set_under('k', alpha=0)
        for i in range(1, columns * rows + 1):
            subplot = fig.add_subplot(rows, columns, i)
            if mris is not None:
                mri = mris[(i - 1) % columns]
                # print('mri range', np.min(mri), np.max(mri), i)
    #             print('mri shape', mri.shape, contour.shape, post_hm.shape)
                subplot.imshow(mri[:, :, slc], cmap=plt.get_cmap('gray'),
                               vmin=np.min(mri), vmax=np.max(mri))
            if mi_gt is not None:
                subplot.set_title("{}: {:.2f}".format(modalities[(i-1)%columns].upper(), mi_gt[modalities[(i-1)%columns]]), y=0.9, color = 'white', fontsize=fontsize)
            else:
                subplot.set_title("{}".format(modalities[(i-1)%columns].upper()), y=0.9, color = 'white', fontsize=fontsize)
            if heatmaps is not None:
                subplot.imshow(heatmaps[(i - 1)% columns ][:, :, slc],
                   # cmap= plt.get_cmap(self.heatmap_cmap),  alpha=self.alpha,
                   cmap=plt.get_cmap(cmap), alpha= heatmaps[(i - 1)% columns ][:, :, slc],
                   vmin=0, vmax=1)
            if seg is not None:
                if mi_gt is not None:
                    subplot.imshow(contour[:,:, slc], cmap=c_cmap, alpha = mi_gt[modalities[(i-1)%columns]], interpolation='none', clim=[0.98, 1])
                else:
                    subplot.imshow(contour[:,:, slc], cmap=c_cmap, alpha = 1, interpolation='none', clim=[0.98, 1])
                splots.append(subplot)
        plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0,
                        hspace=0)

        # remove the x and y ticks
        for i, ax in enumerate(splots):
            ax.set_xticks([])
            ax.set_yticks([])
        if save_dir is not None:
            img_save_dir = Path(save_dir)/'img'
            img_save_dir.mkdir(parents = True, exist_ok = True)
            fig.savefig(os.path.join(img_save_dir, '{:03d}.png'.format(slc)), bbox_inches = 'tight')
        if not show_img:
            plt.clf()

    if save_dir is not None:
        save_img_to_gif(img_save_dir, keep_png= keep_png)
        print('img_save_dir', img_save_dir)
        return img_save_dir
    return


def save_img_to_gif(img_save_dir, keep_png = False):
    # given png save dir, convert them to gif
    gif_dir = Path(img_save_dir).parent
    images = []
    for file_name in sorted(os.listdir(img_save_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(img_save_dir, file_name)
#             print(file_path)
            images.append(imageio.imread(file_path))
    imageio.mimsave(os.path.join(gif_dir, '{}.gif'.format(gif_dir.name)), images)
    if not keep_png:
        for file_name in img_save_dir.rglob('*.png'):#glob.glob(folder/"*.png"):
            os.remove(file_name)


def get_gt(fold, gt_csv_path='/local-scratch/authorid/BRATS_IDH/log/zeroLesion_gt_shapley_multiple_run', modalities = modality):
    # read modality MI gt
    shapley_csv = os.path.join(gt_csv_path, 'multirun_gt_shapley_fold_{}.csv'.format(fold))
    if not os.path.isfile(shapley_csv):
        shapley_csv = Path(gt_csv_path) / 'shapley' / 'fold_{}_modality_shapley.csv'.format(fold)
    # print("shapley_csv",    shapley_csv)
    df = pd.read_csv(shapley_csv)
    df = df.iloc[0]#.to_dict('list')
    # print(df)
    gt_shapley = [df[m] for m in modalities]
    print(gt_shapley)
    # normalize the gt_shapley value
    sh_min = min(gt_shapley)
    for m in modalities:
        df[m] = (df[m] - min(gt_shapley) ) / (max(gt_shapley) - min(gt_shapley))
    return df[modalities]

def vis_save_heatmap(heatmap_dir, seg_dir, save_dir, d_id, fold, xai, label = None, keep_png = True, figwidth = 30, show_img = False, gt = True, show_seg = True, cmap = 'autumn'):
    '''
    Pipeline function to visualize and save heatmaps, given xai method, fold, and data id.
    Read files directly from saved folders.
    :param heatmap_dir:
    :param seg_dir:
    :param save_dir:
    :param d_id:
    :param fold:
    :param xai:
    :param figwidth:
    :return:
    '''
    if gt is not None:
        gt = get_gt(fold = fold)
        print(gt)
    # get MRI and seg
    mri_lst, seg = load_mri(seg_dir, d_id, get_seg=True)
    print(seg.shape)
    for mri in mri_lst:
        print(mri.min(), mri.max())
    mris = np.stack(mri_lst)
    print('mris min max', mris.min(), mris.max())
    # load hm
    if heatmap_dir is not None:
        hm_dir = os.path.join(heatmap_dir, 'fold_{}'.format(fold), 'get_hm_fold_{}'.format(fold), 'heatmap')
        print(hm_dir)
        for k in Path(hm_dir).rglob('{}-{}*.pkl'.format(d_id, xai)):  # get all files contains method or data_id
            fn_segments_list = k.name.split('.')[0].split('-')
            data_id = fn_segments_list[0]
            method_name = fn_segments_list[1]
            if len(fn_segments_list) == 3:
                if fn_segments_list[-1][0] != 'P':
                    continue
            #         print('fn_segments_list', k, fn_segments_list)
            hm = pickle.load(open(k, "rb"))
        #     hm_dict, a, b = get_heatmaps(hm_dir, d_id, by_data = True, hm_as_array= False, return_mri = False)
        #     hm = hm_dict[xai]
        post_hm = postprocess_heatmaps(hm, no_neg=True)
    else:
        post_hm = None
    if label:
        save_dir = Path(save_dir) / '{}-{}-{}-{}'.format(label, fold, d_id, xai)
    else:
        save_dir = Path(save_dir) / '{}-{}-{}'.format(fold, d_id, xai)
    if show_seg:
        seg = seg
    else:
        seg = None
    image3d_to_gif(save_dir = save_dir, mris = mris, heatmaps = post_hm, seg = seg, mi_gt = gt, modalities = modality, keep_png = keep_png, figwidth = figwidth, fontsize = 40, show_img=show_img, cmap = cmap)


#####################
#     MRNet
#####################


def load_modality_image(data_id):
    modalities = ["axial", "sagittal", "coronal"]
    img_path = "/local-scratch/authorid/dld_data/MRNet-v1.0/test/"
    data = []
    for m in modalities:
        data_m = np.load(os.path.join(img_path,m, '{}.npy'.format(data_id)))
        data.append(data_m)
    return data
def load_hm(data_id, hm_dir = None):
    # load all available heatmaps
    hms = dict()
    # hm_dir = '/local-scratch/authorid/log/MRNet/MRNet/1130_161811_fold_6/MRNet_get_hm_fold_6_fold_6/heatmap'
    for k in Path(hm_dir).rglob('{}-*.pkl'.format(data_id)): # get all files contains method or data_id
        fn_segments_list = k.name.split('.')[0].split('-')
        method_name = fn_segments_list[1]
        hm = pickle.load(open(k, "rb"))
        hms[method_name] = hm
    return hms


def normalize_scale_mrnet(single_mod_hm, all_mod_hm, outlier_perc = 1):
    # https://github.com/pytorch/captum/blob/7d21f58371cae981a1707625f85246ad559cea9b/captum/attr/_utils/visualization.py#L41
    scale_factor = _cumulative_sum_threshold(np.abs(all_mod_hm), 100 - outlier_perc)
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = single_mod_hm / scale_factor
    return np.clip(attr_norm, -1, 1)

def postprocess_heatmaps_mrnet(hm_list, no_neg = True, depth_first = True):
    '''

    :param hm_list:
    :param no_neg:
    :param depth_first: whether need the output heatmaps has depth at the 0 dim
    :return:
    '''
    # each modality is : (37, 3, 224, 224), to the same shape as input, list 3 of [ 224, 224, 25]
    # if depth first, do not do transpost. Mathc with original Input shape of (25, 256, 256)
    new_hm_list = []
    # print("postprocess", hm_list[0].shape)
    for hm in hm_list:
        hm = np.squeeze(hm)
        hm = np.mean(hm, axis = 1)

        hm[np.isnan(hm)] = 0
        if no_neg:
            hm[hm <= 0] = 0
        if not depth_first:
            hm = hm.transpose(1,2,0)
            hm = ndimage.gaussian_filter(hm, sigma= [1, 1, 0])
        else:
            hm = ndimage.gaussian_filter(hm, sigma=[0, 1, 1])
        new_hm_list.append(hm)
    if not depth_first:
        all_mod_hm = np.concatenate(new_hm_list, axis = 2)
    else:
        all_mod_hm = np.concatenate(new_hm_list, axis=0)
    norm_hm_list = []
    for hm in new_hm_list:
        norm_hm = normalize_scale_mrnet(hm, all_mod_hm)
        norm_hm_list.append(norm_hm)
    return norm_hm_list


# visualize the input and hms
def visualize_hm_mrnet(data_id, hm_dir = '/local-scratch/authorid/brats_rerun_20220502/visualize/1202_121142_fold_8/MRNet_get_hm_fold_8_fold_8/heatmap',
                       mi_dir='/local-scratch/authorid/project/eval_multimodal_MedIA_result_reporting/MRNet/mi',
                       segment_path = '/local-scratch/authorid/dld_data/Annotation_MRNet/anno_numpy',
                       msfi_fp_mrnet = 'msfi_fp_mrnet.csv', axl_slice=10, sag_slice=10, cor_slice=10, figwidth=30):
    fold= 8
    modalities = ["axial", "sagittal", "coronal"]
    msfi_fp_mrnet = pd.read_csv(msfi_fp_mrnet)
    shapley_csv = Path(mi_dir) / 'seed_{}'.format(fold) /'shapley' / 'fold_{}_modality_shapley.csv'.format(fold)
    df = pd.read_csv(shapley_csv)
    df = df.iloc[0]  # .to_dict('list')
    # print(df)
    gt_shapley = {m: df[m] for m in modalities}
    print(gt_shapley)
    mi_max = max(list(gt_shapley.values()))

    post_hm_dict = dict()
    fontsize = 40
    slice_list = [axl_slice, sag_slice, cor_slice]
    mri_lst = load_modality_image(data_id)
    hm_dict = load_hm(data_id, hm_dir= hm_dir)

    # get seg mask
    seg_list = []
    contour_list = []
    for m in modalities:
        seg_path = os.path.join(segment_path, '{}_{}.npy'.format(data_id, m))
        seg = np.load(seg_path)  # .get_fdata()
        # crop the seg maps the same as in loader.py (25, 256, 256) --> (25, 244, 244)
        pad = int((256 - 224) / 2)
        seg = seg[:, pad:-pad, pad:-pad]
        seg_list.append(seg)
        contour = segmentation.find_boundaries(seg>0)
        contour_list.append(contour)
    c_cmap = cm.winter #cm.gnuplot
    c_cmap.set_under('k', alpha=0)
    b_cmap = cm.binary
    b_cmap.set_under('k', alpha=0)

    xai_seq = list(hm_dict.keys())
    rows = len(hm_dict) + 1
    columns = len(modalities)
    fig = plt.figure(figsize=(figwidth, int(figwidth / columns * (rows + 0.4))))
    ax_placeholder = fig.add_subplot(rows + 1, 1, rows + 1)
    ax_placeholder.axis("off")
    splots = [ax_placeholder]
    #     contour = segmentation.find_boundaries(seg>0)
    c_cmap = cm.winter  # cm.gnuplot
    c_cmap.set_under('k', alpha=0)
    b_cmap = cm.binary
    b_cmap.set_under('k', alpha=0)
    for i in range(1, columns * rows + 1):
        subplot = fig.add_subplot(rows, columns, i)
        mri = mri_lst[(i - 1) % len(modalities)]
        mri = mri.transpose(1, 2, 0)
        if i <= columns:
            subplot.imshow(mri[:, :, slice_list[i % 3]], cmap=plt.get_cmap('gray'),
                           vmin=np.min(mri), vmax=np.max(mri))
            subplot.imshow(contour_list[(i - 1) % columns][slice_list[i % 3], :, :], cmap=c_cmap, alpha = gt_shapley[modalities[(i-1)%columns]]/mi_max, interpolation='none', clim=[0.98, 1])

            # draw seg as contour
        #             contours = []
        #             print(np.unique(seg))
        #             for j in np.unique(seg):
        #                 if j != 0:

        #             for c in contours:
        #             subplot.imshow(contour[:,:, slice_list[i%3]], cmap=c_cmap, alpha = gt[modalities[(i-1)%columns]], interpolation='none', clim=[0.98, 1])
            subplot.set_title("{}\n Modality Importance: {:.2f}\n".format(modalities[(i-1)%columns].upper(), gt_shapley[modalities[(i-1)%columns]] ), y=0.80, zorder = 100, color = 'white', fontsize=fontsize)
        else:
            #             print(i, (i - 1) // columns - 1, (i - 1)% columns)
            method = xai_seq[(i - 1) // columns - 1]
            hm = hm_dict[method]
            msfi_fp = msfi_fp_mrnet.loc[(msfi_fp_mrnet['Fold'] == 8) & (msfi_fp_mrnet['dataID'] == int(data_id)) & (msfi_fp_mrnet['XAI'] == method)]
            msfi =  msfi_fp['msfi'].tolist()[0]
            fp =  msfi_fp['fp'].tolist()[0]
            post_hm = postprocess_heatmaps_mrnet(hm, no_neg=True)
            post_hm_dict[method] = post_hm
            print('size', gt_shapley[modalities[(i-1)%columns]]/mi_max, mi_max,  msfi, fp )
            subplot.imshow(post_hm[(i - 1) % columns][slice_list[i % 3], :, :],
                           # cmap= plt.get_cmap(self.heatmap_cmap),  alpha=self.alpha,
                           cmap=plt.get_cmap('bwr'),  # alpha=np.absolute(scaled_hm_mod[:, :, z]),
                           vmin=-1, vmax=1)
            subplot.imshow(contour_list[(i - 1) % columns][slice_list[i % 3], :, :], cmap=c_cmap, alpha = gt_shapley[modalities[(i-1)%columns]]/mi_max,  interpolation='none', clim=[0.98, 1])

            #             post_hm = postprocess_heatmaps(hm, no_neg = True)
            #             hm_values = get_modality_feature_hm_value(post_hm, seg, portion=True)
            #             msfi_s = 0
            # #             print('hm_values', hm_values, 'gt', gt)
            #             for j, m in enumerate(modalities):
            #                 msfi_s += hm_values[j]* gt[m]
            #             msfi_s /= gt[modalities].sum()

            #             row = rating.loc[(rating['dataID'] == d_id) & (rating['XAI'] ==method)]
            if (i - 1) % columns == 2:
                subplot.set_title("{}   MSFI: {:.2f},   FP: {:.2f}".format(method,msfi, fp), zorder = 100, y=0.90, x =0.9, loc="right", color='black', fontsize=fontsize)

            # if i == 6:
            #     break
        #             positive_hm = postprocess_heatmaps(hm, no_neg = True)
        #         msfi_recalculate = # todo
        splots.append(subplot)
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0,
                        hspace=0)

    # remove the x and y ticks
    for i, ax in enumerate(splots):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig('{}.pdf'.format(data_id), bbox_inches = 'tight')
    return post_hm_dict




if __name__ == '__main__':
    # generate brain masks from brats dataset
    # data_root = '/local-scratch/authorid/dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/'
    data_root = '/scratch/authorid/dld_data/brats2020/MICCAI_BraTS2020_TrainingData/'#'/scratch/authorid/dld_data/brats2020/MICCAI_BraTS_2020_Data_Training/'
    brats_path = os.path.join(data_root, 'all')
    # bgmask_path = '/local-scratch/authorid/dld_data/brainmaskBRATS19'
    bgmask_path = '/scratch/authorid/dld_data/brainmaskBRATS20'
    # bgmask_path = '/local-scratch/authorid/dld_data/tmp'
    # bgmask_path = '/scratch/authorid/dld_data/brainmaskBRATS19'
    get_and_save_bgmask(brats_path, bgmask_path)

