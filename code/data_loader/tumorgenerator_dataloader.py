# tumor sim pipeline
# Generate tumor according to different label, augment tumor and mask
# Get healthy brain, augment
# Combine tumor and brain to put in arbitrary position, with in certain threshold of (overlap area)/(tumor area) to avoid tumor outside brain; avoid tumor across midline
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import glob
import cv2
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import torchvision.utils as vutils
# from torch.autograd import Variable
# import torchvision.models as models
# import torchvision.transforms.functional as TF
import pathlib
import numpy as np
from PIL import Image
from IPython.display import display
import random
import math
import pandas as pd
from monai.transforms import (
    # LoadNiftid,
    # Orientationd,
    RandFlipd,RandFlip,
    RandRotated, RandRotate,
    RandRotate90d, RandRotate90,
    Resized, Resize,
    RandAffined, RandAffine,
    Rand2DElasticd, Rand2DElastic,
    RandShiftIntensityd,
    RandGaussianNoise, RandGaussianNoised,
    # NormalizeIntensityd,
    # Resized,
    ToTensord, ToTensor,
    MapTransform, Flipd, Affine,
    Compose
)
# import albumentations as A
from skimage.morphology import convex_hull_image

import torchvision.transforms as transforms
import monai
from TumorSim.sim_losses import *
# from TumorSim.sim_dataloader import *
from TumorSim.sim_util import *
from TumorSim.sim_model import Inpaint_generator, Tumor_shape, Tumor_grade
from utils.util import prepare_device
import pickle
from datetime import datetime
from .base_data_loader import BaseDataLoader
from pathlib import Path
import logging

# gan_dir = "/local-scratch/authorid/trained_model/GanBrainTumor/model_weight/"
# healthy_brain_dir = "/local-scratch/authorid/BrainTumor/temp_slice"

class GenerateTumorIterFromSavedData(monai.data.CacheDataset):
    def __init__(self, saved_dir):
        self.saved_dir = saved_dir
        self.fns =  [f for f in Path(self.saved_dir).rglob('*.pkl')]

    def __getitem__(self, idx):
        fn = self.fns[idx]
        data = pickle.load(open(fn, "rb"))
        return data
    def __len__(self):
        return len(self.fns)

class GenerateTumorIter(monai.data.CacheDataset):
    def __init__(self,
                 csv_file = 'train',
                 n_gpu = 2,
                 first_gt_align_prob=1,
                 sec_gt_align_prob = 1,
                 healthy_brain_dir="/local-scratch/authorid/dld_data/HealthyBrain2D",
                 brain_trfm = True,
                 tumor_trfm = True,
                 combine_with_brain = [1,1,1,1],
                 input_save_dir = None,
                 gan_dir = "/local-scratch/authorid/trained_model/GanBrainTumor/model_weight/"):
        self.gan_dir = gan_dir
        self.combine_with_brain = combine_with_brain
        self.device, self.device_ids = prepare_device(n_gpu)
        self.input_save_dir = input_save_dir
        self.healthy_brain_dir = healthy_brain_dir
        self.df = pd.read_csv(os.path.join(self.healthy_brain_dir, "{}.csv".format(csv_file)))  # csv: train, val, test. read data filename from csv that Split train 60%, val 10%, test 30%.

        # self.F_test_img_list = glob.glob(healthy_brain_dir + '/F/*.png')
        # self.T1_img_list = glob.glob(healthy_brain_dir + '/T1/*.png')
        # self.T1c_test_img_list = glob.glob(healthy_brain_dir + '/T1c/*.png')
        # self.T2_test_img_list = glob.glob(healthy_brain_dir + '.T2/*.png')
        self.sec_gt_align_prob = sec_gt_align_prob # controlled by experiment to get models with different attention on secondary modality 2.
        self.first_gt_align_prob = first_gt_align_prob

        # select which modality to load, only applicable when use reduced input modality
        # modality = [modality_list[i] for i in range(len(self.input_modality)) if self.input_modality[i]==1]
        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])

        self.brain_trfm = brain_trfm
        self.tumor_trfm = tumor_trfm
        if self.brain_trfm:
            self.brain_trfm = Compose(
                [
                    RandAffined(keys=["brain", "brain_mask"],
                        prob = 0.8,
                        rotate_range = np.pi / 18,
                        translate_range= 10,
                        scale_range = [-0.05, 0.2],
                        padding_mode="zeros"
                        # device = self.device
                    ),
                    Rand2DElasticd(keys=["brain", "brain_mask"],
                        prob=0.5,
                        spacing=(10, 10),
                        magnitude_range=(3, 3),
                        # device=self.device
                    ),
                    # # RandStdShiftIntensity(0.5, prob = 0.3, nonzero= True, channel_wise = True),
                    RandFlipd(keys=["brain", "brain_mask"], prob=0.5, spatial_axis=1),
                    RandGaussianNoised(keys=["brain"], prob=0.2),
                    ToTensord(keys=["brain", "brain_mask"])
                ])
        if self.tumor_trfm:
            self.tumor_trfm = Compose(
                [
                    # RandFlipd(keys=["tumor", "seg"], prob=1),
                    # Flipd(keys=["image", "seg"]),
                    # RandRotate90d(keys=["image", "seg"], prob=1, spatial_axes = (2,3)),
                    RandRotated(keys=["image", "tumor", "seg"], prob=0.5),
                    RandAffined(
                        keys=["image", "tumor", "seg"],
                        prob=0.3,
                        rotate_range=np.pi,
                        shear_range=0,
                        translate_range=0,
                        scale_range= 0,
                        padding_mode="zeros"
                        # device=self.device
                    ),
                    Rand2DElasticd(
                        keys=["image", "tumor", "seg"],
                        prob=0.1,
                        spacing=(1, 1),
                        magnitude_range=(1, 1),
                        # device=self.device
                    ),
                    RandShiftIntensityd(keys=["image", "tumor"], prob=0.3, offsets = 0.5),
                    # RandGaussianNoised(keys=["image", "tumor"], prob=0.2),
                    RandFlipd(keys=["image", "tumor", "seg"], prob=0.8),
                    ToTensord(keys=["image", "tumor", "seg"])
                ])
        if self.input_save_dir:
            dateTimeObj = datetime.now().strftime("%Y%m%d_%H%M")
            self.input_save_dir = Path(self.input_save_dir) / "{}".format(
                dateTimeObj)  # used in validation set, save the input for compare with heatmap gt
            self.input_save_dir.mkdir(parents=True, exist_ok=True)
        logging.info("===Loading TumorGenerator data with first_gt_align_prob = {}, sec_gt_align_prob={}".format(first_gt_align_prob, sec_gt_align_prob))


    def __getitem__(self, index):
        fn  = self.df.loc[index, 'filename']
        # 1. prepare brain img for tumor sim
        T1 = cv2.imread(os.path.join(self.healthy_brain_dir, 'T1', '{}.png'.format(fn)), 0) # Using 0 to read image in grayscale mode
        F = cv2.imread(os.path.join(self.healthy_brain_dir, 'F', '{}.png'.format(fn)), 0)
        T1c = cv2.imread(os.path.join(self.healthy_brain_dir, 'T1c', '{}.png'.format(fn)), 0)
        T2 = cv2.imread(os.path.join(self.healthy_brain_dir, 'T2', '{}.png'.format(fn)), 0)
        img_size = T1.shape[-1]
        # with torch.no_grad():
        F_torch = self.to_tensor(F).reshape(1, 1, img_size, img_size)
        T1_torch = self.to_tensor(T1).reshape(1, 1, img_size, img_size)
        T1c_torch = self.to_tensor(T1c).reshape(1, 1, img_size, img_size)
        T2_torch = self.to_tensor(T2).reshape(1, 1, img_size, img_size)
        brain_torch = torch.cat((F_torch, T1_torch, T1c_torch, T2_torch), 1) # (1,4, 256, 256)
        # print(F.shape, F_torch.shape, brain_torch.shape)
        # transform
        # T1 = self.to_tensor(T1)
        # T2 = self.to_tensor(T2)
        # T1c = self.to_tensor(T1c)
        # F = self.to_tensor(F)
        # brain = torch.stack((F, T1, T1c, T2), dim=0)
        # brain_torch = torch.unsqueeze(brain, 0) # (1,4, 256, 256)

        # 2. generate tumor according to GT label
        gt = random.choice([0,1]) # 0: LGG, 1: HGG

        ran_center = random.sample(range(60, 190), 2)
        [x_center, y_center] = ran_center
        ran_radius = random.sample(range(15, 60), 3) # todo, make tumor not bigger than brain area
        ran_radius.sort()
        [radius3, radius2, radius1] = ran_radius
        circle_para = {"x_center":  x_center,
                       "y_center": y_center,
                       "radius_1": radius1,
                       "radius_2": radius2,
                       "radius_3": radius3
                       } # radius 1-3, inside to outside
        bANDt, tumor, seg, brain_mask= generate_tumor(brain_torch,
                                                      gt = gt,
                                                      first_gt_align_prob  = self.first_gt_align_prob,
                                                      sec_gt_align_prob = self.sec_gt_align_prob,
                                                      circle_para = circle_para,
                                                      device = self.device,
                                                      gan_dir= self.gan_dir)
        # bANDt1, bANDt0, tumor1, tumor0, seg, brain_mask = generate_tumor(brain_torch, gt = gt, circle_para = circle_para, device = self.device)
        # print(bANDt_torch.shape, tumor_torch.shape, grade_mask_torch.shape, brain_mask_torch.shape)
        # print("uni_B", brain_mask.shape, brain_mask.unique())

        # 2.1 tumor and brain augmentation
        brain_torch = torch.squeeze(brain_torch) # (4, 256, 256)
        # brain_mask = np.zeros(brain_torch.shape)
        # brain_mask[np.where(brain_torch > 0)] = 1
        data = {'image': tumor, 'tumor': tumor,  'seg': seg, 'gt': gt, 'brain': brain_torch, 'brain_mask': brain_mask, 'bratsID': fn}
        # print("data['brain_mask'].shape", data['brain_mask'].shape, brain_mask.shape, data['seg'].shape)

        # if self.combine_with_brain:
        data = small_in_big(data)

        if self.brain_trfm:
            data = self.brain_trfm(data)
        if self.tumor_trfm:
            data = self.tumor_trfm(data)

        # 3. output: combine tumor and brain
        # if self.combine_with_brain:
        logging.debug("Combined with brain at getitem: {}".format(self.combine_with_brain))
        # combined_bANDt_mask, displancement_param = small_in_big(seg, brain_mask)
        # combined_bANDt, combined_seg = displancement_param
        combined_bANDt = data['brain'].clone()
        # tumor_4d = torch.unsqueeze(data['tumor'], dim = 0)
        # print(type(tumor), type(data['tumor']), data['tumor'].shape, tumor_4d.shape)
        # print(combined_bANDt.shape,tumor_4d.shape data['tumor'].shape, tumor.shape)
        # aovid the case where tumor is outside brain

        # seg_3d = torch.cat( [data['seg']]*4, 0)
        # tumor_mask = tumor
        for i, value in enumerate(self.combine_with_brain):
            if value == 1:
                combined_bANDt[i][data['seg'][i] > 0] = data['tumor'][i][data['seg'][i] >0]  # [4, 256, 256]
        data['image'] = combined_bANDt
        # also need to transform 'tumor'
        for i, value in enumerate(self.combine_with_brain):
            if value == 0:
                data['tumor'][i][data['tumor'][i]!=0] = 0
                data['seg'][i][data['seg'][i] != 0] = 0
        if self.input_save_dir:
            dateTimeObj = datetime.now().strftime("%Y%m%d_%H%M")
            pickle.dump(data, open(os.path.join(self.input_save_dir, '{}-{}.pkl'.format(fn,dateTimeObj)), 'wb'))
        return data


    def __len__(self):
        return len(self.df)



def generate_tumor(brain_torch,  gt, circle_para, device,
                   first_gt_align_prob = 1.0,
                   sec_gt_align_prob = 1.0,
                   gan_dir="/local-scratch/authorid/trained_model/GanBrainTumor/model_weight/"
                   ):
    '''

    :param brain_torch:  (1,4, 256, 256) (F, T1, T1c, T2) select T1c -2 as the primary mod: 100% align with gt, and F -0 as the secondary mod: xx% align with gt
    :param gt_align_prob: gt_align_prob for secondary modality F - 0
    :param feat: describe generate LGG and HGG according to which distinguish feature ['shape', 'ce', 'edema']
    :param gt: if gt is None, return positive and negative case. Just for test use.
    :return:
    '''
    img_size = brain_torch.shape[-1]
    # load model parameters
    tumor_shape = Tumor_shape().to(device)
    tumor_grade = Tumor_grade().to(device)
    inp_gen = Inpaint_generator().to(device)

    tumor_shape.load_state_dict(torch.load(gan_dir + 'tumor_shape.pth'))
    tumor_grade.load_state_dict(torch.load(gan_dir + 'tumor_grade.pth'))
    inp_gen.load_state_dict(torch.load(gan_dir + 'inp_gen.pth'))
    # Set level circle
    circle = make_level_circle(**circle_para, img_size = img_size)
    # Make a Binary Mask
    with torch.no_grad():

        binary_circle = unify(circle).reshape(1, 1, img_size, img_size)
        binary_circle_torch = binary_circle.to(device)
        uni_B = unify(brain_torch[0][0]).reshape(1, 1, img_size, img_size)
        uni_B_torch = uni_B.to(device)

        binary_mask = unify(tumor_shape(torch.cat([uni_B_torch, binary_circle_torch], 1)).detach().cpu().numpy())
        binary_mask_torch = binary_mask.to(device)

        # Make a Grade Mask
        circle_torch = circle.float().to(device)
        grade_mask = quantize(
            tumor_grade(torch.cat([binary_mask_torch, circle_torch], 1)).detach().cpu().numpy())
        grade_mask_torch = grade_mask.to(device)

        # inpaint
        brain_torch = brain_torch.to(device)
        brain_blank_torch = (brain_torch * (1 - binary_mask_torch))
        # print(brain_torch.shape, binary_mask_torch.shape, brain_blank_torch.shape, grade_mask_torch.shape)
        # print('gt', gt)

        # generate image for gt = 1
        bANDt_torch = inp_gen(brain_blank_torch, grade_mask_torch) #.detach().cpu()
        tumor_torch = bANDt_torch * binary_mask_torch
        grade_mask_torch = torch.cat([grade_mask_torch]*4, dim = 1) # modality wise mask torch.Size([1, 4, 256, 256])
        # print(grade_mask_torch.shape)

        first_rand = np.random.random()
        second_rand = np.random.random()

        # if gt == 0:
        # generate image for gt = 0 class
        circle = circle.to(device)
        bANDt0_torch = inp_gen(brain_blank_torch, binary_circle_torch*0.5)  # .detach().cpu()
        tumor0_torch = bANDt0_torch * binary_circle_torch
        # combine the F and T1c channel with the original generated one
        ## primary modality T1c -2, 100% align
        primary_mod = 2
        sec_mod = 0
        if ((gt == 0) and (first_rand < first_gt_align_prob)) or ((gt == 1) and (first_rand >= first_gt_align_prob)):
            bANDt_torch[:, primary_mod] = bANDt0_torch[:, primary_mod]
            tumor_torch[:, primary_mod] = tumor0_torch[:, primary_mod]
            grade_mask_torch[:, primary_mod] = tumor0_torch[:, primary_mod]>0
        ## secondary modality F - 0 with xx% align
        if ((gt == 0) and (second_rand < sec_gt_align_prob)) or ((gt == 1) and (second_rand >= sec_gt_align_prob)):
            bANDt_torch[:, sec_mod] = bANDt0_torch[:, sec_mod]
            tumor_torch[:, sec_mod] = tumor0_torch[:, sec_mod]
            grade_mask_torch[:, sec_mod] = tumor0_torch[:, sec_mod]>0


        # F_, T1_, T1c_, T2_ = torch.split(bANDt_torch.detach().cpu(), split_size_or_sections=1, dim=1)


        # get rid of extra dim in axis = 1
        bANDt = torch.squeeze(bANDt_torch).detach().cpu() # torch.Size([4, 256, 256])
        tumor = torch.squeeze(tumor_torch).detach().cpu() # torch.Size([4, 256, 256])
        grade_mask = torch.squeeze(grade_mask_torch).detach().cpu() # torch.Size([4, 256, 256])
        # uni_B = torch.squeeze(uni_B_torch, dim=0).detach().cpu()  # torch.Size([1, 256, 256])
        uni_B = torch.squeeze(brain_torch>0, dim=0).detach().cpu()  # torch.Size([1, 256, 256])
    return bANDt, tumor, grade_mask, uni_B

    # return bANDt, tumor, grade_mask, uni_B
def small_in_big(data, threshold = 0.05 ):
    '''
    Position mask  small (tumor) inside mask big (brain). Avoid putting tumor across midline #todo
    Combine tumor and brain to put in arbitrary position, with in certain threshold of (overlap area)/(tumor area) to avoid tumor outside brain
    Return the combined mask, and displacement parameters
    '''
    img_size = data['brain_mask'].shape[-1]
    brain_mask =data['brain_mask'] # (4, 256, 256)
    binary_brain_mask = convex_hull_image(brain_mask[0]) # numpy (256, 256)
    to_tensor = ToTensor()
    binary_brain_mask = to_tensor(np.stack([binary_brain_mask]*4, 0)) #(4, 256, 256)
    binary_tumor_mask = data['seg']>0
    new_binary_tumor_mask = binary_tumor_mask.numpy()
    # determine area outside brain
    area_outside_brain = np.where((binary_brain_mask == 0) & (new_binary_tumor_mask>0), 1, 0)
    out_portion = float(area_outside_brain.sum()/new_binary_tumor_mask.sum())
    if out_portion < threshold:
        # print("no process", out_portion)
        return data
    # print(binary_brain_mask.shape, data['brain_mask'].shape)
    iteration = 0
    min_out_portion = 1
    new_data = data.copy()
    while True:
        x, y = random.sample(range(-img_size+10, img_size-10), 2)
        move = Compose([
            Affine(translate_params=[x, y], padding_mode = "zeros"),
            ToTensor()])
#         print(x,y)
        new_tumor_mask = move(data['seg'])
        new_tumor = move(data['tumor'])
        new_binary_tumor_mask = new_tumor_mask >0
        #  provent move tumor out of the image, or double tumor
        tumor_area_change = int(new_binary_tumor_mask.sum())/ int(binary_tumor_mask.sum())
        if tumor_area_change >1.05 or tumor_area_change <0.95:
            logging.debug('tumor move outside image', int(binary_tumor_mask.sum()),int(new_binary_tumor_mask.sum() ))
            continue
        area_outside_brain = np.where((binary_brain_mask == 0) & (new_binary_tumor_mask>0), 1, 0)
        out_portion = float(area_outside_brain.sum()/new_binary_tumor_mask.sum())
#         print(area_outside_brain.shape, new_binary_tumor_mask.shape, out_portion, )
        if out_portion < threshold:
            min_out_portion = out_portion
            new_data['tumor'] = new_tumor
            new_data['seg'] = new_tumor_mask
            break
        iteration += 1
        if iteration >100:
            threshold += 0.05
            # print('threshold',threshold)
        if out_portion < min_out_portion:
            min_out_portion = out_portion
            new_data['tumor'] = new_tumor
            new_data['seg'] = new_tumor_mask
        if threshold > 0.8:
            # print(min_out_portion, threshold)
            break
    # print("portion", min_out_portion)
    return new_data

# def small_in_big(small, big, avoid_mid_line = True, threshold = 0.1 ):
# def small_in_big(data, threshold = 0.1 ):
#     '''
#     Position mask  small (tumor) inside mask big (brain). Avoid putting tumor across midline #todo
#     Combine tumor and brain to put in arbitrary position, with in certain threshold of (overlap area)/(tumor area) to avoid tumor outside brain
#     Return the combined mask, and displacement parameters
#     '''
#     brain_mask =data['brain_mask']
#     binary_brain_mask = binary_closing(brain_mask) # numpy (1, 256, 256)
#     # print(binary_brain_mask.shape, data['brain_mask'].shape)
#     while True:
#         x, y = random.sample(range(-100, 100), 2)
#         move = Affine(translate_params=[0, x, y])
#         new_tumor_mask = move(data['seg'])
#         new_tumor = move(data['tumor'])
#         binary_tumor_mask = new_tumor_mask >0
#         area_outside_brain = np.where((binary_brain_mask == 0) & (binary_tumor_mask>0), 1, 0)
#         out_portion = area_outside_brain.sum()/binary_tumor_mask.sum()
#         print(area_outside_brain.shape, binary_tumor_mask.shape, out_portion, )
#         if out_portion < threshold:
#             data['tumor'] = new_tumor
#             data['seg'] = new_tumor_mask
#             break
#
#     return data, binary_brain_mask

class TumorSynDataLoader(BaseDataLoader):
    def __init__(self,
                 healthy_brain_dir,
                 input_save_dir,
                 batch_size,
                 brain_trfm= True,
                 csv_file = 'train.csv',
                 sec_gt_align_prob = 0.3,
                 first_gt_align_prob = 1.0,
                 tumor_trfm = True,
                 combine_with_brain = [1,1,1,1],
                 n_gpu = 2,
                 shuffle = True,
                 val_dataset = True,
                 fold = 0,
                 gan_dir =  "/local-scratch/authorid/trained_model/GanBrainTumor/model_weight/"
                 ):
        self.healthy_brain_dir = healthy_brain_dir
        self.input_save_dir = input_save_dir
        self.brain_trfm = brain_trfm
        self.csv_file = csv_file
        self.sec_gt_align_prob = sec_gt_align_prob
        self.first_gt_align_prob = first_gt_align_prob
        self.tumor_trfm = tumor_trfm
        self.combine_with_brain = combine_with_brain
        self.n_gpu = n_gpu
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.val_dataset = val_dataset
        self.gan_dir = gan_dir

        logging.info("Loading TumorSynDataLoader")

    def get_val_loader_presaved(self, saved_dir, batch_size = None, shuffle = None):
        val = GenerateTumorIterFromSavedData(saved_dir)
        if shuffle is not None:
            shuffle = self.shuffle
        if batch_size:
            bs = batch_size
        else:
            bs = self.batch_size
        val_loader = torch.utils.data.DataLoader(val, batch_size= bs , shuffle= shuffle)
        return val_loader




    def get_val_loader(self, val_first_gt_align_prob=None, val_sec_gt_align_prob=None, batch_size = None, save_inputs = None,
                       combine_with_brain = [1,1,1,1]):
        # print("get_val_loader, val_first_gt_align_prob {}, val_sec_gt_align_prob {}".format(val_first_gt_align_prob, val_sec_gt_align_prob))
        if batch_size != None:
            bs = batch_size
        else:
            bs = self.batch_size
        if val_first_gt_align_prob != None:
            first_gt_align_prob = val_first_gt_align_prob
            # print("val_first_gt_align_prob", first_gt_align_prob, val_first_gt_align_prob)
        else:
            first_gt_align_prob = self.first_gt_align_prob
        logging.info("===Validation dataset: 1st modality gt align prob: {} ===".format(first_gt_align_prob))

        if val_sec_gt_align_prob != None:
            sec_gt_align_prob = val_sec_gt_align_prob
            # print("val_first_gt_align_prob", sec_gt_align_prob, val_sec_gt_align_prob)
        else:
            sec_gt_align_prob = self.sec_gt_align_prob
        logging.info("===Validation dataset: 2nd modality gt align prob: {} ===".format(sec_gt_align_prob))

        if self.val_dataset:
            val_or_test = 'val'
            logging.info("===Loading Validation set===")
        else:
            val_or_test = 'test'
            logging.info("===Loading Test set===")


        val = GenerateTumorIter(csv_file ='{}'.format(val_or_test),
                                n_gpu=self.n_gpu,
                                first_gt_align_prob=first_gt_align_prob,
                                sec_gt_align_prob=sec_gt_align_prob,
                                healthy_brain_dir=self.healthy_brain_dir,
                                brain_trfm= None,
                                tumor_trfm = None,
                                combine_with_brain = combine_with_brain,
                                input_save_dir = save_inputs,
                                gan_dir = self.gan_dir
                                )
        val_loader = torch.utils.data.DataLoader(val, batch_size= bs , shuffle=self.shuffle)
        return val_loader

    def get_train_loader(self, save_inputs= None):
        # get train loader
        train = GenerateTumorIter(csv_file ='train',
                                n_gpu=self.n_gpu,
                                first_gt_align_prob=self.first_gt_align_prob,
                                sec_gt_align_prob=self.sec_gt_align_prob,
                                healthy_brain_dir=self.healthy_brain_dir,
                                brain_trfm= self.brain_trfm,
                                tumor_trfm = self.tumor_trfm,
                                # combine_with_brain = self.combine_with_brain,
                                input_save_dir = save_inputs,
                                gan_dir = self.gan_dir
                                )

        train_loader = torch.utils.data.DataLoader(train,batch_size=self.batch_size, shuffle=self.shuffle)
        return train_loader

