import os
import logging
import torch
import pickle
import pandas as pd
import numpy as np
import monai
from monai.transforms import (
    LoadNiftid,
    Compose,
    AsChannelFirstd,
    Orientationd,
    RandFlipd,
    NormalizeIntensityd,
    ScaleIntensityd,
    Compose,
    LoadNiftid,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    ToTensord,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandShiftIntensityd,
    apply_transform,
    CropForegroundd,
    RandRotated
)
from monai.transforms.utils import rescale_array, rescale_instance_array
from torch.utils.data import WeightedRandomSampler
# from biased_data import GenerateBiasedPattern

from .base_data_loader import BaseDataLoader
from pathlib import Path



#########################
#######   BRATS #########
#########################
class BratsIter(monai.data.CacheDataset):
    def __init__(self, csv_file, brats_path, brats_transform, input_modality, label = 'Grade', label_dict = {'LGG':0, 'HGG': 1}, ablated_image_folder = "ablated_brats" ):
        self.transform = brats_transform
        self.image_path = brats_path
        if sum(input_modality) != 4:
            self.ablated_image_path = Path(brats_path).parent / ablated_image_folder  # "zero_ablated_brats"
            print(input_modality, self.ablated_image_path)
        self.df = pd.read_csv(csv_file)
        self.label = label # col name of gt label
        self.label_dict = label_dict
        self.df = self.df[self.df[self.label].notna()]  # get rid of data without gt label
        self.input_modality = input_modality
        # if shuffle:
        #     self.df = self.df.sample(frac=1).reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def  __getitem__(self, idx):
        '''
        Output: 4D torch.Tensor with dimensions (𝐶,𝐷,𝐻,𝑊)
        See: https://torchio.readthedocs.io/data/dataset.html
        '''
        label = self.label_dict[self.df.loc[idx, self.label]]
        bratsID = None
        if self.label == 'IDH':
            bratsID = self.df.loc[idx, 'BraTS19ID']
        elif self.label == 'Grade' or self.label == 'Grade_oversample':
            bratsID = self.df.loc[idx, 'BraTS_2020_subject_ID']
        if bratsID is None:
            assert NotImplementedError("bratsID {} not found".format(bratsID))
            logging.info("bratsID {} {} not found".format(self.df.loc[idx, 'BraTS_2020_subject_ID'], self.label))

        image_path_list = []
        for i in range(4):
            if self.input_modality[i] == 1:
                image_path_list.append(self.image_path)
            else:
                image_path_list.append(self.ablated_image_path)
        # generate file path nii.gz
        T1    = os.path.join(image_path_list[0], bratsID, bratsID+'_t1.nii.gz') # (240, 240, 155)
        T1c   = os.path.join(image_path_list[1], bratsID, bratsID+'_t1ce.nii.gz')
        T2    = os.path.join(image_path_list[2], bratsID, bratsID+'_t2.nii.gz')
        FLAIR = os.path.join(image_path_list[3], bratsID, bratsID+'_flair.nii.gz')
        seg = os.path.join(self.image_path, bratsID, bratsID+'_seg.nii.gz')

        modality_list = [T1, T1c, T2, FLAIR]
        # select which modality to load, only applicable when use reduced input modality
        # modality = [modality_list[i] for i in range(len(self.input_modality)) if self.input_modality[i]==1]

        data = {'image': modality_list, 'seg': seg, 'gt':label, 'bratsID':bratsID}

        if self.transform is not None:
            data = apply_transform(self.transform, data)
        return data

class FGScaleIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensity`.
    Scale the intensity of input image to the given value range (minv, maxv).
    If `minv` and `maxv` not provided, use `factor` to scale image by ``v = v * (1 + factor)``.
    Only scale foregraound intensity, with nonzero image mask
    """
    def _FGscale(self, img):
        """
        Apply the transform to `img`.

        Raises:
            ValueError: When ``self.minv=None`` or ``self.maxv=None`` and ``self.factor=None``. Incompatible values.

        """
        slices = (img != 0)
        if not np.any(slices):
            return img
        return rescale_instance_array(img[slices], 0.001, 1.0, img.dtype)
        # elif self.factor is not None:
        #     return (img[slices] * (1 + self.factor)).astype(img.dtype)
        # else:
        #     raise ValueError("Incompatible values: minv=None or maxv=None and factor=None.")

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self._FGscale(d[key])
        return d

# class AblateModality(MapTransform):
#     """
#     Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensity`.
#     Scale the intensity of input image to the given value range (minv, maxv).
#     If `minv` and `maxv` not provided, use `factor` to scale image by ``v = v * (1 + factor)``.
#     Only scale foregraound intensity, with nonzero image mask
#     """
#     def _FGscale(self, img, seg):
#         """
#         Apply the transform to `img`.
#
#         Raises:
#             ValueError: When ``self.minv=None`` or ``self.maxv=None`` and ``self.factor=None``. Incompatible values.
#
#         """
#         slices = (img != 0)
#         if not np.any(slices):
#             return img
#         return rescale_instance_array(img[slices], 0.001, 1.0, img.dtype)
#         # elif self.factor is not None:
#         #     return (img[slices] * (1 + self.factor)).astype(img.dtype)
#         # else:
#         #     raise ValueError("Incompatible values: minv=None or maxv=None and factor=None.")
#
#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             d[key] = self._FGscale(d[key])
#         return d

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = list()
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(np.logical_or(np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class BRATSDataLoader(BaseDataLoader):
    def __init__(self, task_name, input_size, over_sample, data_dir, roi, fold, batch_size, shuffle, weighted_sampler, input_modality = [1,1,1,1],
                 pattern_dict=None, gt_align_prob=None, slice_wise=False, combine_with_brain=True, val_dataset = True,
                 ):
        self.data_dir = data_dir
        self.input_size = tuple([input_size]*3)
        self.roi = roi
        self.fold = fold
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weighted_sampler = weighted_sampler
        self.input_modality = input_modality
        self.val_dataset = val_dataset

        self.bias_kwargs = {
            'pattern_dict': pattern_dict,
            'gt_align_prob': gt_align_prob,
            'slice_wise': slice_wise,
            'combine_with_brain': combine_with_brain
        }
        if gt_align_prob is not None:
            logging.info("get_biased_brats, BratsIter:: fold = {}. Pattern dict: {}. GT align prob: {}".format(self.fold,pattern_dict, gt_align_prob))
        else:
            logging.info("BratsIter:: fold = {}".format(self.fold))

        # select the prediction task
        self.label = None
        self.label_dict = {}
        if task_name.upper() == 'BRATS_IDH':
            self.label = 'IDH' #gt label col name
            self.label_dict = {'Mutant': 1, 'wt': 0}
        elif task_name.upper() == 'BRATS_HGG':
            if over_sample:
                self.label = 'Grade_oversample'
            else:
                self.label = 'Grade'
            self.label_dict = {'LGG': 0, 'HGG': 1}
        # used for heatmap randomized label heatmap exp
        elif task_name.upper() == 'RANDOM_IDH':
            self.label = 'RANDOM_IDH'
            self.label_dict = {'Mutant': 1, 'wt': 0}
        else:
            assert NotImplementedError("iter {} not found".format(task_name))

    def get_val_loader(self, batch_size = None, modality_selection = None, ablated_image_folder = None):
        if batch_size:
            bs = batch_size
        else:
            bs = self.batch_size
        # load val transform
        if self.bias_kwargs['gt_align_prob'] is not None:
            val_transform = Compose(
                [
                    LoadNiftid(keys=["image", "seg"]),
                    ConvertToMultiChannelBasedOnBratsClassesd(keys=["seg"]),
                    GenerateBiasedPattern(**self.bias_kwargs),
                    ScaleIntensityd(keys=["image", "pattern", "combined"]),
                    NormalizeIntensityd(keys=["image", "pattern", "combined"], nonzero=True, channel_wise=True),
                    Resized(keys=["image", "seg", "bb", "pattern", "combined"], spatial_size=self.input_size),
                    ToTensord(keys=["image", "seg", "bb", "pattern", "combined"]),
                ]
            )
        else:
            if self.roi:
                val_transform = Compose(
                    [
                        LoadNiftid(keys=["image", "seg"]),
                        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
                        # Orientationd(keys=["image", "seg"], axcodes="RAS"),
                        ScaleIntensityd(keys=["image"]),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        CropForegroundd(keys=["image", "seg"], source_key="seg", margin=5),
                        Resized(keys=["image", "seg"], spatial_size=self.input_size),
                        ToTensord(keys=["image", "seg"])
                    ]
                )
            else:
                val_transform = Compose(
                    [
                        LoadNiftid(keys=["image", "seg"]),
                        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
                        # Orientationd(keys=["image", "seg"], axcodes="RAS"),
                        ScaleIntensityd(keys=["image"]),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        # FGScaleIntensityd(keys=["image"]),
                        Resized(keys=["image", "seg"], spatial_size=self.input_size),
                        ToTensord(keys=["image", "seg"]),
                    ]
                )
        if self.val_dataset:
            val_or_test = 'val'
            logging.info("===Loading Validation set===")
        else:
            val_or_test = 'test'
            logging.info("===Loading Test set===")

        if modality_selection:
            input_modality = modality_selection
        else:
            input_modality = self.input_modality

        val = BratsIter(csv_file=os.path.join(self.data_dir, self.label, '{}_fold_{}.csv'.format(val_or_test, self.fold)),
                        brats_path=os.path.join(self.data_dir, 'all'),
                        label=self.label,
                        label_dict=self.label_dict,
                        brats_transform=val_transform,
                        input_modality= input_modality,
                        ablated_image_folder = ablated_image_folder)

        val_loader = torch.utils.data.DataLoader(val, batch_size= bs , shuffle=self.shuffle)
        return val_loader

    def get_train_loader(self):
        # get train loader
        if self.bias_kwargs['gt_align_prob'] is not None:  # load biased dataset

            train_transform = Compose(
                [
                    # load 4 Nifti images and stack them together
                    LoadNiftid(keys=["image", "seg"]),
                    ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
                    GenerateBiasedPattern(**self.bias_kwargs),
                    RandFlipd(keys=["image", "seg", "bb", "pattern", "combined"], prob=0.8),
                    RandRotated(keys=["image", "seg", "bb", "pattern", "combined"], range_x=30.0, range_y=30.0,
                                range_z=30.0, prob=0.8),
                    ScaleIntensityd(keys=["image", "pattern", "combined"]),
                    NormalizeIntensityd(keys=["image", "pattern", "combined"], nonzero=True, channel_wise=True),
                    Resized(keys=["image", "seg", "bb", "pattern", "combined"], spatial_size=self.input_size),
                    ToTensord(keys=["image", "seg", "bb", "pattern", "combined"]),
                ]
            )
        else:
            if self.roi:
                train_transform = Compose(
                    [
                        # load 4 Nifti images and stack them together
                        LoadNiftid(keys=["image", "seg"]),
                        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
                        # Orientationd(keys=["image", "seg"], axcodes="RAS"),
                        RandFlipd(keys=["image", "seg"], prob=0.8),
                        RandRotated(keys=["image", "seg"], range_x=0.5, range_y=0.5, range_z=0.5, prob=0.8),
                        ScaleIntensityd(keys=["image"]),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        CropForegroundd(keys=["image", "seg"], source_key="seg",
                                        margin=[int(np.random.random() * 30 + 5), int(np.random.random() * 30 + 5),
                                                int(np.random.random() * 30 + 5)]),
                        Resized(keys=["image", "seg"], spatial_size=self.input_size),
                        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                        ToTensord(keys=["image", "seg"]),
                    ]
                )
            else:  # whole brain train transform
                train_transform = Compose(
                    [
                        # load 4 Nifti images and stack them together
                        LoadNiftid(keys=["image", "seg"]),
                        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
                        # Orientationd(keys=["image", "seg"], axcodes="RAS"),
                        RandFlipd(keys=["image", "seg"], prob=0.8),
                        RandRotated(keys=["image", "seg"], range_x=0.5, range_y=0.5, range_z=0.5, prob=0.8),
                        ScaleIntensityd(keys=["image"]),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        # ScaleIntensityd(keys=["image"]),
                        Resized(keys=["image", "seg"], spatial_size=self.input_size),
                        # RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                        # RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                        ToTensord(keys=["image", "seg"]),
                    ]
                )

        csv_file = os.path.join(self.data_dir, self.label, 'train_fold_{}.csv'.format(self.fold))
        train = BratsIter(csv_file=csv_file,
                          brats_path=os.path.join(self.data_dir, 'all'),
                          label=self.label,
                          label_dict=self.label_dict,
                          brats_transform=train_transform,
                          input_modality= self.input_modality)
        # weighted random sampler for training data
        if self.weighted_sampler:
            train_loader = torch.utils.data.DataLoader(train,batch_size=self.batch_size, sampler=self.__get_weighted_sampler(csv_file))
        else:
            train_loader = torch.utils.data.DataLoader(train,batch_size=self.batch_size, shuffle=self.shuffle)
        return train_loader

    def __get_weighted_sampler(self, csv_file):
        df = pd.read_csv(csv_file)
        y_train = df[self.label]
        labels = [self.label_dict[t] for t in y_train]
        labels = np.array(labels)
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        class_sample_probabilities = 1. / class_sample_count
        sample_probabilities = np.array([class_sample_probabilities[t] for t in labels])
        sample_probabilities = torch.from_numpy(sample_probabilities)
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_probabilities.type('torch.DoubleTensor'),
                                                         num_samples=len(sample_probabilities), replacement=True)
        return sampler


if __name__ == "__main__":
    import time
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('start test iterator_brats')
    data_root = '/local-scratch/authorid/dld_data/brats2020/MICCAI_BraTS2020_TrainingData'
    fold = 1
    over_sample = False
    csv_file = os.path.join(data_root, 'Grade','train_fold_{}.csv'.format(fold) )
    name = 'BRATS_HGG'
    roi = False
    train, val = get_data(name, over_sample, roi, data_root, 1)
    for i in range(0, 3):
        img, seg, lab, bratsID = val.__getitem__(i)['image'],val.__getitem__(i)['seg'], val.__getitem__(i)['gt'], val.__getitem__(i)['bratsID']
        logging.info('max: {}, min: {}, mean: {}, std: {}'.format(img.max(), img.min(), img.mean(), img.std()))
        logging.info("{}: {}, {}".format(i, img.shape, seg.shape))
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=1, shuffle=True,
                                               num_workers=12, pin_memory=True)
