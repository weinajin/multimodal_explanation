import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
# from  skimage import draw
import sys
import os
import logging
# from data import BratsIter, ConvertToMultiChannelBasedOnBratsClassesd
from heatmap_utlis import load_mri, ImageSliceViewer3D, read_bgmask_from_id
from skimage.draw import rectangle
from skimage.color import rgb2gray
from monai.transforms import (
    # LoadNiftid,
    # Orientationd,
    # RandFlipd,
    # RandRotated,
    # NormalizeIntensityd,
    # Resized,
    # ToTensord,
    MapTransform,
    Compose
)
# import torch


# bgmask_path = '../../../tmp'


# def get_biased_brats(name, over_sample, data_root, fold, batch_size,
#                      pattern_dict, gt_align_prob, presaved, slice_wise=False, combine_with_brain = True,
#                      val_only=False,
#                       # data_root='/scratch/authorid/dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/',
#                       # data_root='/local-scratch/authorid/dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/',
#                       **kwargs):
#     """
#     data iterator for brats
#     val_only: if True, only get validation dataloader
#     """
#     logging.debug("get_biased_brats, BratsIter:: fold = {}".format(fold))
#     # args for transforms
#     # d_size, h_size, w_size = 155, 240, 240
#     # input_size = (64, 64, 64)
#     input_size = (128, 128, 128)
#     # spacing = (d_size/input_size[0], h_size/input_size[1], w_size/input_size[2])
#     # Mean, Std, Max = read_brats_mean(fold, data_root)
#
#     val_transform = Compose(
#         [
#             LoadNiftid(keys=["image", "seg"]),
#             ConvertToMultiChannelBasedOnBratsClassesd(keys=["seg"]),
#             GenerateBiasedPattern(pattern_dict=pattern_dict, gt_align_prob=gt_align_prob, presaved =  presaved,
#                                   slice_wise=slice_wise, combine_with_brain=combine_with_brain),
#             NormalizeIntensityd(keys=["image", "pattern", "combined"], nonzero=True, channel_wise=True),
#             Resized(keys=["image", "seg", "bb", "pattern", "combined"], spatial_size=input_size),
#             ToTensord(keys=["image", "seg", "bb", "pattern", "combined"]),
#         ]
#     )
#     # select the prediction task
#     label = None
#     label_dict = {}
#     if name.upper() == 'BRATS_IDH':
#         label = 'IDH'
#         label_dict = {'Mutant': 1, 'wt':0}
#     elif name.upper() == 'BRATS_HGG' :
#         if over_sample:
#             label = 'Grade_oversample'
#         else:
#             label = 'Grade'
#         label_dict = {'LGG': 0, 'HGG': 1}
#     # used for heatmap randomized label heatmap exp
#     elif name.upper() == 'RANDOM_IDH':
#         label = 'RANDOM_IDH'
#         label_dict = {'Mutant': 1, 'wt': 0}
#     else:
#         assert NotImplementedError("iter {} not found".format(name))
#
#     val   = BratsIter(csv_file=os.path.join(data_root, label, 'val_fold_{}.csv'.format(fold)),
#                       brats_path = os.path.join(data_root, 'all'),
#                       label=label,
#                       label_dict=label_dict,
#                       brats_transform=val_transform,
#                       shuffle=False)
#
#     val_loader = torch.utils.data.DataLoader(val,
#         batch_size=batch_size, shuffle=False)
#     if val_only:
#         return val_loader
#
#     train_transform = Compose(
#         [
#             # load 4 Nifti images and stack them together
#             LoadNiftid(keys=["image", "seg"]),
#             ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
#             GenerateBiasedPattern(pattern_dict = pattern_dict, gt_align_prob= gt_align_prob, presaved = presaved, slice_wise = slice_wise, combine_with_brain = combine_with_brain),
#             RandFlipd(keys=["image", "seg", "bb", "pattern", "combined"], prob=0.8),
#             RandRotated(keys=["image", "seg", "bb", "pattern", "combined"], range_x=30.0, range_y=30.0, range_z=30.0, prob=0.8),
#             NormalizeIntensityd(keys=["image", "pattern", "combined"], nonzero=True, channel_wise=True),
#             Resized(keys=["image", "seg", "bb", "pattern", "combined"], spatial_size=input_size),
#             ToTensord(keys=["image", "seg", "bb", "pattern", "combined"]),
#         ]
#     )
#     csv_file = os.path.join(data_root, label, 'train_fold_{}.csv'.format(fold))
#     train = BratsIter(csv_file=csv_file,
#                       brats_path = os.path.join(data_root, 'all'),
#                       label = label,
#                       label_dict = label_dict,
#                       brats_transform=train_transform,
#                       shuffle=True)
#     # weighted random sampler for training data
#     df = pd.read_csv(csv_file)
#     y_train = df[label]
#     labels = [label_dict[t] for t in y_train]
#     labels = np.array(labels)
#     class_sample_count = np.array([len(np.where(labels==t)[0]) for t in np.unique(labels)])
#     class_sample_probabilities = 1./class_sample_count
#     sample_probabilities = np.array([class_sample_probabilities[t] for t in labels])
#     sample_probabilities = torch.from_numpy(sample_probabilities)
#     sampler =  torch.utils.data.WeightedRandomSampler(weights = sample_probabilities.type('torch.DoubleTensor'), num_samples = len(sample_probabilities), replacement = True)
#
#     train_loader = torch.utils.data.DataLoader(train,
#         batch_size=batch_size, sampler = sampler,
#         drop_last=True)
#     return (train_loader, val_loader)





# data transform to generate biased pattern from mri and its brain masks
class GenerateBiasedPattern(MapTransform):
    def __init__(self, pattern_dict, gt_align_prob, slice_wise=False, combine_with_brain = True):
        '''
        :param pattern_dict: a dict, {0: pat1, 1: pat2} pattern = 'shape', 'text', 'ruler', 'skull', 'blank'
        :param gt_align_prob: how noisy t.he pattern is to align with gt label. 1 - 100% alignment, no noise. between 0.5-1
        :param slice_wise: if True, create pattern for each 2D slices. False: the pattern across axial slices is the same 2D image
        :param combine_with_brain: if False, generate pattern only.
        '''
        # super().__init__(keys)
        self.pattern_dict = pattern_dict
        self.gt_align_prob = gt_align_prob
        self.slice_wise = slice_wise
        self.combine_with_brain = combine_with_brain
        # self.presaved = presaved # presave the generated patterns in disk

    def __call__(self, data):
        d = dict(data)
        # only reverse the gt then probability above threshold (defined as self.gt_align_prob)
        # assume binary classification
        pattern = None
        if np.random.uniform() < self.gt_align_prob:
            pattern = self.pattern_dict[d['gt']]
        else:
            pattern = self.pattern_dict[1- d['gt']] # get opposite gt
        # load the pre-saved brain mask
        # for key in self.keys:
        #     print(key, self.keys, d['image'].shape)
        if self.combine_with_brain:
            d['bb'], d['pattern'], d['combined'] = combine_pattern_with_brain(mri_lst= d['image'], bratsID=d['bratsID'], pattern=pattern,
                                                                              slice_wise=self.slice_wise)
        else:
            d['bb'], d['pattern'] = combine_pattern_with_brain(bratsID=d['bratsID'], pattern=pattern,
                                                                              slice_wise=self.slice_wise)
            d['combined'] = np.zeros((4, 240, 240, 155))
        # logging.info(d.keys())
        return d

# ===helper function to construct biased images===
# # construct biased model to add to the brain MRI bg
# 1. synethisis data, generate using diffrent parameters
# 2. train a classifier on the sythetic data
#
# ## generate biased data
# 1. get brain mask
# 2. create random bounding box with random position
# 3. based on axial img (axis2): to make sure the bb is outside brain region, calculate the overlap between bb and mask, use the bb if pass the overlap threshold; also make sure the bb has enough area.
# 4. use the bb as mask, generate different types of pattern within the bounding box
#     - combine the generated bg pattern with fg brain, make sure their pixel distriubtion histograms are the same level.
# <del>(2. generate pattern image of the same input size, need to make sure the major content is outside the mask region
# <del>3. add two image together by mask)
# 5. use parameter to control the correspondance to gt label



def generate_raw_bb(img_size = (240, 240)):
    '''
    The bbox can be a single one or multiple combined one
    :param img_size:
    :return:
    '''
    num_bb = np.random.choice(range(3))
    offset = 20
    raw_bb = np.zeros(img_size)
    for i in range(num_bb):
        start = (np.random.choice(range(img_size[0]-offset)), np.random.choice(range(img_size[1]-offset)))
        extent = (np.random.choice(range(offset, img_size[0])), np.random.choice(range(offset, img_size[1])))
        rr, cc  = rectangle(start, extent=extent, shape=img_size)
        raw_bb[rr, cc] = 1
    return raw_bb

# 3. based on axial img (axis2): 
# to make sure the bb is outside brain region, 
# calculate the overlap between bb and mask,
# use the bb if pass the overlap threshold; also make sure the bb has enough area.
def get_valid_bb4d(bg_masks):
    '''
    Valid bounding box outside brain region is labelled as 1
    '''
    img_size = (240, 240)
    bb_masks = []
    for m in bg_masks:
        bb_slices = []
        for s in range(m.shape[2]):
            bg = m[:,:,s]
            valid_bb = np.zeros(bg.shape)
            while valid_bb.sum() < 1000:
                raw_bb = generate_raw_bb()
                valid_bb = np.where( (raw_bb == 1) & (bg == 0), 1, 0)
            bb_slices.append(valid_bb)
        bbs = np.dstack(bb_slices)
        bb_masks.append(bbs)
    return bb_masks

def get_valid_bb3d(bg_mask3d):
    '''
    Valid bounding box outside brain region is labelled as 1
    '''
    img_size = (240, 240)
    bb_slices = []
    for s in range(bg_mask3d.shape[2]):
        bg = m[:,:,s]
        valid_bb = np.zeros(bg.shape)
        while valid_bb.sum() < 1000:
            raw_bb = generate_raw_bb()
            valid_bb = np.where( (raw_bb == 1) & (bg == 0), 1, 0)
        bb_slices.append(valid_bb)
    bb_mask3d = np.dstack(bb_slices)
    return bb_mask3d

def get_valid_bb2d(bg_mask2d):
    '''
    Valid bounding box outside brain region is labelled as 1
    '''
    img_size = (240, 240)
    valid_bb = np.zeros(bg_mask2d.shape)
    while valid_bb.sum() < 1000:
        raw_bb = generate_raw_bb()
        valid_bb = np.where( (raw_bb == 1) & (bg_mask2d == 0), 1, 0)
    return valid_bb

def range_parse(rng):
    min,max = rng.split("-")
    return "".join(chr(i) for i in range(ord(min),ord(max)+1))

def rand_word():
    lower = range_parse("a-z")
    upper = range_parse("A-Z")
    number = range_parse("0-9")

    word = ''
    i =0
    length = np.random.choice(range(1,20))
    while i < length:
        r = np.random.uniform()

        if r <0.4:
            word += lower[np.random.choice(len(lower))]
        elif r > 0.6:
            word += upper[np.random.choice(len(upper))]
        else:
            word += number[np.random.choice(len(number))]
        i +=1
    return word

def rand_line():
    words_list = ['name', 'gender', 'id', 'mri', 'hospital', 'radiology', 'patient', 'male', 'female', 'age', 'year', 'month', 'brain', 'day', 'date', 'ge', 'philips', 'siemens', 'neuro']

    line = ''
    length = np.random.choice(range(1,50))
    i = 0
    while i < length:
        r = np.random.uniform()
        if r < 0.5:
            line += words_list[np.random.choice(len(words_list))] + ' '
        else:
            line += rand_word() + ' '
        i += 1
    return line

def rand_para():
    para = ''
    length = np.random.choice(range(50,100))
    i = 0
    while i < length:
        para += rand_line() + ' \n '
        i += 1
    return para

def generate_pattern(image_size = (240, 240), filename='tmp', pattern = 'text', save= False):
    '''
    Generate image full of text with the given size, and rotate the image with random degree.
    Return 2D image of image_size
    param: pattern = text, ruler, shape, skull
    '''
    def draw_ruler():
        length = int(image_size[0]/np.random.choice(range(3,5)))
        width = np.random.choice(range(2,5))
        line_len = np.random.choice(range(5,15))
        verticle = np.random.choice([True, False])
        if verticle:
            start = (np.random.choice(range(0, image_size[0]-line_len-width)), np.random.choice(range(0, image_size[1]-length)) )
            end = (start[0], start[1]+ length)

        else: 
            start = (np.random.choice(range(0, image_size[0]-length)), np.random.choice(range(0, image_size[1]-line_len-width)) )
            end = (start[0]+length, start[1])

        # draw ruler axis with thickness (width)
        draw.line([start, end], width = width, fill = tuple(np.random.choice(range(256), size=3)))
    #     img[rr, cc] = 1
        mark_num = np.random.choice(range(8,20)) 
        interval = int(np.ceil(length / mark_num ))
        # draw ruler
        if verticle:
            for i in range(mark_num+1):                
                sx, sy = start[0], start[1]+i*interval
                if sx<image_size[0] and sy<image_size[1]:
                    if i%5 ==0:
                        draw.rectangle([(sx, sy), (sx+line_len+5, sy)], width = 1)
                    else:
                        draw.rectangle([(sx, sy), (sx+line_len, sy)], width = 1)
                else:
                    break
        else:
            for i in range(mark_num+1):                
                sx, sy = start[0]+i*interval, start[1]
                if sx<image_size[0] and sy<image_size[1]:
                    if i%5 ==0:
                        draw.rectangle([(sx, sy), (sx, sy+line_len+5)], width = 1)
                    else:
                        draw.rectangle([(sx, sy), (sx, sy+line_len)], width = 1)
                else:
                    break
    # end of draw_ruler()
    string = rand_para()
    img = Image.new('RGB', image_size)
    draw = ImageDraw.Draw(img)
    assert pattern.lower() in ['text', 'ruler', 'shape'], print('ERROR in biased_model: no pattern specified! Pattern should be in [text, ruler, shape, skull]')
    if pattern.lower() == 'ruler':
        n = 15
        while n > 0:
            draw_ruler()
            n -= 1
    elif pattern.lower() == 'text':
        text_color = tuple(np.random.choice(range(256), size=3))

        draw.text((0,0), string, fill=text_color)
    elif pattern.lower() == 'shape':
        for i in range(100):
            draw.regular_polygon(bounding_circle=(int(np.random.choice(image_size[0])), 
                                                  int(np.random.choice(image_size[1])), 
                                                  int(np.random.choice(range(1,15)))), \
                                 n_sides = int(np.random.choice(range(3,10))), rotation = np.random.choice(359), fill= tuple(np.random.choice(range(256), size=3)))
            if np.random.uniform() < 0.2:
                x1, y1 = int(np.random.choice(image_size[0])),int(np.random.choice(image_size[1]))
                x2, y2 = x1 + int(np.random.choice(range(1, 5))), y1+int(np.random.choice(range(1, 5)))
                draw.ellipse([(x1, y1),(x2,y2)], fill = tuple(np.random.choice(range(256), size=3)))
    
    if save:
        img.save('{}.png'.format(filename))
    img = rgb2gray(np.array(img)) # convert to grayscale 2D image, random gray intensity
    # random rotate90 the image, and flip
    random_level = 0.2
    if np.random.uniform() > random_level:
        img = np.flip(img, axis = 0)
    if np.random.uniform() > random_level:
        img = np.flip(img, axis = 1)
    if np.random.uniform() > random_level:
        img = np.rot90(img, k = np.random.choice([1,2,3]))
    assert np.array(img).shape == image_size, 'ERROR! Generated image is not in size {}'.format(np.array(img).shape)
    return img


def generate_masked_pattern(bb_mask, pattern = 'text' ):
    '''
    I/O: 2D image 
    params: pattern_func: generate_text_image , or generate_ruler_image
    return: pattern_bb: [C, H,W,D] use bb_masks and put center of patterned image over the mask
    '''
    pattern_img = generate_pattern(pattern = pattern)
    assert bb_mask.shape == np.array(pattern_img).shape, 'ERROR! bb_mask slice shape of {} does not match pattern_img {}'.format(bb_mask.shape, np.array(pattern_img).shape)
    masked_pattern = np.where(bb_mask == 1, pattern_img, 0)
    return masked_pattern



def combine_mri_pattern(mri, masked_pattern, mri_max):
    '''
    Receive input of 2D/3D image
    '''
    assert mri.shape == masked_pattern.shape, 'ERROR! MRI slice shape of {} does not match masked_pattern {}'.format(mri.shape, masked_pattern.shape)
    # convert masked_pattern to the same type of mri
    masked_pattern = masked_pattern.astype(mri.dtype) 
    # normalize masked_pattern to the same scale of the mri modality
    pattern_alone2d = masked_pattern * mri_max
    combined2d = mri + pattern_alone2d
    return combined2d, pattern_alone2d

def combine_pattern_with_brain(mri_lst = None, bratsID = None, pattern = None, slice_wise = False, bgmask_path = '/scratch/authorid/dld_data/brainmaskBRATS20'):
    '''
    Generate pattern 
    params: slice_wise: True: generate 2D image patterns according to axial slices, or False: the same 2D images across all axial slices
    mri_lst: image array of 4D. a list of 3D image array. if not None, generate the combined image of brain + pattern. Else if None: return pattern only based on the brain masks.
    bratsID: used to get the saved brain masks
    pattern: generate one of the ['text', 'ruler', 'shape','skull', 'blank']. Support the blank pattern with no image patterns generated.
    return: image array of 4D: bbox mask4d, pattern4d, combined_mri4d
    '''
    # normalize the pattern_bb to be the same distribution of each mri modality
    # save the normalized pattern before combine
    # if combined_with_brain:
    #     assert mri_lst, print("ERROR! combined_with_brain is %r, but no mri provided!" % (bool(combined_with_brain)))
    assert pattern.lower() in ['text', 'ruler', 'shape','skull', 'blank'], print('ERROR in biased_model: no pattern specified! Pattern should be in [text, ruler, shape,skull, blank]')
    if pattern.lower() == 'blank':
        if mri_lst is not None:
            return np.zeros((4, 240, 240, 155)), np.zeros((4, 240, 240, 155)), mri_lst
        return np.zeros((4, 240, 240, 155)), np.zeros((4, 240, 240, 155))
    # get the saved brain mask from bratsID, to save the time in computing masks from mri_lst
    if slice_wise:
        bg_mask4d, skull_lst = read_bgmask_from_id(bgmask_path, bratsID,  slice_wise = slice_wise)
    else: # if not generating individual patterns for each slices, then only the 2D slice with max brain area is needed
        bg_mask2d, skull_lst = read_bgmask_from_id(bgmask_path, bratsID, slice_wise = slice_wise)
    if pattern.lower() =='skull' and not mri_lst: # skull pattern only. if skull, slice_wise = True
        return np.stack(skull_lst), np.stack(skull_lst) # 4d image array
    pattern_alone4d = []
    combined4d = []
    bb4d = []
    Channel, W, H, Depth = len(skull_lst), skull_lst[0].shape[0], skull_lst[0].shape[1], skull_lst[0].shape[2]
    for i in range(Channel):
        if mri_lst is not None:
            mri3d = mri_lst[i]
            # get the mean std of mri
            mri_max = mri3d.max()
        pattern_alone3d = []
        combined3d = []
        bb3d = []
        if pattern.lower() in ['text', 'ruler', 'shape']:
            if slice_wise:
                for s in range(Depth):
                    bb_mask2d = get_valid_bb2d(bg_mask4d[i][:,:,s])
                    bb3d.append(bb_mask2d)
                    pattern_alone2d = generate_masked_pattern(bb_mask2d, pattern = pattern)
                    if mri_lst is not None:
                        mri2d = mri3d[:,:,s]
                        combined2d, pattern_alone2d = combine_mri_pattern(mri2d, pattern_alone2d, mri_max)
                        combined3d.append(combined2d)
                    pattern_alone3d.append(pattern_alone2d)
                pattern_alone3d = np.dstack(pattern_alone3d)
                bb3d = np.dstack(bb3d)
                if mri_lst is not None:
                    combined3d = np.dstack(combined3d)
            else:
                # generate a single 2D image to combine with all axial slices
                bb_mask = get_valid_bb2d(bg_mask2d[i])
                bb3d = np.dstack([bb_mask]* Depth) # it's actually 2d , same stacked together
                # select the 2D image that has the max out of brain info
                max_masked_pattern = None
                j = 50
                max_area = 0
                while j >0:
                    masked_pattern = generate_masked_pattern(bb_mask, pattern = pattern)
                    area = (masked_pattern>0).sum()
                    if area > max_area:
                        max_area = area
                        max_masked_pattern = masked_pattern
                    j -= 1
                # stack the 2D masked pattern to 3D
                pattern_alone3d = np.dstack([max_masked_pattern]*Depth)
                if mri_lst is not None:
                    # combine 3D images directly
                    combined3d, pattern_alone3d = combine_mri_pattern(mri3d, pattern_alone3d, mri_max)
        elif pattern.lower() == 'skull':
            if mri_lst is not None:
                # combine 3D images directly
                combined3d, pattern_alone3d = combine_mri_pattern(mri3d, skull_lst[i], mri_max)
        pattern_alone4d.append(pattern_alone3d)
        bb4d.append(bb3d)
        if mri_lst is not None:
            combined4d.append(combined3d)
    if mri_lst is not None:
        return np.stack(bb4d), np.stack(pattern_alone4d), np.stack(combined4d)
    return np.stack(bb4d), np.stack(pattern_alone4d)


if __name__ == '__main__':
    # generate brain masks from brats dataset
    data_root = '/local-scratch/authorid/dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/'
    brats_path = os.path.join(data_root, 'all')
    import time
    # logging.getLogger().setLevel(logging.DEBUG)
    logging.info('start test iterator_brats')
    fold = 1
    batch_size = 2
    over_sample = False
    csv_file = os.path.join(data_root, 'Grade','train_fold_{}.csv'.format(fold) )
    name = 'BRATS_IDH'
    pattern_dict = {0: 'text', 1: 'shape'}
    gt_align_prob = 1
    presaved= False
    (train_loader, val_loader) = get_biased_brats(name, over_sample, data_root, fold, batch_size, presaved = presaved, val_only=False,
                     pattern_dict = pattern_dict, gt_align_prob=gt_align_prob, slice_wise=False, combine_with_brain=True)
    # for data in val_loader:
    #     print(data['image'].shape, data['seg'].shape, data['bb'].shape, data['combined'].shape, data['pattern'].shape)
    #     mri_lst, seg, bb, combined, pattern = data['image'], data['seg'], data['bb'], data['combined'], data['pattern']
    #     break
    for data in train_loader:
        print(data['image'].shape, data['seg'].shape, data['bb'].shape, data['combined'].shape, data['pattern'].shape)
