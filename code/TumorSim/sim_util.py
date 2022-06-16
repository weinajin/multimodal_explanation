import numpy as np
import math
import os
import random
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import pathlib

def gradient_loss_abs(gt, pd, ch=4):
    filter1x = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    filter1y = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., -0., -1.]])
    # f1_x = filter1x.view((1,1,3,3))
    # f1_y = filter1y.view((1,1,3,3))
    f1_x = torch.zeros([1, ch, 3, 3])
    f1_y = torch.zeros([1, ch, 3, 3])
    for i in range(ch):
        f1_x[0, i, :, :] = filter1x
        f1_y[0, i, :, :] = filter1y
    Tensor = torch.cuda.FloatTensor
    # filter2x = torch.tensor([[2., 2., 4., 2., 2.],[1., 1., 2., 1., 1.]\
    #     , [0., 0., 0., 0., 0.], [-1.,-1., -2., -1., -1.], [-2., -2., -4., -2., -2.]])
    # filter2y = torch.tensor([[2., 1., 0., -1., -2.], [2., 1., 0., -1., -2.] \
    #                             , [4., 2., 0., -2., -4.], [2., 1., 0., -1., -2.], [2., 1., 0., -1., -2.]])

    # f2_x = filter1x.view((1, 1, 5, 5))
    # f2_y = filter1y.view((1, 1, 5, 5))
    f1_x = f1_x.type(Tensor)
    f1_y = f1_y.type(Tensor)
    gt_x_sobel = F.conv2d(gt, f1_x, padding=1)
    gt_y_sobel = F.conv2d(gt, f1_y, padding=1)
    pd_x_sobel = F.conv2d(pd, f1_x, padding=1)
    pd_y_sobel = F.conv2d(pd, f1_y, padding=1)
    loss = torch.mean(torch.abs(gt_x_sobel - pd_x_sobel) + torch.abs(gt_y_sobel - pd_y_sobel))
    return loss

def one2three(x):
    temp = torch.cat([x, x, x], dim=1)
    return temp
def monitor_loss(i, dataloader_len, step=0, tumor_shape_loss=0, tumor_grade_loss=0, inp_gb_loss=0, inp_lc_loss=0, \
                 inp_grad_loss=0, inp_adv_loss=0, content_loss=0, inp_dis_loss=0, mode = 'Train'):
    if step == 0:
        print( "\t[%d/%d]\t Shape loss: %.4f\t Grade loss: %.4f " % (
            i, dataloader_len, tumor_shape_loss, tumor_grade_loss))
    elif step == 1:
        print("\t[%d/%d]\t Inpaint loss G [gb, lc, grad, adv, content]: [%.4f, %.4f, %.4f, %.4f, %.4f] \t Inpaint loss D: %.4f " % (
        i, dataloader_len, inp_gb_loss, inp_lc_loss, inp_grad_loss,inp_adv_loss,content_loss, inp_dis_loss))
    else:
        print("\t[%d/%d]\t Shape loss: %.4f \t Grade loss: %.4f \t Inpaint loss G [gb, lc, grad, adv, content]: [%.4f, %.4f, %.4f, %.4f, %.4f] \t Inpaint loss D: %.4f " % (
            i, dataloader_len, tumor_shape_loss, tumor_grade_loss, inp_gb_loss, inp_lc_loss, inp_grad_loss, inp_adv_loss, content_loss,inp_dis_loss))

    return 0

def save_result(result_dir, epoch, i, step, M=0, Binary_M=0, Circle_M=0, level_Circle_M=0, out_mask_shape=0, out_mask_grade=0, brain=0, uni_B=0, out_brain=0, mode='Train'):

    if mode == 'Train':
        if step == 0 or step == 2:
            result_img = torch.cat((uni_B, Circle_M, out_mask_shape, Binary_M, Binary_M, level_Circle_M, out_mask_grade, M), 0)
            vutils.save_image(result_img, filename=result_dir + "/[%03d]A_%04d.png" % (epoch, i), nrow=4)

        if step == 1 or step == 2:
            F, T1, T1c, T2 = torch.split(brain, split_size_or_sections=1, dim=1)
            out_F, out_T1, out_T1c, out_T2 = torch.split(out_brain, split_size_or_sections=1, dim=1)
            error_F, error_T1, error_T1c, error_T2 = torch.split(torch.abs(brain-out_brain), split_size_or_sections=1, dim=1)
            result_img = torch.cat((F, T1, T1c, T2, out_F, out_T1, out_T1c, out_T2, \
                                    error_F, error_T1, error_T1c, error_T2), 0)
            vutils.save_image(result_img, filename=result_dir + "/[%03d]B_%04d.png" % (epoch, i), nrow=4)

    if mode == 'Test':
        if step == 0 or step == 2:
            F, T1, T1c, T2 = torch.split(brain, split_size_or_sections=1, dim=1)
            out_F, out_T1, out_T1c, out_T2 = torch.split(out_brain, split_size_or_sections=1, dim=1)
            error_F, error_T1, error_T1c, error_T2 = torch.split(torch.abs(brain - out_brain), split_size_or_sections=1, dim=1)
            result_img = torch.cat((uni_B, Circle_M, coarse_mask, refine_mask, F, T1, T1c, T2, out_F, out_T1, out_T1c, out_T2, \
                                    error_F, error_T1, error_T1c, error_T2), 0)
            vutils.save_image(result_img, filename=result_dir + "/[%03d]B_%04d.png" % (epoch, i), nrow=4)

    if step == 1 or step == 2:
        F, T1, T1c, T2 = torch.split(brain, split_size_or_sections=1, dim=1)
        out_F, out_T1, out_T1c, out_T2 = torch.split(out_brain, split_size_or_sections=1, dim=1)
        error_F, error_T1, error_T1c, error_T2 = torch.split(torch.abs(brain - out_brain), split_size_or_sections=1,
                                                             dim=1)
        result_img = torch.cat((F, T1, T1c, T2, out_F, out_T1, out_T1c, out_T2, \
                                error_F, error_T1, error_T1c, error_T2), 0)
        vutils.save_image(result_img, filename=result_dir + "/[%03d]B_%04d.png" % (epoch, i), nrow=4)
def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def unify(mask):
    temp = np.zeros(mask.shape)
    temp[np.where(mask>0.3)] = 1
    return torch.from_numpy(temp).float()

def quantize(mask):
    temp = np.zeros(mask.shape)
    temp[np.where(mask > 0.85)]= 1 # ET
    temp[np.where(mask > 0.63) and np.where(mask < 0.85)] = 0.75 # ED
    temp[np.where(mask > 0.36) and np.where(mask < 0.63)] = 0.5 # NET
    temp[np.where(mask < 0.36)] = 0
    return  torch.from_numpy(temp).float()

def random_circle(mask):
    [c, h, w]= mask.shape
    #find radius
    l = random.randint(100,2000)
    radius = int((l/3.14)**(.5))
    #find center
    xx, yy = np.mgrid[:h, :w]

    x_center= random.randint(40,210)
    y_center= random.randint(40,210)

    circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
    circle = circle < radius ** 2
    circle = np.array(np.reshape(circle, [c, h, w]), dtype=np.uint8)
    return torch.from_numpy(circle).float()

def make_circle(mask, radius, x_center, y_center):
    [b, c, h, w]= mask.shape

    xx, yy = np.mgrid[:h, :w]
    circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
    circle = circle < radius ** 2
    circle = np.array(np.reshape(circle, [b, c, h, w]), dtype=np.uint8)

    return torch.from_numpy(circle).float()

def mask2circle(mask, level_shake = False):
    [b, c, h, w]= mask.shape
    # find radius
    l1 = np.array(np.where(mask > 0.1)).shape[1]
    l2 = np.array(np.where(mask > 0.7)).shape[1]
    l3 = np.array(np.where(mask > 0.8)).shape[1]

    if level_shake == True:
        if l1 < 4:
            l1, l2, l3 = 0, 0, 0
        else:
            ran_num = random.sample(range(0,l1), 2)
            ran_num.sort()
            [l3, l2] = ran_num

    radius_1 = int((l1 / 3.14) ** (.5))
    radius_2 = int((l2 / 3.14) ** (.5))
    radius_3 = int((l3 / 3.14) ** (.5))

    xx, yy = np.mgrid[:h, :w]
    nonzero_idx = np.where(mask > 0)
    x_center=np.sum(nonzero_idx[2])/(l1+0.001)
    y_center=np.sum(nonzero_idx[3])/(l1+0.001)

    circle = (xx - x_center) ** 2 + (yy - y_center) ** 2

    temp1 = circle < radius_1**2
    temp1 = temp1*0.5
    temp2 = circle<radius_2 ** 2
    temp2 = temp2*0.25
    temp3 = circle<radius_3 ** 2
    temp3 = temp3*0.25

    temp = temp1 + temp2 + temp3

    result = np.array(np.reshape(temp, [b, c, h, w]))
    return torch.from_numpy(result).float()

def make_level_circle(x_center, y_center, radius_1, radius_2, radius_3, img_size = 256 ):
    [b, c, h, w]= [1, 1, img_size, img_size]
    xx, yy = np.mgrid[:h, :w]
    circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
    temp1 = circle < radius_1 ** 2
    temp1 = temp1*0.25
    temp2 = circle<radius_2 ** 2
    temp2 = temp2*0.25
    temp3 = circle<radius_3 ** 2
    temp3 = temp3*0.5

    temp = temp1 + temp2 + temp3

    result = np.array(np.reshape(temp, [b, c, h, w]))

    return torch.from_numpy(result).float()


def random_circle_generate():
    ran_radius = random.sample(range(5, 35), 3)
    ran_radius.sort()
    [radius3, radius2, radius1] = ran_radius
    ran_center = random.sample(range(80, 170), 2)
    [x_center, y_center] = ran_center

    level_circle = make_level_circle(x_center, y_center, radius3, radius2, radius1)

    return level_circle

def make_aug_dir(Augment_dir):
    pathlib.Path(Augment_dir + 'M').mkdir(parents=True, exist_ok=True)
    pathlib.Path(Augment_dir + 'F').mkdir(parents=True, exist_ok=True)
    pathlib.Path(Augment_dir + 'T1').mkdir(parents=True, exist_ok=True)
    pathlib.Path(Augment_dir + 'T1c').mkdir(parents=True, exist_ok=True)
    pathlib.Path(Augment_dir + 'T2').mkdir(parents=True, exist_ok=True)
    return True

def interpolate_level_circle(mask1, mask2, tot_num=10, idx=0):
    [b, c, h, w]= mask1.shape
    m1_l1 = np.array(np.where(mask1 > 0.24)).shape[1]
    m1_l2 = np.array(np.where(mask1 > 0.49)).shape[1]
    m1_l3 = np.array(np.where(mask1 > 0.74)).shape[1]
    m1_l4 = np.array(np.where(mask1 > 0.99)).shape[1]

    m2_l1 = np.array(np.where(mask2 > 0.24)).shape[1]
    m2_l2 = np.array(np.where(mask2 > 0.49)).shape[1]
    m2_l3 = np.array(np.where(mask2 > 0.74)).shape[1]
    m2_l4 = np.array(np.where(mask2 > 0.99)).shape[1]

    l1 = ((tot_num - idx) * m1_l1 + idx * m2_l1) / tot_num
    l2 = ((tot_num - idx) * m1_l2 + idx * m2_l2) / tot_num
    l3 = ((tot_num - idx) * m1_l3 + idx * m2_l3) / tot_num
    l4 = ((tot_num - idx) * m1_l4 + idx * m2_l4) / tot_num

    radius_1 = int((l1 / 3.14) ** (.5))
    radius_2 = int((l2 / 3.14) ** (.5))
    radius_3 = int((l3 / 3.14) ** (.5))
    radius_4 = int((l4 / 3.14) ** (.5))

    xx, yy = np.mgrid[:h, :w]
    nonzero_idx = np.where(mask1 > 0)
    x_center=np.sum(nonzero_idx[2])/(m1_l1+0.001)
    y_center=np.sum(nonzero_idx[3])/(m1_l1+0.001)
    circle = (xx - x_center) ** 2 + (yy - y_center) ** 2
    temp1 = circle < radius_1**2
    temp1 = temp1*0.25
    temp2 = circle<radius_2 ** 2
    temp2 = temp2*0.25
    temp3 = circle<radius_3 ** 2
    temp3 = temp3*0.25
    temp4 = circle<radius_4 ** 2
    temp4 = temp4*0.25
    temp = temp1 + temp2 + temp3 + temp4

    result = np.array(np.reshape(temp, [b, c, h, w]))
    return torch.from_numpy(result).float()




import glob
from PIL import Image
def make_gif(file_dir, save_dir):
    print(file_dir, save_dir)
    frames=[]
    imgs = glob.glob(file_dir +"/*.png")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(save_dir, format = 'GIF', append_images=frames[1:], save_all=True, duration=300, loop=0)

