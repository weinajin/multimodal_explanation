import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data

import pdb

from torch.autograd import Variable

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class Dataset(data.Dataset):
    def __init__(self, datadir, trainval_label, tear_type, use_gpu, input_modality = [1,1,1]):
        super().__init__()
        self.use_gpu = use_gpu

        label_dict = {}
        # self.paths = []
        abnormal_label_dict = {}
        
        if datadir[-1]=="/":
            datadir = datadir[:-1]
        self.datadir = datadir
        if trainval_label == 'train' or trainval_label == 'val':
            self.trainval_label = 'trainval'
        elif trainval_label == 'test':
            self.trainval_label = 'test'

        for i, line in enumerate(open(datadir+'/'+trainval_label+ '-'+tear_type+'.csv').readlines()):
            line = line.strip().split(',')
            filename = line[0]
            label = line[1]
            label_dict[filename] = int(label)

        # for i, line in enumerate(open(datadir+'/'+trainval_label+'-'+"abnormal"+'.csv').readlines()):
        #     line = line.strip().split(',')
        #     filename = line[0]
        #     label = line[1]
        #     abnormal_label_dict[filename] = int(label)

        # for filename in os.listdir(os.path.join(datadir, "axial")):
        #     if filename.endswith(".npy"):
        #         self.paths.append(filename)
        
        # self.labels = [label_dict[path.split(".")[0]] for path in self.paths]
        # self.abnormal_labels = [abnormal_label_dict[path.split(".")[0]] for path in self.paths]
        self.fns = [k for k in label_dict]
        self.labels = [label_dict[fn] for fn in self.fns]
        # self.abnormal_labels = [abnormal_label_dict[fn] for fn in label_dict.keys()]

        # if tear_type != "abnormal":
        #     temp_labels = [self.labels[i] for i in range(len(self.labels)) if self.abnormal_labels[i]==1]
        #     neg_weight = np.mean(temp_labels)
        # else:
        neg_weight = np.mean(self.labels)
        
        self.weights = [neg_weight, 1 - neg_weight]

        # for MI experiment
        self.input_modality = input_modality

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        data_id = self.fns[index].zfill(4)
        filename = "{}.npy".format(data_id)
        vol_axial = np.load(os.path.join(self.datadir, self.trainval_label, "axial", filename))
        vol_sagit = np.load(os.path.join(self.datadir, self.trainval_label, "sagittal", filename))
        vol_coron = np.load(os.path.join(self.datadir, self.trainval_label, "coronal", filename))

        # axial
        pad = int((vol_axial.shape[2] - INPUT_DIM)/2)
        vol_axial = vol_axial[:,pad:-pad,pad:-pad]
        vol_axial = (vol_axial-np.min(vol_axial))/(np.max(vol_axial)-np.min(vol_axial))*MAX_PIXEL_VAL
        vol_axial = (vol_axial - MEAN) / STDDEV
        vol_axial = np.stack((vol_axial,)*3, axis=1)
        vol_axial_tensor = torch.FloatTensor(vol_axial)
        
        # sagittal
        pad = int((vol_sagit.shape[2] - INPUT_DIM)/2)
        vol_sagit = vol_sagit[:,pad:-pad,pad:-pad]
        vol_sagit = (vol_sagit-np.min(vol_sagit))/(np.max(vol_sagit)-np.min(vol_sagit))*MAX_PIXEL_VAL
        vol_sagit = (vol_sagit - MEAN) / STDDEV
        vol_sagit = np.stack((vol_sagit,)*3, axis=1)
        vol_sagit_tensor = torch.FloatTensor(vol_sagit)

        # coronal
        pad = int((vol_coron.shape[2] - INPUT_DIM)/2)
        vol_coron = vol_coron[:,pad:-pad,pad:-pad]
        vol_coron = (vol_coron-np.min(vol_coron))/(np.max(vol_coron)-np.min(vol_coron))*MAX_PIXEL_VAL
        vol_coron = (vol_coron - MEAN) / STDDEV
        vol_coron = np.stack((vol_coron,)*3, axis=1)
        vol_coron_tensor = torch.FloatTensor(vol_coron)

        label_tensor = torch.FloatTensor([self.labels[index]])
        if np.all(self.input_modality): # do not ablate modality
            return vol_axial_tensor, vol_sagit_tensor, vol_coron_tensor, label_tensor, data_id
        else:
            for i, on_off in enumerate(self.input_modality):
                if i == 0 and on_off == 0:
                    vol_axial_tensor = torch.zeros(vol_axial_tensor.shape)
                    print('ablate axial')
                if i == 1 and on_off == 0:
                    vol_sagit_tensor = torch.zeros(vol_sagit_tensor.shape)
                    print('ablate sagit')
                if i == 2 and on_off == 0:
                    vol_coron_tensor = torch.zeros(vol_coron_tensor.shape)
                    print('ablate coron')
            return vol_axial_tensor, vol_sagit_tensor, vol_coron_tensor, label_tensor, data_id



    def __len__(self):
        return len(self.labels)

def load_data(task="acl", use_gpu=False, test_loader = False, input_modality = [1,1,1], machine = 'solar'):
    if machine == 'solar':
        data_dir = '/project/labname-lab/authorid/dld_data/MRNet-v1.0' # solar
    else:
        data_dir = '/local-scratch/authorid/dld_data/MRNet-v1.0' # ts12

    # data_dir = '/scratch/authorid/dld_data/MRNet-v1.0' # cedar
    if test_loader:
        test_dataset = Dataset(data_dir, 'test', task, use_gpu, input_modality = input_modality)

        test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

        return test_loader

    train_dataset = Dataset(data_dir, 'train', task, use_gpu, input_modality = input_modality)
    valid_dataset = Dataset(data_dir, 'val', task, use_gpu, input_modality = input_modality)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False)

    return train_loader, valid_loader
