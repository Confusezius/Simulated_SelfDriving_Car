"""=========================================================================="""
"""================== Base Libraries ========================================"""
"""=========================================================================="""
import numpy as np
import matplotlib.pyplot as plt
import os, copy
import time
import datetime
import cPickle as pkl

#import helper_functions as hf
#import prediction_network as prednet

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import mirror
import pandas as pd
from matplotlib.pyplot import imread
import csv
from PIL import Image, ImageDraw
from skimage import color
import sys


"""===== File-Listing Utility ============"""
def file_lister(base_folder, init_str = "Please Choose:", inp_str = "Please enter the respective number:  "):
    folder_files    = os.listdir(base_folder)
    folder_files.sort()
    output_string   = init_str+"\n"
    valid_nums      = []
    for i,ff in enumerate(folder_files):
        valid_nums.append(i)
        output_string+="["+str(i)+"] "+ff+"\n"
    print output_string
    foldernums   = input("\n"+inp_str)
    if foldernums == "all":
        foldernums = valid_nums
    else:
        foldernums  = [foldernums] if isinstance(foldernums,int) else list(foldernums)
        if any(np.array(foldernums)<0):
            assert all(np.array(foldernums)<0), "Input number must either all be positive or negative!"
            foldernums = [x for x in valid_nums if x not in np.abs(np.array(foldernums))]
        else:
            foldernums  = [int(foldernum) for foldernum in foldernums]

    assert all([foldernum in valid_nums]), "Invalid number entered!"
    return foldernums, folder_files


"""===== Numpy-Torch-Transformations ====="""
def maketorch(x):
    return torch.from_numpy(x.transpose(2,0,1)).float()

def makenumpy(x):
    return x.numpy().transpose(1,2,0).astype(np.uint8)

def activation_conv(op_activation):
    if op_activation=="relu":
        return nn.ReLU()
    if op_activation=="elu":
        return nn.ELU()
    if op_activation=="prelu":
        return nn.PReLU()
    if op_activation=="leaky":
        return nn.LeakyReLU()


"""===== Image Loader with optional Augmentation ====="""
def crop_PIL(img, force_shape, no_center_crop=True):
    if any(force_shape):
        width, height               = img.size
        forced_width, forced_height = [width if force_shape[0] is None else force_shape[0],
        height if force_shape[1] is None else force_shape[1]]

        if no_center_crop:
            left, right = (width - forced_width)/2, (width + forced_width)/2
            bottom, top = height, height-forced_height
        else:
            left, right = (width - forced_width)/2, (width + forced_width)/2
            bottom, top = (height + forced_height)/2, (height - forced_height)/2

        crop_box    = (left, top, right, bottom)
        return img.crop(crop_box)

class Shadowmaker():
    def __init__(self, seed=1):
        self.bound  = 5


    def shadowmaker(self, input_img, seed=None):
        self.rng    = np.random.RandomState(seed) if seed is not None else np.random.RandomState(np.random.randint(10000))
        self.nrr    = self.rng.randint
        self.nrc    = self.rng.choice
        width,height = input_img.size
        sort_b = self.nrc([False,True])
        sort_t = self.nrr(2)
        complexity  = self.nrr(2,20)
        base_idx    = self.nrr(12)
        alpha       = 0.4+0.55*self.rng.uniform()

        if base_idx==0:
            base = {"polygon":[(0,h) for h in self.nrc(range(0,height-self.bound),2,replace=False)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
        elif base_idx==1:
            base = {"polygon":[(w,0) for w in self.nrc(range(0,width-self.bound),2,replace=False)], "type":sort_t, "rev":True, "sort":sort_b, "insert":True}
        elif base_idx==2:
            base = {"polygon":[(width-1,h) for h in self.nrc(range(0,height-self.bound),2, replace=False)], "type":sort_t, "rev":True, "sort":sort_b, "insert":True}
        elif base_idx==3:
            base = {"polygon":[(w,height-1) for w in self.nrc(range(0,width-self.bound),2, replace=False)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
        elif base_idx==4:
            base = {"polygon":[(0,self.nrr(self.bound,height)),(0,0),(self.nrr(self.bound,width),0)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
        elif base_idx==5:
            base = {"polygon":[(self.nrr(width-self.bound),0),(width-1,0),(width-1,self.nrr(self.bound,height))], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
        elif base_idx==6:
            base = {"polygon":[(0,self.nrr(height-self.bound)),(width-1,height-1),(self.nrr(self.bound,width),height-1)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
        elif base_idx==7:
            base = {"polygon":[(self.nrr(self.bound,width),0),(0,height-1),(0,self.nrr(height-self.bound))], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
        elif base_idx==8:
            base = {"polygon":[(0,self.nrr(self.bound,height-self.bound)),(0,0),(width-1,0),(width-1,self.nrr(self.bound,height-self.bound))], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
        elif base_idx==9:
            base = {"polygon":[(width-1,self.nrr(self.bound,height-self.bound)),(width-1,height-1),(0,height-1),(0,self.nrr(self.bound,height-self.bound))], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
        elif base_idx==10:
            base = {"polygon":[(self.nrr(self.bound,width-self.bound),0),(width-1,0),(width-1,height-1),(self.nrr(self.bound,width-self.bound),height-1)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
        elif base_idx==11:
            base = {"polygon":[(self.nrr(self.bound,width-self.bound),height-1),(0,height-1),(0,0),(self.nrr(self.bound,width-self.bound),0)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}


        fill_points     = [list(set([(x,y) for x,y in zip(self.nrr(5,height-5,complexity),self.nrr(5,width-5,complexity))]))]
        if base["sort"]:
            fill_points     = sorted(fill_points[0],key=lambda i:i[base["type"]], reverse=base["rev"])
        for x in fill_points[0]:
            if base["insert"]:
                base["polygon"].insert(1,x)
            else:
                base["polygon"].extend(x)

        # print "{}, {}, {}, {}, {}".format(base_idx, complexity, alpha, width, height)
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(base["polygon"], outline=1, fill=1)
        mask = np.array(img)
        input_im = np.array(input_img)
        input_im = color.rgb2hsv(input_im)
        mask = mask==1
        input_im[:,:,2][mask] = input_im[:,:,2][mask]*alpha
        input_im = color.hsv2rgb(input_im)
        lsm = Image.fromarray((input_im*255).astype(np.uint8))
        return lsm




class Images(Dataset):
    def __init__(self, load_paths, transform = [], force_shape=[None, None], angle_corr=0.25, return_single=True, center_only=False, center_crop=True, ch_dim=1, same_trafo=["flip"]):
        """
        Arguments:
            to_be_added...
        """
        super(Images, self).__init__()
        self.training_files = pd.concat([pd.read_csv(mini_load+"/driving_log.csv", header=None) for mini_load in load_paths])
        self.transform      = transform
        self.force_shape    = force_shape
        self.center_crop    = center_crop
        self.angle_corr     = angle_corr
        self.return_single  = return_single
        self.center_only    = center_only
        self.ch_dim         = ch_dim
        self.same_trafo     = same_trafo
        self.Shadowmaker    = Shadowmaker(1)


    def __getitem__(self, idx):
        base = np.clip(idx,self.ch_dim,None).astype(int)
        # img_info = np.array(self.training_files[base:base+1+self.ch_dim])[0]
        img_info = [list(x) for x in np.array(self.training_files[base-self.ch_dim:base])]
        #Note: class Images assumes that the ordering in the data-csv is center, left, right.
        data = {"imgs":[]}


        if not self.center_only:
            r_idx    = np.random.randint(3)
            for ch in xrange(self.ch_dim):
                data["imgs"].append(Image.open(img_info[ch][r_idx]))
            ang = np.repeat(float(img_info[-1][3]),3)
            if self.angle_corr and not self.center_only:
                ang[1] = ang[1]+self.angle_corr
                ang[2] = ang[2]-self.angle_corr
            data["angle"]   = ang[r_idx]
            data["accel"]   = float(img_info[-1][4])
            data["break"]   = float(img_info[-1][5])
            data["vel"]     = float(img_info[-1][6])
        else:
            for ch in xrange(self.ch_dim):
                data["imgs"].append(Image.open(img_info[ch][0]))
            data["angle"]   = float(img_info[-1][3])
            data["accel"]   = float(img_info[-1][4])
            data["break"]   = float(img_info[-1][5])
            data["vel"]     = float(img_info[-1][6])


        if any(self.force_shape):
            width, height               = data["imgs"][0].size
            forced_width, forced_height = [width if self.force_shape[0] is None else self.force_shape[0],
                                           height if self.force_shape[1] is None else self.force_shape[1]]
            if self.center_crop:
                left, right = (width - forced_width)/2, (width + forced_width)/2
                bottom, top = (height + forced_height)/2, (height - forced_height)/2
            else:
                left, right = (width - forced_width)/2, (width + forced_width)/2
                bottom, top = height, height-forced_height

            crop_box    = (left, top, right, bottom)
            data["imgs"]= [data["imgs"][i].crop(crop_box) for i in xrange(len(data["imgs"]))]

        if len(self.transform)!=0:
            if "flip" in self.transform:
                #Randomly flip along the vertical axis
                data = self.random_flip(data)
            if "jitter" in self.transform:
                #Randomly perform brightness, contrast, saturation and hue change.
                seed = np.random.randint(1e8)
                for i in xrange(len(data["imgs"])):
                    np.random.seed(seed)
                    data["imgs"][i] = transforms.ColorJitter(0.4,0.2,0.2,0.06)(data['imgs'][i])

            if "angle_jitter" in self.transform:
                #Jitter the angle by a small amount to learn reaction invariance.
                data['angle']= data['angle'] + np.random.normal(0,0.06/(1+(2*np.abs(data['angle'])))) if np.random.randint(2) else data['angle']

            ### Add translation
            if "translate" in self.transform:
                pass
            ### Add shadowing
            if "shadow" in self.transform:
                if np.random.randint(3)==0:
                    data["imgs"] = [self.Shadowmaker.shadowmaker(data["imgs"][i],seed=idx) for i in xrange(len(data["imgs"]))]

        #Add Normalization
        #convert to torch
        output_trafo = transforms.Compose([transforms.ToTensor()])
        data['imgs'] = [output_trafo(data['imgs'][i]) for i in xrange(len(data["imgs"]))]
        if self.ch_dim>1:
            data["imgs"] = [torch.cat(data["imgs"], dim=0)]
        return data


    def random_flip(self, data):
        if np.random.rand()>0.5:
            data['imgs'] = [mirror(data['imgs'][i]) for i in xrange(len(data['imgs']))]
            data['angle'] = -data['angle']
        return data


    def __len__(self):
        return len(self.training_files)



"""===== Loss Navigator Module ====="""
class Loss_Provider(nn.Module):
    def __init__(self, loss_choice, weight=None, **kwargs):
        super(Loss_Provider, self).__init__()
        self.loss_choice    = loss_choice
        if loss_choice=="MSE":
            self.loss_func = nn.MSELoss(**kwargs)
        elif loss_choice=="wMSE":
            self.loss_func  = nn.MSELoss(reduce=False)
            self.weight     = weight
            self.inv        = weight>0
        else:
            raise NotImplementedError("This choice of loss has not been implemented: {}".format(loss_choice))

    def forward(self, inp, tar, importance=1.):
        if self.loss_choice=="wMSE":
            if self.inv:
                return torch.mean(self.loss_func(inp, tar)*(1/(self.weight*tar.view(1,-1).pow(2).pow(0.5)+importance)))
            else:
                return torch.mean(self.loss_func(inp, tar)*(-1.*self.weight*tar.view(1,-1).pow(2).pow(0.5)+importance))
        else:
            return self.loss_func(inp, tar)





"""===== Angle Occurence Histograms and Weights ====="""
def steering_hist_and_weights(Dataset):
    """
    Gives back histogram of steering angles and weights to get unbiased training set.

    Arguments:
        Dataset:    Instance of the torch dataset for which angle occurence and weighting
                    is to be performed on.
    """
    angles = np.array(Dataset.training_files[3])
    n, bins, patches = plt.hist(np.array(angles), 20, facecolor='blue', alpha=0.4)
    plt.close()
    #Give a weighting proportional to the inverse occurence
    weights_hist = (1/n)/sum(1/n)
    #For each image (tuple) in the dataset, compute the occurence "probability" for each slice.
    #Note that this will be normalized later on in the SubsetWeightedRandomSampler.
    weights = [weights_hist[np.digitize(angles[i], bins, right=True)-1] for i in xrange(len(Dataset))]
    return weights



"""===== Frequency-unbiased Training Sampler ====="""
class SubsetWeightedRandomSampler():
    """
    Random sampler for angle-frequency unbiased training.
    Samples elements from [0,..,len(weights)-1] with given probabilities (weights) proportional to their
    inverse occurence frequency - see steering_hist_and_weights().

    Arguments:
        weights (list):     a list of weights, not necessary summing up to one;
                            normally precomputed for each dataset via steering_hist_and_weights()
        num_samples (int):  number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
                            If not, they are drawn without replacement, which means that when a
                            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, indices, num_samples, weights=None, replacement=True, no_random=False):
        self.indices = np.array(indices)
        self.weights = np.array(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.no_random   = no_random

    def __iter__(self):
        if not self.no_random:
            return iter(np.random.choice(self.indices, replace=self.replacement, size=self.num_samples, p=self.weights[self.indices]/sum(self.weights[self.indices])))
        else:
            return iter(self.indices)

    def __len__(self):
        return self.num_samples


"""===== Logging Helper ====="""
def write_log(logname,epoch, times, learning_rate, train_loss, val_loss):
    with open(logname,"a") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        if epoch==0:
            writer.writerow(["Epoch", "Time", "LR", "Training Loss", "Validation Loss"])
        writer.writerow([epoch, round(times,2), learning_rate, train_loss, val_loss])

class CSVlogger():
    def __init__(self, logname, header_names):
        self.header_names = header_names
        self.logname      = logname
        if os.path.exists(logname):
            os.remove(logname)
        with open(logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(header_names)
    def write(self, inputs):
        with open(self.logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(inputs)
