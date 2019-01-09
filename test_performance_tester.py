""" Libraries """
import argparse
import cPickle as pkl
import os, time, sys


import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import prediction_network as prednet
import helper_functions as hf
from scipy import stats as ss


import matplotlib.pyplot as plt


""" Input Argument Parser """

"""
What to run ideally:
python test_performance_tester.py --cuda --show_hists --save
"""
parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument("--base_folder",    default=os.getcwd()+"/Results/", help="Network name")
parser.add_argument("--base_folder_test", default=os.getcwd()+"/Data/", help="Test folder name")
parser.add_argument("--force_shape",    nargs="+",  default=["None","None"], help="Force an image shape: [width, height]. Values can be None if only one axis should be cropped")
parser.add_argument("--test_all",       action="store_true", help="Test images in all directions.")
parser.add_argument("--cuda",           action="store_true", help="Use cuda.")
parser.add_argument("--show_hists",     action="store_true", help="Show histograms.")
parser.add_argument("--save",           action="store_true", help="save histograms.")
parser.add_argument("--savename",       type=str, default="angle_hist_emd", help="save histograms name.")
parser.add_argument("--was_not_cuda",   action="store_true", help="Training was not performed on cuda.")

opt = parser.parse_args()



"""=========================================================================="""
"""================== Start Testing ========================================="""
"""=========================================================================="""

foldernums, folder_files = hf.file_lister(opt.base_folder_test, "Choose data folders to test on:")
opt.base_folder_test  = [opt.base_folder_test + "/"+folder_files[foldernum] for foldernum in foldernums]
opt.names_test        = [folder_files[foldernum] for foldernum in foldernums]

print "-------------"

foldernums, folder_files = hf.file_lister(opt.base_folder, "Choose networks to test:")
opt.base_folder     = [opt.base_folder + "/"+folder_files[foldernum] for foldernum in foldernums]
opt.name            = [folder_files[foldernum] for foldernum in foldernums]

info_per_image    = {key:{"loss":[], "out":[], "gt":[], "t":0} for key in opt.name}


crit      = hf.Loss_Provider("MSE") #also available: wMSE
print "-------------"


for j,(base_folder, net_name) in enumerate(zip(opt.base_folder, opt.name)):
    start = time.time()
    print "Computing test results for [{}/{}]\r".format(j+1,len(opt.name)),
    sys.stdout.flush()

    ### Load saved parameter set
    net_opt   = pkl.load(open(base_folder+"/hypa.pkl"))

    ### Reconstruct the network.
    net_type    = net_opt.name.split("_")[0]

    test_img_set        = hf.Images(opt.base_folder_test, transform=[], force_shape=net_opt.force_shape, center_only=True, ch_dim=net_opt.channels//3)
    test_img_loader     = DataLoader(test_img_set, num_workers=net_opt.n_workers, batch_size=net_opt.bs, shuffle=False)



    if net_type=="NVIDIA":
        angle_net = prednet.nvidia_net(**net_opt.reconstruction)
    else:
        raise NotImplementedError("Network type not implemented!")


    load_name = base_folder+"/"+net_name
    if opt.cuda and not opt.was_not_cuda:
        _ = angle_net.cuda()
        angle_net.load_state_dict(torch.load(load_name))
    if not opt.cuda and not opt.was_not_cuda:
        angle_net.load_state_dict(torch.load(load_name, map_location=lambda storage, loc: storage))
    if not opt.cuda and opt.was_not_cuda:
        angle_net.load_state_dict(torch.load(load_name))

    angle_net.eval()


    for i, batch in enumerate(test_img_loader):
        inputs, target = Variable(batch["imgs"][0]), Variable(batch["angle"].float())

        if opt.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        out = angle_net(inputs)
        loss = crit(out, target)

        info_per_image[net_name]["loss"].append(loss.data[0])
        info_per_image[net_name]["out"].extend(out.data.cpu().numpy().reshape(-1))
        info_per_image[net_name]["gt"].extend(target.data.cpu().numpy().reshape(-1))

    info_per_image[net_name]["t"] = time.time()-start


print "Testing done, printing evaluation metrics...\n----------------------\n"


n_y = int((len(opt.name)-4*(len(opt.name)//4))>0)+len(opt.name)//4
n_x = np.clip(len(opt.name),None,4).astype(int)

f,ax = plt.subplots(n_y,n_x)
try:
    len(ax)
    for i,(key,axs) in enumerate(zip(opt.name,ax.reshape(-1))):
        h1,_,_ = axs.hist(info_per_image[key]["out"],range=(-1,1),bins=50, histtype="step", label="Predictions")
        h2,_,_ = axs.hist(info_per_image[key]["gt"],range=(-1,1),bins=50, histtype="step", label="Ground Truth")
        emd    = ss.wasserstein_distance(h1,h2)
        performance          = '\033[95m'+">>> [{}] Evaluation Metrics for: {} ({})s <<< \n".format(i,key, info_per_image[key]["t"])+'\033[0m'+\
                               "Overall loss:\t {}\n".format(np.mean(info_per_image[key]["loss"]))+\
                               "Avg. Out-angle:\t {}\n".format(np.mean(info_per_image[key]["out"]))+\
                               "Avg. In-angle:\t {}\n".format(np.mean(info_per_image[key]["gt"]))+\
                               "EMD angle-hists:\t {}".format(emd)
        axs.set_title("[{}] Ang. dist. with EMD [{}] ".format(foldernums[i],np.round(emd,2)))
        axs.legend()
        print performance
    for i,axs in enumerate(ax.reshape(-1)):
        if i>len(opt.name)-1:
            axs.axis("off")
    f.set_size_inches(n_x*5, n_y*5)
    if opt.save:
        f.savefig(os.getcwd()+"/"+opt.savename+".pdf",pad=0)
    if opt.show_hists:
        plt.show()
    else:
        plt.close()
except:
    for i,key in enumerate(opt.name):
        h1,_,_ = ax.hist(info_per_image[key]["out"],range=(-1,1),bins=50, histtype="step", label="Predictions")
        h2,_,_ = ax.hist(info_per_image[key]["gt"],range=(-1,1),bins=50, histtype="step", label="Ground Truth")
        emd    = ss.wasserstein_distance(h1,h2)
        performance          = "Overall loss:    {0:.5f}\n".format(np.mean(info_per_image[key]["loss"]))+\
                               "EMD angle-hists: {0:.2f}".format(emd)
        ax.set_title("{}".format(key))
        ax.legend()
        ax.set_xlabel(performance)
        print performance

    f.set_size_inches(n_x*5, n_y*7)
    if opt.save:
        f.savefig(os.getcwd()+"/"+opt.savename+".pdf",pad=0)
    if opt.show_hists:
        plt.show()
    else:
        plt.close()
