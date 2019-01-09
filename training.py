"""=========================================================================="""
"""================== Base Libraries ========================================"""
"""=========================================================================="""
import numpy as np
import matplotlib.pyplot as plt

import os,time, datetime,json
import cPickle as pkl

import helper_functions as hf
import prediction_network as prednet
# import utility_functions as ut

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from scipy import stats as ss

import argparse



if not os.path.exists(os.getcwd()+"/Results"):
    os.makedirs(os.getcwd()+"/Results")

if int(torch.__version__.split(".")[1])>2:
    torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",       type=int,   default=100, help="Number of training epochs.")
parser.add_argument("--save_folder",    type=str,   default=os.getcwd()+"/Results", help="Where to save the results.")
parser.add_argument("--image_folders",  nargs="+",  type=str, default=["Road1_bf", "Road1_fb"], help="Which data folders to include. Available: Road1_bf, Road1_fb,Road2_bf, Road2_fb")
parser.add_argument("--test_image_folders",  nargs="+",  type=str, default=["Road2_bf", "Road2_fb"], help="Which data folders to use for testing. Available: Road1_bf, Road1_fb,Road2_bf, Road2_fb")
parser.add_argument("--get_test_perf",  action="store_true", help="Whether to use the test data to test overfitting behaviour.")
parser.add_argument("--lr",             type=float, default=3e-4, help="Learning rate.")
parser.add_argument("--cuda",           action="store_true", help="Use cuda.")
parser.add_argument("--bs",             type=int,   default=32, help="Batchsize.")
parser.add_argument("--gamma",          type=float, default=0.5, help="Gamma for scheduling.")
parser.add_argument("--tau",            type=int,   default=25, help="Tau for scheduling.")
parser.add_argument("--n_workers",      type=int,   default=7, help="Number of cores used for training.")
parser.add_argument("--tvsplit",        type=float, default=0.8, help="Training/validation split.")
parser.add_argument("--all",            action="store_true", help="Train with images from all three cameras.")
parser.add_argument("--augment",        nargs="+",  type=str, default=[],
                                        help="Augmentation procedures to use. Available: flip, jitter, angle_jitter and normalize.")
parser.add_argument("--force_shape",    nargs="+",  default=["None","None"], help="Force an image shape: [width, height]. Values can be None if only one axis should be cropped")
parser.add_argument("--no_center_crop", action="store_true", help="Crop from bottom to top instead around center.")
parser.add_argument("--channels",       type=int,   default=3, help="Use image concatenation via channel axis if channels>3. Must be multiple of 3.")
parser.add_argument("--n_segs",         type=int,   default=20, help="Number of segments to divide the data into.")
parser.add_argument("--use_random",     action="store_true", help="Randomly divide the dataset into single chunks.")
parser.add_argument("--angle_weight",   type=float, default=0)
#Network parameters:
parser.add_argument("--use_pool",       action="store_true",    help="Use average pooling")
parser.add_argument("--init",           action="store_true",    help="Initialize with Kaiming He normal.")
parser.add_argument("--make_full_conv", action="store_true",    help="Make the network fully convolutional.")
parser.add_argument("--pools",          nargs="+", type=int,    default=[0,0,0,0,0],        help="Kernel size for convolutions")
parser.add_argument("--kernel_sizes",   nargs="+", type=int,    default=[5,5,3,3,3],        help="Kernel size for convolutions")
parser.add_argument("--n_filters",      nargs="+", type=int,    default=[24,36,48,64,64],   help="Kernel size for convolutions")
parser.add_argument("--fc_layers",      nargs="+", type=int,    default=[500, 100, 50, 10], help="Kernel size for convolutions")
parser.add_argument("--conv_dropout",   nargs="+", type=float,  default=[0,0,0,0,0],          help="Kernel size for convolutions")
parser.add_argument("--fc_dropout",     nargs="+", type=float,  default=[0,0,0,0],          help="Kernel size for convolutions")
parser.add_argument("--strides",        nargs="+", type=int,    default=[2,2,2,2,1],        help="Kernel size for convolutions")
parser.add_argument("--use_BN",         action="store_true",    help="Use image concatenation via channel axis if channels>3. Must be multiple of 3.")
parser.add_argument("--conv_activation",default="relu",         help="Use image concatenation via channel axis if channels>3. Must be multiple of 3.")
parser.add_argument("--fc_activation",  default="relu",         help="Use image concatenation via channel axis if channels>3. Must be multiple of 3.")
parser.add_argument("--cat_name",       type=str, default="", help="Encode the save name not by time but by Netname_cat_name.")

opt = parser.parse_args(["--cuda"])
opt.loadpath = os.getcwd()+"/Data/"
opt.force_shape = [None if x=="None" else int(x) for x in opt.force_shape]
opt.conv_activation_f = hf.activation_conv(opt.conv_activation)
opt.fc_activation_f   = hf.activation_conv(opt.fc_activation)


"""=========================================================================="""
"""================== Initialize DataLoader ================================="""
"""=========================================================================="""
imagepath   = [opt.loadpath+x for x in opt.image_folders]

print "Folders that are used for training: \n"+"\n".join("[{}] ".format(i)+imageminipath for i,imageminipath in enumerate(imagepath))
print "-"*30

train_img_set     = hf.Images(imagepath, transform=opt.augment, force_shape=opt.force_shape, center_only=not opt.all, ch_dim=opt.channels//3, center_crop = not opt.no_center_crop)
val_img_set       = hf.Images(imagepath, transform=[], force_shape=opt.force_shape, center_only=True, ch_dim=opt.channels//3, center_crop = not opt.no_center_crop) #no augmentation for validation data


### Do SLAM - not viable; no true position input source!
### Measure correlation between val loss amd test loss
### Test weighted MSE
### Finished multichannel - untested
### Try inception style


weights             = hf.steering_hist_and_weights(train_img_set)
# ut.make_angle_dist_plot(img_set)
###Random Assignment
if opt.use_random:
    train_subset        = list(np.random.choice(range(len(train_img_set)),size=int(len(train_img_set)*opt.tvsplit), replace=False))
    val_subset          = list(set(range(len(train_img_set)))-set(train_subset))
else:
    splits = np.ceil(len(train_img_set)/opt.n_segs).astype(int)
    t_idx_split = int(opt.n_segs*opt.tvsplit)
    np.random.seed(42)
    seg_idxs    = np.random.choice(range(opt.n_segs),opt.n_segs, replace=False)
    im_idxs     = range(len(train_img_set))
    train_subset= list(np.hstack([im_idxs[seg_idxs[i]*splits:(seg_idxs[i]+1)*splits] for i in xrange(t_idx_split)]))
    val_subset  = list(set(im_idxs)-set(train_subset))
np.random.seed(1337)
np.random.shuffle(train_subset)
np.random.seed(1)
np.random.shuffle(val_subset)

train_sampler       = hf.SubsetWeightedRandomSampler(train_subset, len(train_subset), weights)
val_sampler         = hf.SubsetWeightedRandomSampler(val_subset, len(val_subset), no_random=True)

train_img_loader    = DataLoader(train_img_set, num_workers=opt.n_workers, batch_size=opt.bs, sampler=train_sampler)
val_img_loader      = DataLoader(val_img_set, num_workers=opt.n_workers, batch_size=opt.bs, sampler=val_sampler)

if opt.get_test_perf:
    test_img_set        = hf.Images([opt.loadpath+x for x in opt.test_image_folders], transform=[], force_shape=opt.force_shape, center_only=True, ch_dim=opt.channels//3, center_crop = not opt.no_center_crop)
    test_img_loader     = DataLoader(test_img_set, num_workers=opt.n_workers, batch_size=opt.bs)


"""=========================================================================="""
"""================== Network Setup ========================================="""
"""=========================================================================="""

print "Loading network... ",
opt.image_shape = [opt.channels]+[320 if opt.force_shape[0] is None else opt.force_shape[0], 160 if opt.force_shape[1] is None else opt.force_shape[1]][::-1]
angle_net       = prednet.nvidia_net(image_size=opt.image_shape, use_BN=opt.use_BN, conv_activation=opt.conv_activation_f, fc_activation=opt.fc_activation_f,\
                                     kernel_sizes=opt.kernel_sizes, n_filters=opt.n_filters, fc_layers=opt.fc_layers, conv_dropout=opt.conv_dropout, fc_dropout=opt.fc_dropout,\
                                     strides=opt.strides, make_full_conv=opt.make_full_conv, use_pooling = opt.use_pool, pools=opt.pools)
if opt.init:
    angle_net.weight_init()

num_params      = prednet.gimme_params(angle_net,sum_only=True)
opt.num_params  = num_params
print "Completed with {} params.".format(num_params)
print "Saving parameters... ",

if opt.cuda:
    _ = angle_net.cuda()

crit      = hf.Loss_Provider("wMSE", opt.angle_weight) if opt.angle_weight!=0 else hf.Loss_Provider("MSE")
v_crit    = hf.Loss_Provider("MSE")
optimizer = torch.optim.Adam(angle_net.parameters(), lr=opt.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.tau, gamma=opt.gamma)

#Save Hyperparameters
opt.name           = angle_net.name
if opt.cat_name!="":
    opt.name=opt.name.split("_")[0]+"_net_"+opt.cat_name
save_folder = opt.save_folder+"/"+opt.name

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

### Save rudimentary info parameters to text-file
with open(save_folder+'/Parameter_Info.txt','w') as f:
    kt = vars(opt)
    del kt["conv_activation_f"], kt["fc_activation_f"]
    json.dump(kt, f)


opt.reconstruction = angle_net.reconstruction


pkl.dump(opt,open(save_folder+"/hypa.pkl","wb"))

print "Finished."
print "-"*30



"""=========================================================================="""
"""================== Start Training========================================="""
"""=========================================================================="""

best_val_loss = np.inf
best_val_emd  = np.inf

save_name     = "placeholder_name"
start         = time.time()

if opt.get_test_perf:
    performance_logger = hf.CSVlogger(save_folder+"/log.csv", ["Epoch", "Time", "LR", "Training Loss", "Validation Loss", "V_EMD", "Test Loss", "T_EMD"])
else:
    performance_logger = hf.CSVlogger(save_folder+"/log.csv", ["Epoch", "Time", "LR", "Training Loss", "Validation Loss"])

print "Running training..."

for epoch in xrange(opt.n_epochs):
    """ Run training """
    train_loss = 0.0
    scheduler.step()
    _ = angle_net.train()
    for i, batch in enumerate(train_img_loader):
        inputs, target = Variable(batch["imgs"][0]), Variable(batch["angle"].float())

        if opt.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        out = angle_net(inputs)

        loss = crit(out, target)
        train_loss += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_img_loader)

    """ Run validation to choose best set of weights """
    angle_net.eval()
    val_loss = 0.0
    orgv, tarv = [], []

    for i, batch in enumerate(val_img_loader):
        inputs, target = Variable(batch["imgs"][0]), Variable(batch["angle"].float())

        if opt.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        out = angle_net(inputs)

        loss = v_crit(out, target)

        val_loss += loss.data[0]

        orgv.extend(out.data.cpu().numpy().reshape(-1))
        tarv.extend(target.data.cpu().numpy().reshape(-1))


    h1,_ = np.histogram(np.array(orgv),range=(-1,1),bins=50)
    h2,_ = np.histogram(np.array(tarv),range=(-1,1),bins=50)

    v_emd    = ss.wasserstein_distance(h1,h2)

    val_loss /= len(val_img_loader)

    """ (Optional) Run testing """
    if opt.get_test_perf:
        angle_net.eval()
        test_loss = 0.0
        orgv, tarv = [], []
        for i, batch in enumerate(test_img_loader):
            inputs, target = Variable(batch["imgs"][0]), Variable(batch["angle"].float())

            if opt.cuda:
                inputs, target = inputs.cuda(), target.cuda()

            out     = angle_net(inputs)
            loss    = v_crit(out, target)
            test_loss += loss.data[0]

            orgv.extend(out.data.cpu().numpy().reshape(-1))
            tarv.extend(target.data.cpu().numpy().reshape(-1))


        h1,_ = np.histogram(np.array(orgv),range=(-1,1),bins=50)
        h2,_ = np.histogram(np.array(tarv),range=(-1,1),bins=50)

        t_emd    = ss.wasserstein_distance(h1,h2)

        test_loss /= len(test_img_loader)
        print '[epoch %3d/%3d] train loss: %.6f | val loss: %.6f | Val EMD: %8.2f | test loss: %.6f | Test EMD: %8.2f' % (epoch+1, opt.n_epochs,train_loss, val_loss, v_emd, test_loss, t_emd)
        performance_logger.write([epoch, np.round(time.time()-start,2), scheduler.get_lr()[0], train_loss, val_loss, v_emd, test_loss, t_emd])
    else:
        print '[epoch %3d/%3d] train loss: %.6f | val loss: %.6f' % (epoch+1, opt.n_epochs,train_loss, val_loss)
        performance_logger.write([epoch, np.round(time.time()-start,2), scheduler.get_lr()[0], train_loss, val_loss])


    if val_loss < best_val_loss:
        save_name = save_folder+"/"+opt.name
        if os.path.isfile(save_name):
            os.remove(save_name)
        torch.save(angle_net.state_dict(), save_name)
        best_val_loss = val_loss

    if v_emd < best_val_emd:
        save_name_emd = save_folder+"/"+opt.name+"_emd"
        if os.path.isfile(save_name_emd):
            os.remove(save_name_emd)
        torch.save(angle_net.state_dict(), save_name_emd)
        best_val_emd = v_emd
