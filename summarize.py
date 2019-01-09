import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import cPickle as pkl
import collections


plt.style.use("ggplot")

"""
If there are no other reasons to do so, try to always use:
python summarize.py --log --inc_test_loss --inc_emd --set_baselines
                    --base_folder /home/karsten_dl/e-kar/Results/Minitest_Comparisons --save_plot here
This will include all useful details and information.
"""
parser = argparse.ArgumentParser()
parser.add_argument("--base_folder",    default=os.getcwd()+"/Results/", help="Network name")
parser.add_argument("--save_plot",      default="", type=str, help="Save plots.")
parser.add_argument("--log",            action="store_true", help="use logarithmic scaling")
parser.add_argument("--inc_test_loss",  action="store_true", help="Plot test loss (if available).")
parser.add_argument("--inc_emd",        action="store_true", help="Plot val and test emd (if available).")
parser.add_argument("--plot_names",     nargs="+",  default=[], help="Names that are going to be used in the plot.")
parser.add_argument("--no_plots",       action="store_true", help="Don't plot")
parser.add_argument("--set_baselines",   action="store_true", help="Choose baselines.")
opt = parser.parse_args()



""" Allow the user to choose the set of weights to initialize the network with """
folder_files    = os.listdir(opt.base_folder)
folder_files.sort()
output_string   = "Choose folder to plot loss and metric curves for: \n"
valid_nums      = []
for i,ff in enumerate(folder_files):
    valid_nums.append(i)
    output_string+="["+str(i)+"] "+ff+"\n"
print output_string


foldernums = input("\nPlease enter the respective number(s):")

if foldernums!="all":
    foldernums = [foldernums] if isinstance(foldernums,int) else list(foldernums)
    foldernums = [int(foldernum) for foldernum in foldernums] if all([x>=0 for x in foldernums]) else [int(valnum) for valnum in valid_nums if -valnum not in foldernums]
else:
    foldernums = valid_nums

assert all([np.abs(foldernum) in valid_nums for foldernum in foldernums]), "Invalid number entered!"

print "Summarizing:\n{} ".format("\n".join(folder_files[foldernum] for foldernum in foldernums))
print "------------\n\n"

opt.base_folder  = [opt.base_folder + "/"+folder_files[foldernum] for foldernum in foldernums]
if len(opt.plot_names)!=0:
    assert len(opt.plot_names)==len(opt.base_folder),"Number of new plotnames must equal the number of selected folders."
    opt.name         = opt.plot_names
else:
    opt.name         = [folder_files[foldernum] for foldernum in foldernums]




""" Allow the user to choose the set baselines weights that are marked specially in the plots"""
if opt.set_baselines:
    output_string   = "Choose weights to set as baseline: \n"
    valid_nums      = []
    for i,ff in enumerate(folder_files):
        valid_nums.append(i)
        output_string+="["+str(i)+"] "+ff+"\n"
    print output_string


    foldernums_base = input("\nPlease enter the respective number(s):")
    foldernums_base = [foldernums_base] if isinstance(foldernums_base,int) else list(foldernums_base)
    foldernums_base = [int(foldernum) for foldernum in foldernums_base]
    assert all([np.abs(foldernum) in valid_nums for foldernum in foldernums_base]), "Invalid number entered!"
    base_names      = [folder_files[foldernum] for foldernum in foldernums_base]






### Colorset to iterate through for plotting
t_colorset = ["darkorange","darkolivegreen","deeppink","darkgoldenrod","darkred","dodgerblue","tomato","turquoise","limegreen","grey","orange","y","violet","peru","salmon","skyblue"]*10
c_colorset = ["dodgerblue","fuchsia","darkolivegreen","darkgoldenrod","darkred","darkorange","tomato","turquoise","limegreen","grey","orange","y","violet","peru","salmon","skyblue"]*10
b_colorset = ["r","b","g","k"]



### Set up Plot structure
if opt.inc_test_loss:
    if opt.inc_emd:
        f,ax = plt.subplots(1,5)
    else:
        f,ax = plt.subplots(1,4)
else:
    f,ax = plt.subplots(1,3)

### performance data
best_data = collections.OrderedDict()
for key in opt.name:
    best_data[key] = {}
base_idx  = []
times     = []
base_count= 0

best_values = [0.0024,0.00121,0.0581, 19.137]

for i,(base_folder,name) in enumerate(zip(opt.base_folder,opt.name)):
    save_file   = pd.read_csv(base_folder+"/log.csv", header=0)
    opt_net     = pkl.load(open(base_folder+"/hypa.pkl","rb"))
    if opt.inc_test_loss:
        if not opt_net.get_test_perf:
            raise ValueError("{} does not contain any testing information!".format(name))

    lr_change_full  = np.where(np.diff(np.array(save_file["LR"]))!=0)[0]+1
    lrs_full        = [save_file["LR"][0]]+[np.array(save_file["LR"])[ikl] for ikl in lr_change_full]
    lrs_full        = ["{0: g}".format(lr) for lr in lrs_full]
    if len(lrs_full)>4:
        lrs_full = lrs_full[:2]+["..."]+lrs_full[-2:]

    sep_n = 20
    recon = opt_net.reconstruction
    recon_string = "\n".join(key+": "+(20-len(key))*" "+"\t{}".format(recon[key]) for key in recon.keys())

    training_information = '\033[95m'+">>> Total Training Time: {}s <<< \n".format(np.array(save_file["Time"])[-1])+'\033[0m'+\
                           '\033[95m'+">>> Total Parameters: {}s <<< \n".format(opt_net.num_params)+'\033[0m'+\
                           "-"*sep_n+'\033[92m'+" Training Setup "+'\033[0m'+"-"*sep_n+"\n"+\
                           "Number of epochs:\t {}\n".format(np.array(save_file["Epoch"])[-1])+\
                           "Used Files: \t\t {}\n".format(" ".join(imf for imf in opt_net.image_folders))+\
                           "Batchsize:\t\t {} (with channel-size {})\n".format(opt_net.bs, opt_net.channels)+\
                           "Forced shape:\t\t {}\n".format(opt_net.force_shape)+\
                           "Validation split:\t {}\n".format(opt_net.tvsplit)+\
                           "Contains Test perf:\t {}\n".format(opt_net.get_test_perf)+\
                           "-"*sep_n+'\033[92m'+" Loss Setup "+'\033[0m'+"-"*sep_n+"\n"+\
                           "Scheduling:\t\t Gamma: {} & Tau: {}\n".format(opt_net.gamma, opt_net.tau)+\
                           "Learning rate(s):\t {}\n".format(lrs_full)+\
                           "-"*sep_n+'\033[92m'+" Network Setup (type: {})".format(opt_net.name.split("_")[0])+'\033[0m'+"-"*sep_n+"\n"+\
                           recon_string

    print '\033[93m'+"[{}]   Relevant Setup, Learning Curves & Training Summary for {}:\n".format(i,name)+'\033[0m'+training_information
    print "\n\n"

    best_data[name]["best train loss"]  = np.min(save_file["Training Loss"])
    best_data[name]["best val loss"]    = np.min(save_file["Validation Loss"])
    best_data[name]["best test loss"]   = np.min(save_file["Test Loss"])
    best_data[name]["best test emd"]    = np.min(save_file["T_EMD"])
    times.append(np.round(np.array(save_file["Time"])[-1]/3600.,2))

    try:
        lwd = 1 if name not in base_names else 2
        if name in base_names:
            base_idx.append(i)
    except:
        lwd = 1

    if opt.set_baselines:
        if name in base_names:
            c_set = b_colorset
            idx = base_count
            base_count += 1
        else:
            c_set = t_colorset
            idx = i
    else:
        c_set = t_colorset
        idx = i

    if opt.log:
        ax[0].semilogy(save_file["Epoch"],save_file["Training Loss"], c_set[idx], linewidth=lwd, label="Tr.Loss {} ({}h)".format(name,np.round(np.array(save_file["Time"])[-1]/3600.,2)))
        ax[1].semilogy(save_file["Epoch"],save_file["Validation Loss"], c_set[idx], linewidth=lwd, label="Val. Loss {} ({}h)".format(name,np.round(np.array(save_file["Time"])[-1]/3600.,2)))
        if opt_net.get_test_perf:
            ax[2].semilogy(save_file["Epoch"],save_file["Test Loss"], c_set[idx], linewidth=lwd, label="Test. Loss {} ({}h)".format(name,np.round(np.array(save_file["Time"])[-1]/3600.,2)))
            if "T_EMD" in save_file.columns.tolist():
                ax[3].semilogy(save_file["Epoch"],save_file["T_EMD"], c_set[idx], linewidth=lwd, label="Test. EMD {} ({}h)".format(name,np.round(np.array(save_file["Time"])[-1]/3600.,2)))

    else:
        ax[0].plot(save_file["Epoch"],save_file["Training Loss"], c_set[idx], linewidth=lwd, label="Tr.Loss {} ({}h)".format(name,np.round(np.array(save_file["Time"])[-1]/3600.),2))
        ax[1].plot(save_file["Epoch"],save_file["Validation Loss"], c_set[idx], linewidth=lwd, label="Val. Loss {} ({}h)".format(name,np.round(np.array(save_file["Time"])[-1]/3600.),2))
        if opt_net.get_test_perf:
            ax[2].plot(save_file["Epoch"],save_file["Test Loss"], c_set[idx], linewidth=lwd, label="Test. Loss {} ({}h)".format(name,np.round(np.array(save_file["Time"])[-1]/3600.,2)))
            if "T_EMD" in save_file.columns.tolist():
                ax[3].plot(save_file["Epoch"],save_file["T_EMD"], c_set[idx], linewidth=lwd, label="Test. EMD {} ({}h)".format(name,np.round(np.array(save_file["Time"])[-1]/3600.,2)))


ax[0].set_xlabel("Epochs", fontsize=15)
ax[0].set_ylabel("MSE Loss", fontsize=15)
ax[0].set_title("Training Loss", fontsize=15)
ax[1].set_xlabel("Epochs", fontsize=15)
ax[1].set_ylabel("MSE Loss", fontsize=15)
ax[1].set_title("Validation Loss", fontsize=20)

ax[-1].axis("off")
base_count = 0
for i,key in enumerate(best_data.keys()):
    if opt.set_baselines:
        if key in base_names:
            ax[-1].text(0,1-i*1./len(best_data.keys()),str(i)+" ({}h): ".format(times[i])+key,color=b_colorset[base_count])
            base_count += 1
        else:
            ax[-1].text(0,1-i*1./len(best_data.keys()),str(i)+" ({}h): ".format(times[i])+key,color=t_colorset[i])
    else:
        ax[-1].text(0,1-i*1./len(best_data.keys()),str(i)+" ({}h): ".format(times[i])+key, color=t_colorset[i])


for axs in ax.reshape(-1):
    for tick in axs.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in axs.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)


if opt_net.get_test_perf:
    ax[2].set_xlabel("Epoch", fontsize=15)
    ax[2].set_ylabel("MSE Loss", fontsize=15)
    ax[2].set_title("Test Loss", fontsize=20)
    if "T_EMD" in save_file.columns.tolist():
        ax[3].set_xlabel("Epoch", fontsize=15)
        ax[3].set_ylabel("MSE Loss", fontsize=15)
        ax[3].set_title("Test EMD", fontsize=20)
        f.set_size_inches(50,6)
        #f.suptitle("Training vs. Validation vs. Test Loss vs. EMD", fontsize=25)
    else:
        f.set_size_inches(40,6)
        #f.suptitle("Training vs. Validation vs. Test Loss", fontsize=25)
else:
    ax[0].legend()
    ax[1].legend()
    f.set_size_inches(34,5)
    #f.suptitle("Training vs. Validation Loss", fontsize=25)
f.tight_layout(rect=[0, 0.03, 1, 0.95])


if opt.save_plot!="":
    if opt.save_plot=="here":
        print "Saving plot to {}... ".format(os.getcwd()),
        plt.savefig(os.getcwd()+"/losscurves.svg")
        print "Complete."
    elif opt.save_plot=="everywhere":
        for base_folder,name in zip(opt.base_folder,opt.name):
            print "Saved file to: ",base_folder+"/losscurves.svg"
            plt.savefig(base_folder+"/losscurves.svg")
        print "Complete."
    else:
        print "Saving plot to {}... ".format(opt.save_plot)
        plt.savefig(opt.save_plot+"/losscurves.svg")
        print "Complete."


if opt.no_plots:
    plt.close()
else:
    plt.show()


### Set up Plot structure
if opt.inc_test_loss:
    if opt.inc_emd:
        f,ax = plt.subplots(1,5)
    else:
        f,ax = plt.subplots(1,4)
else:
    f,ax = plt.subplots(1,3)

titles = ["Lowest Training Loss", "Lowest Validation Loss", "Lowest Test Loss", "Lowest Test EMD"]
y_labs = ["Loss","Loss","Loss","EMD"]

ax[0].plot([best_data[key]["best train loss"] for key in best_data.keys()],b_colorset[0]+"o--")
ax[1].plot([best_data[key]["best val loss"] for key in best_data.keys()],b_colorset[1]+"o--")
ax[2].plot([best_data[key]["best test loss"] for key in best_data.keys()],b_colorset[2]+"o--")
ax[3].plot([best_data[key]["best test emd"] for key in best_data.keys()],b_colorset[3]+"o--")
for i,axs in enumerate(ax.reshape(-1)[:-1]):
    axs.set_xticks(range(len(best_data.keys())))
    axs.set_xticklabels(range(len(best_data.keys())),rotation="vertical")
    axs.set_title(titles[i],fontsize=25)
    axs.set_ylabel(y_labs[i], fontsize=20)
    for tick in axs.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in axs.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for label in axs.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

if opt.set_baselines:
    for i,base in enumerate(base_names):
        ax[0].axhline(best_data[base]["best train loss"],color=c_colorset[i])
        ax[1].axhline(best_data[base]["best val loss"],color=c_colorset[i])
        ax[2].axhline(best_data[base]["best test loss"],color=c_colorset[i])
        ax[3].axhline(best_data[base]["best test emd"],color=c_colorset[i])
    # ax[0].axhline(best_values[0],color="k")
    # ax[1].axhline(best_values[1],color="k")
    ax[2].axhline(best_values[2],color="k")
    ax[3].axhline(best_values[3],color="k")

ax[4].axis("off")
base_count = 0
for i,key in enumerate(best_data.keys()):
    if opt.set_baselines:
        if key in base_names:
            ax[4].text(0,1-i*1./len(best_data.keys()),str(i)+": "+key, color=c_colorset[base_count])
            base_count += 1
        else:
            ax[4].text(0,1-i*1./len(best_data.keys()),str(i)+": "+key)
    else:
        ax[4].text(0,1-i*1./len(best_data.keys()),str(i)+": "+key)

f.set_size_inches(len(ax)*7,5)
f.tight_layout()


if opt.save_plot!="":
    if opt.save_plot=="here":
        print "Saving plot to {}... ".format(os.getcwd()),
        plt.savefig(os.getcwd()+"/comparison.svg")
        print "Complete."
    elif opt.save_plot=="everywhere":
        for base_folder,name in zip(opt.base_folder,opt.name):
            print "Saved file to: ",base_folder+"/comparison.svg"
            plt.savefig(base_folder+"/comparison.svg")
        print "Complete."
    else:
        print "Saving plot to {}... ".format(opt.save_plot)
        plt.savefig(opt.save_plot+"/comparison.svg")
        print "Complete."

if opt.no_plots:
    plt.close()
else:
    plt.show()
