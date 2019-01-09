import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cPickle as pkl

"""
After running this script, go to the plot folder and run:
    ffmpeg -framerate 15 -i image_%000d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p name_of_choice.mp4
"""

base_path   = os.getcwd()+"/Plots/FeatureVisualisation"
folders     = [base_path+"/"+x for x in os.listdir(base_path) if "mov_" in x]
save_names  = [x for x in os.listdir(base_path) if "mov_" in x]
video_names = ["Test: Road 1 | Train: Road 1","Test: Road 1 | Train: Road 2","Test: Road 2 | Train: Road 2"]
folders.sort()
save_names.sort()

for jk,folder in enumerate(folders):
    files = [folder+"/"+x for x in os.listdir(folder)]
    files = sorted(files,key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))
    file_holder = []
    print "Creating video files [{}/{}]...".format(jk+1,len(folders))
    for i,filename in enumerate(files):
        try:
            with open(filename,"rb") as kl:
                file_holder.append(pkl.load(kl))
            print "Loading Progress: {}/{}\r".format(i+1,len(files)),
            sys.stdout.flush()
        except:
            pass
    save_folder = os.getcwd()+"/Plots/FeatureVisualisation/"+save_names[jk]+"_saveplots"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for jm,data in enumerate(file_holder):
        coll_list = []
        for k in xrange(len(data)-1)  :
            coll_list.extend(list(data[k+1]))
        fig, ax = plt.subplots(3,len(data[1])+1)
        count = 0
        count2= 0
        im_types = ["First Layer", "Second Layer", "Last Layer"]
        for i,axs in enumerate(ax.reshape(-1)):
            if i==len(data[1])+1:
                axs.imshow(data[0])
                axs.set_xticks([])
                axs.set_yticks([])
                axs.set_title("Camera")
            elif i!=0 and i%(len(data[1])+1)!=0:
                axs.set_xticks([])
                axs.set_yticks([])
                axs.imshow(1-coll_list[count],cmap="gray_r")
                count+=1
            else:
                axs.axis("off")
            if i%(len(data[1])+1)==1:
                axs.set_title(im_types[count2],fontsize=10)
                count2+=1

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1)
        fig.set_size_inches(5*len(data)+5,5)
        fig.savefig(save_folder+"/image_"+str(jm)+".png")
        plt.close()
        plt.show()
        print "Plotting Progress: {}/{}\r".format(jm+1,len(file_holder)),
        sys.stdout.flush()
