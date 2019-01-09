# import numpy as np
# import matplotlib.pyplot as plt
#
# import os,time, datetime,json
# import cPickle as pkl
#
# import helper_functions as hf
# import prediction_network as prednet
#
# import torch
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
#
# import argparse



""" Plot angle distribution as histogram """
# def make_angle_dist_plot(img_set):
#     plt.style.use("ggplot")
#     plt.figure(figsize=(10,10))
#     a,b,c = plt.hist(img_set.training_files[3],100)
#     plt.title("Steering angle distribution", fontsize=20)
#     plt.xlabel("Steering Angle", fontsize=20)
#     plt.ylabel("Frequency", fontsize=20)
#     plt.tight_layout()
#     plt.savefig(os.getcwd()+"/Plots/steering_angle_distribution.pdf", pad=0)
#     plt.show()
#
#
# def make_crop_comp():
#     ol,new = train_img_set[0]
#     f,ax = plt.subplots(2,2)
#     ax[0,0].imshow(ol["imgs"][0])
#     ax[1,0].imshow(new["imgs"][0])
#     ax[0,1].imshow(ol["imgs"][1])
#     ax[1,1].imshow(new["imgs"][1])
#     titles = ["original center", "original right", "cropped center", "cropped right"]
#     for i,axs in enumerate(ax.reshape(-1)):
#         axs.set_xticks([])
#         axs.set_yticks([])
#         axs.set_title(titles[i])
#     f.tight_layout()
#     f.subplots_adjust(hspace=0.)
#     f.set_size_inches(8,4)
#     f.savefig("v_cropped.pdf")
#     plt.show()
#
# def make_steerimg():
#     i = 10
#     f,ax = plt.subplots(1,3)
#     cop = [hf.makenumpy(x) for x in train_img_set[i]["imgs"]]
#     tit = train_img_set[i]["angle"]
#     ax[0].imshow(cop[1])
#     ax[0].set_title("Steering angle: "+str(np.round(tit[1],3)))
#     ax[1].imshow(cop[0])
#     ax[1].set_title("Steering angle: "+str(np.round(tit[0],3)))
#     ax[2].imshow(cop[2])
#     ax[2].set_title("Steering angle: "+str(np.round(tit[2],3)))
#     txt = ["left", "center", "right"]
#     for j,axs in enumerate(ax.reshape(-1)):
#         axs.set_xticks([])
#         axs.set_yticks([])
#         axs.set_xlabel(txt[j], fontsize=15)
#     f.tight_layout()
#     f.subplots_adjust(wspace=0.1)
#     f.set_size_inches(18,14)
#     f.savefig("steering_angle_corr_with_jitter.pdf")
#     plt.show()


""" Pic Dump """
# f,ax = plt.subplots(3,3)
# height = 160
# width  = 320
#
# nrr = np.random.randint
# nrc = np.random.choice
# bound=5
# sort_b = np.random.choice([False,True])
# sort_t = np.random.randint(2)
# for axs in ax.reshape(-1):
#     complexity = np.random.randint(2,20)
#     base_l = {"polygon":[(0,h) for h in nrc(range(0,height-bound),2,replace=False)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#     base_t = {"polygon":[(w,0) for w in nrc(range(0,width-bound),2,replace=False)], "type":sort_t, "rev":True, "sort":sort_b, "insert":True}
#     base_r = {"polygon":[(width-1,h) for h in nrc(range(0,height-bound),2, replace=False)], "type":sort_t, "rev":True, "sort":sort_b, "insert":True}
#     base_b = {"polygon":[(w,height-1) for w in nrc(range(0,width-bound),2, replace=False)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#
#     base_lct = {"polygon":[(0,nrr(bound,height)),(0,0),(nrr(bound,width),0)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#     base_tcr = {"polygon":[(nrr(width-bound),0),(width-1,0),(width-1,nrr(bound,height))], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#     base_rcb = {"polygon":[(0,nrr(height-bound)),(width-1,height-1),(nrr(bound,width),height-1)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#     base_bcl = {"polygon":[(nrr(bound,width),0),(0,height-1),(0,nrr(height-bound))], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#
#
#     base_lr  = {"polygon":[(0,nrr(bound,height-bound)),(0,0),(width-1,0),(width-1,nrr(bound,height-bound))], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#     base_rl  = {"polygon":[(width-1,nrr(bound,height-bound)),(width-1,height-1),(0,height-1),(0,nrr(bound,height-bound))], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#     base_tb  = {"polygon":[(nrr(bound,width-bound),0),(width-1,0),(width-1,height-1),(nrr(bound,width-bound),height-1)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#     base_bt  = {"polygon":[(nrr(bound,width-bound),height-1),(0,height-1),(0,0),(nrr(bound,width-bound),0)], "type":sort_t, "rev":False, "sort":sort_b, "insert":True}
#
#     bases = [base_l, base_t, base_r, base_b, base_lct, base_tcr, base_rcb, base_bcl, base_lr, base_rl, base_tb, base_bt]
#     base  = np.random.choice(bases)
#     fill_points     = [list(set([(x,y) for x,y in zip(np.random.randint(0,height,complexity),np.random.randint(0,width,complexity))]))]
#     if base["sort"]:
#         fill_points     = sorted(fill_points[0],key=lambda i:i[base["type"]], reverse=base["rev"])
#     for x in fill_points[0]:
#         if base["insert"]:
#             base["polygon"].insert(1,x)
#         else:
#             base["polygon"].extend(x)
#
#     res = ims["imgs"][0]
#     img = Image.new('L', (width, height), 0)
#     ImageDraw.Draw(img).polygon(base["polygon"], outline=1, fill=1)
#     mask = np.array(img)
#
#     alpha=0.25+0.7*np.random.uniform()
#     im = np.array(res)
#     img_hsv = color.rgb2hsv(im)
#     mask1 = mask==1
#     # img_hsv[:,:,0][mask] = img_hsv[:,:,0][mask]*alpha
#     img_hsv[:,:,2][mask1] = img_hsv[:,:,2][mask1]*alpha
#     # img_hsv[:,:,2][mask2] = img_hsv[:,:,2][mask2]*alpha
#     # img_hsv[:,:,2][mask] = img_hsv[:,:,2][mask]*alpha
#     img_rgb = color.hsv2rgb(img_hsv)
#     km = (img_rgb*255).astype(int)
#     lsm = Image.fromarray(km.astype(np.uint8))
#     axs.imshow(img_rgb)
#     axs.set_xticks([])
#     axs.set_yticks([])
# f.tight_layout(rect=[0, 0.03, 1, 0.95])
# f.subplots_adjust(hspace=0.2)
# f.set_size_inches(15,8)
# f.suptitle("Low complexity shadows - vertically aligned",fontsize=20)
# f.savefig("low_ver_compl_shad.pdf")
# plt.show()


""" Map Testing """
###Testing mapping, but its not really working - not enough information
# writer = hf.CSVlogger(os.getcwd()+"/spd_log2.csv",["Time", "Angle", "Throttle", "Speed"])
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# data = pd.read_csv("spd_log.csv")
# ang = np.array(data["Angle"][1:]).astype(float)
#
# stang = ang*39./25
# car_lgn = 65.
#
# spd = np.array(data["Speed"][1:]).astype(float)
# tim = np.array(data["Time"][1:]).astype(float)
# thr = np.array(data["Throttle"][1:]).astype(float)
#
# mvmt   = spd*tim + 0.5*thr*tim**2
# plt.plot(ang)
# plt.show()
# plt.plot(mvmt)
# plt.show()
#
# def gp(angc=1.6, tr_x=0,tr_y=0):
#     pos = [(0,0)]
#     defl= 0
#     xc = []
#     yc = []
#     defs = []
#     for i in xrange(len(ang)):
#         ov      = pos[-1]
#         defl   += ang[i]*angc/180.*np.pi
#         defl    = defl%(2*np.pi)
#
#         x_corr = np.sin(defl)*mvmt[i]-tr_x
#         y_corr = np.cos(defl)*mvmt[i]-tr_y
#         defs.append(defl)
#         xc.append(x_corr)
#         yc.append(y_corr)
#         pos.append((ov[0]+x_corr,ov[1]+y_corr))
#     km = np.array(pos[1:])
#     plt.plot(km[:,0], km[:,1],".",markersize=0.1)
#     plt.title(str(angc))
#     plt.show()
#     plt.plot(defs)
#     plt.show()
