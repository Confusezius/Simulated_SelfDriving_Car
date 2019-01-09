""" Libraries """
import argparse
import base64
import cPickle as pkl
import os


import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import prediction_network as prednet
import helper_functions as hf

import sys
import time


""" Input Argument Parser """
parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument("--name",           default="", help="Network name")
parser.add_argument("--base_folder",    default=os.getcwd()+"/Results/", help="Network name")
parser.add_argument("--cuda",           action="store_true", help="Use cuda.")
parser.add_argument("--was_not_cuda",   action="store_true", help="Training was not performed on cuda.")
parser.add_argument("--speed",          type=float, default=9, help="Network name")
parser.add_argument("--extract_features",action="store_true", help="Whether to extract features of first, second and last layer during forward pass (default=False)")
parser.add_argument("--savename",       type=str, default="mov_1",help="Savename for video data.")
opt = parser.parse_args()

global glob_count
glob_count = 0
input_coll = []


""" Allow the user to choose the set of weights to initialize the network with """
if opt.name=="":
    folder_files    = os.listdir(opt.base_folder)
    folder_files.sort()
    output_string   = "Choose folder to drive with:\n"
    valid_nums      = []
    for i,ff in enumerate(folder_files):
        valid_nums.append(i)
        output_string+="["+str(i)+"] "+ff+"\n"
    print output_string
    foldernum = input("\nPlease enter the respective number:")
    if int(foldernum) in valid_nums:
        print "Initializing network with {}... ".format(folder_files[foldernum]),
    else:
        raise ValueError("Invalid number entered!")

    opt.base_folder +="/"+folder_files[foldernum]
    opt.name         = folder_files[foldernum]
else:
    opt.base_folder +="/"+opt.name
    print ">>> Initializing Network with {}... ".format(opt.name),


class Mapper():
    def __init__(self):
        self.positions = []

    def update(self, speed, angle, time):
        pass

# mapper =


"""=========================================================================="""
"""================== Network Setup ========================================="""
"""=========================================================================="""
### Load saved parameter set
net_opt     = pkl.load(open(opt.base_folder+"/hypa.pkl"))
### Reconstruct the network.
net_type    = net_opt.name.split("_")[0]
use_multi   = net_opt.channels//3
used_shape  = net_opt.image_shape[1:][::-1] #Using PIL-style shapes, so need to convert to (width,height)
if net_type=="NVIDIA":
    angle_net = prednet.nvidia_net(extract_features=opt.extract_features, **net_opt.reconstruction)
else:
    raise NotImplementedError("Network type not implemented!")


load_name = opt.base_folder+"/"+opt.name
if opt.cuda and not opt.was_not_cuda:
    _ = angle_net.cuda()
    angle_net.load_state_dict(torch.load(load_name))
if not opt.cuda and not opt.was_not_cuda:
    angle_net.load_state_dict(torch.load(load_name, map_location=lambda storage, loc: storage))
if not opt.cuda and opt.was_not_cuda:
    angle_net.load_state_dict(torch.load(load_name))
print "Complete."
angle_net.eval()



"""=========================================================================="""
"""================== Car Controller ========================================"""
"""=========================================================================="""
class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

controller = SimplePIController(0.1, 0.002)
set_speed = opt.speed
controller.set_desired(set_speed)


"""=========================================================================="""
"""================== Server Comms =========================================="""
"""=========================================================================="""
sio = socketio.Server()
app = Flask(__name__)
prev_image_array = None

global start
start = time.time()
# fig = plt.figure(figsize=(6,2))



idxs1      = []
idxs2      = []
idxs3      = []

@sio.on('telemetry')
def telemetry(sid, data):
    global glob_count, start
    if data:
        steering_angle  = data["steering_angle"]
        throttle        = data["throttle"]
        speed           = data["speed"]
        imgString       = data["image"]
        image           = Image.open(BytesIO(base64.b64decode(imgString)))
        # image.save("image_"+str(glob_count)+".png")
        # glob_count+=1
        image  = hf.crop_PIL(image, used_shape, net_opt.no_center_crop)
        inputs = torch.unsqueeze(transforms.ToTensor()(image),0)

        if use_multi>1:
            inputcoll.insert(0,inputs)
            if len(inputcoll)==2:
                inputcoll.insert(0,inputs)
            elif len(inputcoll)==1:
                inputcoll.insert(0,inputs)
                inputcoll.insert(0,inputs)
            else:
                del inputcoll[-1]
            inputs = torch.cat(inputcoll,dim=1)

        inputs = Variable(inputs)

        if opt.cuda:
            inputs = inputs.cuda()


        if opt.extract_features:
            first_feature_layer, second_feature_layer, last_feature_layer, steering_angle = angle_net(inputs)
            steering_angle = float(steering_angle)

            # plt.ion()
            if glob_count%15==0 and glob_count!=0:
                if opt.cuda:
                    first_feature_layer = first_feature_layer.cpu().data.numpy()
                    second_feature_layer= second_feature_layer.cpu().data.numpy()
                    last_feature_layer  = last_feature_layer.cpu().data.numpy()
                # Plot input together with feature maps from first, second and last feature extraction layer
                n_plts          = 5
                # first_features  = range(10,10+n_plts)
                # second_features = range(10,10+n_plts)
                # last_features   = range(10,10+n_plts)

                if not os.path.exists(os.getcwd()+"/Plots/FeatureVisualisation/"+opt.savename):
                    os.makedirs(os.getcwd()+"/Plots/FeatureVisualisation/"+opt.savename)

                if len(idxs1)==0:
                    for i,fmap in enumerate(first_feature_layer[0,:]):
                        if np.max(fmap)>0.1:
                            idxs1.append(i)
                        if len(idxs1)==n_plts:
                            break
                    for i,fmap in enumerate(second_feature_layer[0,:]):
                        if np.max(fmap)>0.1:
                            idxs2.append(i)
                        if len(idxs2)==n_plts:
                            break
                    for i,fmap in enumerate(last_feature_layer[0,:]):
                        if np.max(fmap)>0.1:
                            idxs3.append(i)
                        if len(idxs3)==n_plts:
                            break


                to_save = [np.array(image),first_feature_layer[0,idxs1,:,:], second_feature_layer[0,idxs2,:,:],last_feature_layer[0,idxs3,:,:]]
                with open(os.getcwd()+"/Plots/FeatureVisualisation/"+opt.savename+"/glob_"+str(glob_count)+".pkl","wb") as fl:
                    pkl.dump(to_save, fl)

            glob_count+=1
        else:
            steering_angle = float(angle_net(inputs))

        throttle = controller.update(float(speed))
        print "Current steering angle: {}     (throttle: {})\r".format(np.round(steering_angle,4), np.round(throttle,4)),
        sys.stdout.flush()
        send_control(steering_angle,throttle)

        # writer.write([time.time()-start, steering_angle, throttle, float(speed)])
        start = time.time()



        # save frame
        # if args.image_folder != '':
        #     timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        #     image_filename = os.path.join(args.image_folder, timestamp)
        #     image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


"""=========================================================================="""
"""================== Start Host ============================================"""
"""=========================================================================="""
app = socketio.Middleware(sio, app)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
