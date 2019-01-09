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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

import prediction_network_for_feature_extraction as prednet
import helper_functions as hf

import sys




""" Input Argument Parser """
parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument("--name",           default="", help="Network name")
parser.add_argument("--base_folder",    default=os.getcwd()+"/Results/", help="Network name")
parser.add_argument("--cuda",           action="store_true", help="Use cuda.")
parser.add_argument("--was_not_cuda",   action="store_true", help="Training was not performed on cuda.")
parser.add_argument("--speed",          type=float, default=9, help="Network name")

opt = parser.parse_args()


global glob_count
glob_count = 0

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

net_opt   = pkl.load(open(opt.base_folder+"/hypa.pkl"))

"""=========================================================================="""
"""================== Network Setup ========================================="""
"""=========================================================================="""
### Load saved parameter set
net_opt   = pkl.load(open(opt.base_folder+"/hypa.pkl"))
### Reconstruct the network.
net_type    = net_opt.name.split("_")[0]
use_multi   = net_opt.channels
used_shape  = net_opt.image_shape[1:][::-1] #Using PIL-style shapes, so need to convert to (width,height)

if net_type=="NVIDIA":
    angle_net = prednet.nvidia_net(**net_opt.reconstruction)
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

@sio.on('telemetry')
def telemetry(sid, data):
    global glob_count
    if data:
        steering_angle = data["steering_angle"]
        throttle = data["throttle"]
        speed = data["speed"]
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image  = hf.crop_PIL(image, used_shape)
        inputs = Variable(torch.unsqueeze(transforms.ToTensor()(image),0))

        if opt.cuda:
            inputs = inputs.cuda()
        first_feature_layer, second_feature_layer, last_feature_layer, steering_angle = angle_net(inputs)
        if opt.cuda:
            first_feature_layer.cpu(), second_feature_layer.cpu(), last_feature_layer.cpu()

        steering_angle = float(steering_angle)
        throttle = controller.update(float(speed))

        # Plot input together with feature maps from first, second and last feature extraction layer
        first_features = np.random.choice(net_opt.n_filters[0],size=9,replace=False)
        second_features = np.random.choice(net_opt.n_filters[1],size=9,replace=False)
        last_features = np.random.choice(net_opt.n_filters[-1],size=9,replace=False)
        fig = plt.figure(figsize=(30,5))
        for plt_num in range(1,31):
            plt.subplot(3,10,plt_num)
            if plt_num==1:
                plt.imshow(image)
            elif 2 <= plt_num <= 10:
                plt.imshow(first_feature_layer.data.numpy()[0,first_features[plt_num-2],:,:], cmap='gray')
            elif 12 <= plt_num <= 20:
                plt.imshow(second_feature_layer.data.numpy()[0,second_features[plt_num-12],:,:], cmap='gray')
            elif 22 <= plt_num <= 30:
                plt.imshow(last_feature_layer.data.numpy()[0,last_features[plt_num-22],:,:], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.getcwd()+"/Plots/FeatureVisualisation/Features_"+str(glob_count)+".pdf")
        plt.close()
        glob_count+=1

        print "Current steering angle: {}     (throttle: {})\r".format(np.round(steering_angle,4), np.round(throttle,4)),
        sys.stdout.flush()
        send_control(steering_angle, throttle)

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
