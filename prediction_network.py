import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import datetime
import numpy as np


""" NVIDIA Network Proposal"""
class nvidia_net(nn.Module):
    def __init__(self, kernel_sizes=[5,5,5,3,3], image_size=[3,160,320], n_filters=[24, 36, 48, 64, 64], use_pooling = False, pools=[0,0,0,0,0], use_BN=False,\
                 fc_layers=[500, 100, 50, 10], conv_dropout=[0.3,0.2,0.2,0.2,0], fc_dropout=[0.3,0.2,0.2,0], make_full_conv=False,\
                 strides = [2,2,2,2,1], conv_activation=nn.ReLU(), fc_activation=nn.ReLU(), n_channels=3, extract_features=False):
        super(nvidia_net, self).__init__()

        ### Sanity Checks
        assert len(kernel_sizes)==len(strides)==len(n_filters), "Length of Kernel size list must match stride list and filter list length!"
        assert image_size[0]%3==0, "Number of channels must be multiples of 3."
        assert (len(conv_dropout)==len(kernel_sizes) or isinstance(conv_dropout,int)) and\
               (len(fc_dropout)==len(fc_layers) or isinstance(fc_dropout,int)), "Dropout input list length must match layer list length or be integers!"


        ### Setting up the convolutional layers via ModuleList
        conv_layer_list = []
        fc_layer_list   = []
        self.n_filters  = [image_size[0]]+n_filters
        for i in xrange(len(kernel_sizes)):
            conv_layer_list.append(nn.Conv2d(self.n_filters[i],self.n_filters[i+1],kernel_sizes[i], strides[i]))
        self.conv_layers = nn.ModuleList(conv_layer_list)


        ### Layer shapes
        self.kernel_sizes = kernel_sizes
        self.strides      = strides
        widths      = {"output_sizes":[], "pad_on_input":[]}
        heights     = {"output_sizes":[], "pad_on_input":[]}
        self.heights     = self.get_sizes(image_size[2], heights)
        self.widths      = self.get_sizes(image_size[1], widths)

        ### Use pooling
        self.use_pool = use_pooling
        if use_pooling:
            self.pools = pools

        ### Dropout Lists
        if isinstance(conv_dropout, int):
            conv_dropout = list(np.repeat(conv_dropout,len(kernel_sizes)))
        if isinstance(fc_dropout, int):
            fc_dropout   = list(np.repeat(fc_dropout,len(fc_layers)))

        self.conv_dropout= conv_dropout
        self.fc_dropout  = fc_dropout
        self.conv_drops  = nn.ModuleList([nn.Dropout2d(p) for p in conv_dropout if p])
        self.fc_drops    = nn.ModuleList([nn.Dropout(p) for p in fc_dropout if p])


        ### Activations
        self.conv_activation    = conv_activation   #in-conv-layer activation
        self.fc_activation      = fc_activation     #in-fc-layer activation
        self.out_activation     = nn.Tanh()         #output activation to generate usable angle values

        self.make_full_conv = make_full_conv
        if self.make_full_conv:
            ### Make the network fully convolutional.
            self.conv_pre_out       = nn.Conv2d(self.n_filters[-1],self.n_filters[-1],1,1)
            self.pre_out_activation = conv_activation
            self.conv_out           = nn.Conv2d(self.n_filters[-1],1,1,1)
            self.out_pool           = F.avg_pool2d      #Avg. Pooling as fully-connected replacement
        else:
            ### Setting up the fully connected layers via ModuleList.
            self.fc_layer_nums       = fc_layers
            self.fc_layer_nums       = [n_filters[-1]*self.widths["output_sizes"][-1]*self.heights["output_sizes"][-1]]+self.fc_layer_nums+[1]
            for i in xrange(len(self.fc_layer_nums)-1):
                fc_layer_list.append(nn.Linear(self.fc_layer_nums[i], self.fc_layer_nums[i+1]))
            self.fc_layers  = nn.ModuleList(fc_layer_list)


        ### Make BatchNormalization List
        if use_BN:
            if not self.make_full_conv:
                self.BN_list_fc = nn.ModuleList([nn.BatchNorm1d(self.fc_layer_nums[i+1]) for i in xrange(len(self.fc_layer_nums)-1)])
            self.BN_list_cv = nn.ModuleList([nn.BatchNorm2d(self.n_filters[i+1]) for i in xrange(len(self.n_filters)-1)])
        self.use_BN = use_BN


        ### Creating a unique save name for repurpose purposes
        rundate     = datetime.datetime.now()
        savetime    = "{}{}{}_{}{}{}".format(rundate.day, rundate.month, rundate.year, rundate.hour, rundate.minute, rundate.second)
        self.name   = "NVIDIA_net_"+savetime

        self.extract_features = extract_features
        ### Setting a reconstruction dict with which this network can be directly replicated.
        self.reconstruction = {"kernel_sizes":      kernel_sizes,
                               "image_size":        image_size,
                               "n_filters":         n_filters,
                               "use_BN":            use_BN,
                               "fc_layers":         fc_layers,
                               "strides":           strides,
                               "conv_dropout":      conv_dropout,
                               "fc_dropout":        fc_dropout,
                               "conv_activation":   conv_activation,
                               "fc_activation":     fc_activation,
                               "make_full_conv":    make_full_conv,
                               "use_pooling":       use_pooling,
                               "pools":             pools}


    def forward(self, x):
        ### Forward pass through the convolutional layers
        for i in xrange(len(self.conv_layers)):
            #pad input to correct (same) size for further convolutional layer
            x = F.pad(x,(0,self.heights["pad_on_input"][i],0,self.widths["pad_on_input"][i]))
            #convolutional layer
            x = self.conv_layers[i](x)
            #use Batch Normalization
            if self.use_BN:
                x =  self.BN_list_cv[i](x)
            #use convolutional activation
            x = self.conv_activation(x)
            #use dropout
            if self.conv_dropout[i]!=0:
                x = self.conv_drops[i](x)

            if self.extract_features:
                if i == 0:
                    first_feature_layer = x
                elif i==1:
                    second_feature_layer = x
            if self.use_pool:
                if self.pools[i]==1:
                    x = F.pad(x,(0,x.size()[2]%2,0,x.size()[3]%2))
                    x = F.avg_pool2d(x,2)

        if self.extract_features:
            last_feature_layer = x


        if self.make_full_conv:
            ### Forward pass through final convolutional layer and average.
            x = self.conv_pre_out(x)
            x = self.pre_out_activation(x)
            x = self.conv_out(x)
            x = self.out_pool(x, x.size()[2:])
            x = self.out_activation(x)
        else:
            ### Forward pass through the fully connected end-layers
            x = x.view(x.size(0),-1)
            for i in xrange(len(self.fc_layers)):
                #fully-connected layer
                x = self.fc_layers[i](x)
                if i<len(self.fc_layers)-1:
                    #use Batch Normalization
                    if self.use_BN:
                        x =  self.BN_list_fc[i](x)
                    #use fully-connected activation
                    x = self.fc_activation(x)
                    #use dropout
                    if self.fc_dropout[i]!=0:
                        x = self.fc_drops[i](x)
                else:
                    #output activation
                    x = self.out_activation(x)
        if self.extract_features:
            return first_feature_layer, second_feature_layer, last_feature_layer, x
        else:
            return x


    def get_sizes(self, input_size, sizes, state=0):
        """ Function to derive the output and padding values """
        output_size = (input_size+1-self.kernel_sizes[state])/self.strides[state]+1 if input_size%2==0 else (input_size-self.kernel_sizes[state])/self.strides[state]+1
        to_pad      = 1 if input_size%2==0 else 0
        sizes["output_sizes"].append(output_size)
        sizes["pad_on_input"].append(to_pad)

        if state==len(self.kernel_sizes)-1:
            return sizes
        return self.get_sizes(output_size, sizes, state+1)


    def weight_init(self):
        for net_segment in self.modules():
            if isinstance(net_segment, nn.Conv2d):# or isinstance(net_segment, nn.Linear):
                init.kaiming_normal(net_segment.weight.data)
                init.constant(net_segment.bias.data, 0)
            elif isinstance(net_segment, nn.BatchNorm2d):
                net_segment.weight.data.fill_(1)
                net_segment.bias.data.zero_()
            elif isinstance(net_segment, nn.Linear):
                init.kaiming_normal(net_segment.weight.data)
                init.constant(net_segment.bias.data, 0)



""" Find total parameter number """
def gimme_params(angle_net, sum_only=True):
    parameters = [np.sum([np.prod(x.data.numpy().shape) for x in list(module.parameters()) if len(list(module.parameters()))!=0 ]) for module in list(angle_net.modules())[1:]]
    number_of_parameters    = np.sum(parameters)

    if not sum_only:
        layer_names= [str(type(module)).split(">")[0].split(".")[-1][:-1] for module in list(angle_net.modules())[1:]]
        out_string = "\n".join("["+str(i+1)+"] "+name+":"+str(param) for i,(name,param) in enumerate(zip(layer_names, parameters)))

        print 'Initialized model with {} parameters.'.format(number_of_parameters)
        print 'Layer setup:#Parameters'
        print out_string
        print '---------\n'

    return int(number_of_parameters)
