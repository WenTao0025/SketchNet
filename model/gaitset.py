import torch
import torch.nn as nn
import numpy as np
from Nets.basic_blocks import SetBlock,BasicConv2d
class SetNet(nn.Module):
    def __init__(self,hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None
        _set_in_channels = 1#输入的是二值图，所以channels为1
        _set_channels = [32,64,128]#这是C1，C2,C3,C4,C5,C6的channel
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels,_set_channels[0],5,padding = 2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0],_set_channels[0],3,padding = 1),True)#第二个参数默认为False，如果修改为True则证明需要池化
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0],_set_channels[1],3,padding = 1))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1],_set_channels[1],3,padding = 1),True)
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1],_set_channels[2],3,padding = 1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2],_set_channels[2],3,padding = 1))

        #MGP这一部分的CNN与池化操作
        _gl_in_channels = 32
        _gl_channels = [64,128]
        self.gl_layer1 = BasicConv2d(_gl_in_channels,_gl_channels[0],3,padding = 1)#第一次SP后的feature，输入gl_layer1做CNN，再经过layer2，再经过pooling
        self.gl_layer2 = BasicConv2d(_gl_channels[0],_gl_channels[0],3,padding = 1)
        self.gl_layer3 = BasicConv2d(_gl_channels[0],_gl_channels[1],3,padding = 1)#第二次SP后的feature + 前两层处理后的feature，经过layer3，layer4
        self.gl_layer4 = BasicConv2d(_gl_channels[1],_gl_channels[1],3,padding = 1)
        self.gl_pooling = nn.MaxPool2d(2)
        self.bin_num = [1,2,4,8,16]#HPM的5个scale
        #其实parameterList()就是一种和列表、元组之类一样的一种新的数据格式，用于保存神经网络权重及参数。
        #类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter,所以在参数优化的时候可以进行优化的），所以经过类型转换这个self.fc_bin变成了模型的一部分，成为了模型中根据训练可以改动的参数了
        self.fc_bin = nn.ParameterList([nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(sum(self.bin_num) * 2 , 128,hidden_dim)))])#xavier初始化，均匀分布#参数的形状为62*128*256
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data,0.0)
            elif isinstance(m,(nn.BatchNorm2d,nn.BatchNorm1d)):
                nn.init.normal(m.weight.data,1.0,0.02)
                nn.init.constant(m.bias.data,0.0)

    def frame_max(self,x):
        if self.batch_frame is None:
            return torch.max(x,1)
        else:
            _tmp = [
                torch.max(x[:,self.batch_frame[i]:self.batch_frame[i + 1],:,:,:],1)
                for i in range(len(self.batch_frame) - 1)
            ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))],0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))],0)
            return max_list,arg_max_list
    def frame_median(self,x):
        if self.batch_frame is None:
            return torch.median(x,1)
        else:
            _tmp = [
                torch.median(x[:,self.batch_frame[i]:self.batch_frame[i+1],:,:,:] , 1)
                for i in range(len(self.batch_frame) - 1)
            ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))] , 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))],0)
            return median_list,arg_median_list
    def forward(self,sliho,batch_frame = None):#sliho torch.size([-1,-1,224,224]
        #n:batch_size s:frame_num k:keypoints_num , c:channel






