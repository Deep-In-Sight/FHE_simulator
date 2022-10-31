import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch.nn import functional as F
from hlee_model import ConvNeuralNet_simple as cnn
from hlee_utils import *

class TorchConv_infer(cnn):
    def __init__(self, num_classes, ho,wo,co, activation=F.relu, fn_param="", device='cpu'):
        super().__init__(num_classes, co=co, activation=F.relu)
        self.fn_param = fn_param
        trained_param = torch.load(fn_param, map_location = torch.device(device))
        trained_param = {key : value.cpu()   for key,value in trained_param.items()}
        params_np     = {key : value.numpy() for key,value in trained_param.items()}
        self.load_state_dict(trained_param)
        self.eval()
        U = self.conv_layer1.weight
        Ut = self.conv_layer1.weight.clone().detach()
        self.U_multconv = get_channel_last(U)
        self.U_torchconv = Ut.type(torch.DoubleTensor)
        print(f"[cnn] U_multconv  : {self.U_multconv.shape}")
        print(f"[cnn] U_torchconv : {self.U_torchconv.shape}")
        self.ho=ho
        self.wo=wo
        self.co=co
    def TorchConv(self,fname):
        image = cv2.imread(fname)
        image = cv2.resize(image,(self.ho,self.wo))
        img = get_channel_first(image)
        img = torch.tensor(img)
        img = img.type(torch.DoubleTensor)
        return F.conv2d(img,self.U_torchconv,padding="same")
    def TorchConv_single(self,img):
        #image = cv2.imread(fname)
        #image = cv2.resize(image,(self.ho,self.wo))
        img = get_channel_first(img)
        img = torch.tensor(img)
        img = img.type(torch.DoubleTensor)
        return F.conv2d(img,self.U_torchconv,padding="same")
        

