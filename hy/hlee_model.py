import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch.nn import functional as F



class ConvNeuralNet_simple(nn.Module):
    def __init__(self,num_classes,activation=F.relu,co=0):
        super().__init__()
        print(f"channel out : {co}")
        
        #32,32,3
        if co==4:
            print(f"channel out : {co}")
        #simple_model_hlee.pt
            self.conv_layer1 = nn.Conv2d(in_channels=3,  out_channels=4, kernel_size=3, padding="same") #32,32,4
            self.conv_layer2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding="same") #32,32,16
        elif co==8:
            print(f"channel out : {co}")
        #simple_model_hlee_co8.pt
            self.conv_layer1 = nn.Conv2d(in_channels=3,  out_channels=8, kernel_size=3, padding="same") #32,32,8
            self.conv_layer2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding="same") #32,32,16
        elif co==16:
            print(f"channel out : {co}")
        #simple_model_hlee_co16.pt
            self.conv_layer1 = nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=3, padding="same") #32,32,8
            self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding="same") #32,32,16
        

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)                                           #16,16,16
        self.conv_layer3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same")#16,16,32
        self.bn1 = nn.BatchNorm2d(32)                    
        self.activation = activation 
        #self.pool = nn.AvgPool2d(kernel_size=2, stride=2)                                          # 8, 8,32
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")# 8, 8,64
        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")# 8, 8,64     
        self.bn2 = nn.BatchNorm2d(64)                                                               # 8, 8,64
        #self.activation = activation 
        #self.pool = nn.AvgPool2d(kernel_size=2, stride=2)                                          # 4, 4,64 ->4*4*64=1024
        self.fc1 = nn.Linear(1024,128)                                                              
        #self.activation = activation 
        self.fc2 = nn.Linear(128,num_classes)
               
    def forward(self,x):
        #  [FC2] <----(A) < FC1 < (flatten) < (p) < (A) < BN2 < C5 < C4 < (p) < (A) < BN1 < C3 < (p) < C2 < C1<----[x]
        out = self.activation(self.bn1(self.conv_layer3(self.pool(self.conv_layer2(self.conv_layer1(x))))))
        out = self.activation(self.bn2(self.conv_layer5(self.conv_layer4(self.pool(out)))))
        out = self.pool(out)
        out = out.reshape(out.size(0),-1)
        out = self.activation(self.fc1(out))
        return self.fc2(out)
    
