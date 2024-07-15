import torch
from torch import nn
import torch.nn.functional as F
import math

"""
    # double factorial
    # 3!! = 3 * 1 
    # 4!! = 4 * 2
    # 5!! = 5 * 3 * 1
    # 6!! = 6 * 4 * 2
    # n!! = n * (n-2) * (n-4) * ... * 1 if n is odd
    # n!! = n * (n-2) * (n-4) * ... * 2 if n is even
    # (n-3)!! = 1 or 1*3 or 1*3*5. maybe 1*3. 
"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, 
                        stride=1, 
                        affine=False, 
                        activation=F.relu):
        self.activation = activation

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(
                              in_planes,
                              self.expansion * planes,
                              kernel_size=1,
                              stride=stride,
                              bias=False), 
                            nn.BatchNorm2d(self.expansion * planes, affine=affine))


    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

    # def validate(self, x):
    #     out = self.bn1(self.conv1(x))
    #     omi1 = out.min()
    #     oma1 = out.max()
    #     out = self.activation(out)
    #     out = self.bn2(self.conv2(out))
    #     out += self.shortcut(x)
    #     omi2 = out.min()
    #     oma2 = out.max()
    #     out = self.activation(out)
    #     return out, (min(omi1, omi2), max(oma1, oma2))

import torch.nn.functional as F

class HermitePN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(HermitePN, self).__init__()
        self.num_features = num_features
        
        # Batch normalization layer
        #self.batch_norm = nn.BatchNorm2d(num_features)
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean1', torch.zeros(num_features))
        self.register_buffer('running_var1', torch.ones(num_features))
        
        # # Running statistics
        # self.register_buffer('running_mean2', torch.zeros(num_features))
        # self.register_buffer('running_var2', torch.ones(num_features))
    
    def _normalize1(self, x):
        if self.training:
            # Calculate mean and variance
            batch_mean = x.mean([0, 2, 3], keepdim=True)
            batch_var = x.var([0, 2, 3], unbiased=False, keepdim=True)
            
            # Update running statistics
            self.running_mean1 = (1 - self.momentum) * self.running_mean1 + self.momentum * batch_mean.view(-1)
            self.running_var1 = (1 - self.momentum) * self.running_var1 + self.momentum * batch_var.view(-1)
        else:
            # Use running statistics
            batch_mean = self.running_mean1.view(1, self.num_features, 1, 1)
            batch_var = self.running_var1.view(1, self.num_features, 1, 1)
        
        # Normalize
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        return x_hat    
    
    def _normalize2(self, x):
        if self.training:
            # Calculate mean and variance
            batch_mean = x.mean([0, 2, 3], keepdim=True)
            batch_var = x.var([0, 2, 3], unbiased=False, keepdim=True)
            
            # Update running statistics
            self.running_mean2 = (1 - self.momentum) * self.running_mean2 + self.momentum * batch_mean.view(-1)
            self.running_var2 = (1 - self.momentum) * self.running_var2 + self.momentum * batch_var.view(-1)
        else:
            # Use running statistics
            batch_mean = self.running_mean2.view(1, self.num_features, 1, 1)
            batch_var = self.running_var2.view(1, self.num_features, 1, 1)
        
        # Normalize
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        return x_hat    
        
    def _scale(self, x):
        return self.gamma.view(1, self.num_features, 1, 1) * x + self.beta.view(1, self.num_features, 1, 1)


    def forward(self, x):
        # print("HermitePN begining", x.min(), x.max())
        H0 = 1 
        H1 = x
        H2 = x**2 - 1

        # Normalized Hermite polynomials
        h0 = H0
        h1 = H1
        h2 = H2/math.sqrt(2)

        h0n = h0 
        h1n = self._normalize1(h1)
        h2n = self._normalize1(h2)
        
        # print("h1n", h1n.min(), h1n.max())
        # print("h2n", h2n.min(), h2n.max())
        # Coefficients of Hermite approximation of ReLU
        
        f0 = 1/math.sqrt(2*math.pi)
        f1 = 1/2
        f2 = 1/math.sqrt(2*math.pi*2) 

        fh0n = h0n * f0
        fh1n = h1n * f1
        fh2n = h2n * f2         

        # # Coefficients of Hermite approximation of ReLU
        # f0 = 1/math.sqrt(2*math.pi)
        # f1 = 1/2
        # f2 = 1/math.sqrt(2*math.pi*2) 

        # fh0 = h0 * f0
        # fh1 = h1 * f1
        # fh2 = h2 * f2 
        
        # fh0n = fh0 
        # fh1n = self._normalize1(fh1)
        # fh2n = self._normalize2(fh2)

        #print("BN", self.running_mean1, self.running_var1, self.running_mean2, self.running_var2)
        
        # scale(gamma)과 bias(beta)는 learnable parameter임. 
        # gamma1, gamma2, gamma3, ... 로 여러 gamma를 유지할 것이 아니므로, 모든 term을 합친 뒤 최종적으로 gamma와 beta 적용.
        # if self.training:
        #     print("training....")
        #print("fh1", fh1n.min(), fh1n.max())
        #print("fh2", fh2n.min(), fh2n.max())
        result = self._scale(fh0n + fh1n + fh2n) 
        # print("HermitePN result", result.min(), result.max())
        return result
        

class BasicBlockHer(nn.Module):
    """BasicBlock with Hermitian activation"""
    expansion = 1

    def __init__(self, in_planes, planes, 
                        stride=1, 
                        affine=False):
                        # activation=F.relu):
        # self.activation = activation

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes, affine=affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(
                            in_planes,
                            self.expansion * planes,
                            kernel_size=1,
                            stride=stride,
                            bias=False)
                            )#, 
        self.herPN1 = HermitePN(planes)
        self.herPN2 = HermitePN(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.herPN1(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.herPN2(out)
        return out

class ResNetHer(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.hermitepn = HermitePN(self.in_planes)
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlockHer, 16, num_blocks[0], 
                         stride=1)
        self.layer2 = self._make_layer(BasicBlockHer, 32, num_blocks[1], 
                            stride=2)
        self.layer3 = self._make_layer(BasicBlockHer, 64, num_blocks[2], 
                            stride=2)
        self.linear = nn.Linear(64 * BasicBlockHer.expansion, num_classes)
        
        self.avgpool = nn.AvgPool2d(8,8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.hermitepn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        
        return F.log_softmax(out, dim=1)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation=F.relu, hermite=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.activation = activation
        # self._hermite = hermite
        # self.hermitepn = HermitePN(self.in_planes)
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 
                         stride=1, activation=self.activation)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 
                            stride=2, activation=self.activation)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 
                            stride=2, activation=self.activation)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        
        self.avgpool = nn.AvgPool2d(8,8)

    def _make_layer(self, block, planes, num_blocks, stride, activation=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation=activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # if self._hermite:
        #     out = self.hermitepn(out) # 3? 16?
        # else:
        out = self.activation(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        
        return F.log_softmax(out, dim=1)


def ResNet20(activation=F.relu):
    return ResNet(BasicBlock, [2, 2, 2], activation=activation)