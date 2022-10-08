import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import torch.nn.functional as F
from approximate import approx_relu, approx_sign
import numpy as np

class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, num_classes, activation=F.relu):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=3, padding="same")
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same")
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = activation
        #self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.pool(out)
        out = self.conv_layer3(out)
        out = self.activation(self.bn1(out))
        out = self.pool(out)
        
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = self.activation(self.bn2(out))
        out = self.pool(out)
        
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        return out


def get_test_model(train=False):
    xfactor = 20
    num_workers = 0
    batch_size = 32
    valid_size = 0.2

    activation = lambda x : xfactor * approx_relu(x/xfactor, degree = 5, repeat=3)
    org_model = ConvNeuralNet(num_classes=10, activation=F.relu) # F.relu가 아니고..??

    if train:
        ## Scale 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        train_data = datasets.CIFAR10('data', train=True,
                                    download=True,
                                    transform=transform
                                    )
        test_data = datasets.CIFAR10('data', train=False,
                                    download=True, 
                                    transform=transform
                                    )

        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
            sampler=valid_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
            num_workers=num_workers)

    return org_model