from hemul.heconv import Multipxed_conv, selecting_tensor
import numpy as np

def avgpool(ctx, wi, hi, ki, ti):
    """Average pool and re-ordering for FC layer"""
    cta = ctx.copy()
    for j in range(np.log2(wi)):
        np.roll(cta, 2**j*ki)

    for j in range(np.log2(hi)):
        np.roll(cta, 2**j*ki**2*wi)

    ctb = np.zeros(len(ctx))
    for i1 in range(ki):
        for i2 in range(ti):
            rot = ki**2*hi*wi*i2 + ki*wi*i1 - ki*(ki*i2+i1)
            ctb += np.roll(cta, -rot) * selecting_tensor(ki*i2+i1)


class HEBasicBlock():

    def __init__(self):
        pass



class HEResNet():

    def __init__(self, tor_model, conf):
        self.nslots = conf["nslots"]
        self.activation = conf["activation"]
        ##
        ## To do: more automatic configuration
        ##
        self.conv1 = Multipxed_conv(tor_model.conv_layer1.weight.detach().numpy(), 
                        32, 32, 1, 1, 1, 8, 2, 2, self.nslots)
        self.conv2a = Multipxed_conv(tor_model.conv_layer2a.weight.detach().numpy(), 
                                32, 32, 1, 1, 1, 2, 2, 8, self.nslots)
        self.conv2b = Multipxed_conv(tor_model.conv_layer2b.weight.detach().numpy(), 
                                32, 32, 1, 1, 1, 2, 2, 8, self.nslots)
        self.conv3a1 = Multipxed_conv(tor_model.conv_layer1.weight.detach().numpy(), 
                                16, 16, 1, 2, 1, 2, 4, 16, self.nslots)
        self.conv3a = Multipxed_conv(tor_model.conv_layer1.weight.detach().numpy(), 
                                16, 16, 1, 1, 1, 4, 4, 8, self.nslots)
        self.conv3b = Multipxed_conv(tor_model.conv_layer1.weight.detach().numpy(), 
                                16, 16, 1, 1, 1, 4, 4, 8, self.nslots)
        self.conv4a1 = Multipxed_conv(tor_model.conv_layer1.weight.detach().numpy(), 
                                16, 8, 1, 1, 1, 8, 2, 2, self.nslots)
        self.conv4a = Multipxed_conv(tor_model.conv_layer1.weight.detach().numpy(), 
                                32, 32, 1, 1, 1, 8, 2, 2, self.nslots)
        self.conv4b = Multipxed_conv(tor_model.conv_layer1.weight.detach().numpy(), 
                                32, 32, 1, 1, 1, 8, 2, 2, self.nslots)
        self.convs1 = Multipxed_conv(tor_model.conv_layer1.weight.detach().numpy(), 
                                32, 32, 1, 1, 1, 8, 2, 2, self.nslots)
        self.convs2 = Multipxed_conv(tor_model.conv_layer1.weight.detach().numpy(), 
                                32, 32, 1, 1, 1, 8, 2, 2, self.nslots)

        self.avgpool = None # depends on conv parameter (ko, to,...)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.maxpool(out)

        out = self.basicblock1(out)

        out = self.activation(self.bn3(self.conv3(out)))
        out = self.maxpool(out)
        out = self.activation(self.bn4(self.conv4(out)))
        out = self.maxpool(out)

        out = self.basicblock2(out)
        
        out = self.avgpool(out)
        out = self.linear(out)
        
        return out 