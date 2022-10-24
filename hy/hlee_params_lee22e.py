import numpy as np
###################################
#  params
###################################

class Parameters_lee22e:
    def __init__(self,ins,outs,kernels,nslots,device,classes):
        self.hi=ins[0]
        self.wi=ins[1]
        self.ci=ins[2]
        self.ki,self.ti=self.get_k_and_t(self.ci)
        self.ho=outs[0]
        self.wo=outs[1]
        self.co=outs[2]
        self.ko,self.to = self.get_k_and_t(self.co)
        self.fh=kernels[0]
        self.fw=kernels[1]
        self.nslots =nslots
        self.device=device
        self.classes=classes
    def get_k_and_t(self,c):
        k = 1
        while(1):
            if c > k**2:
                k = k + 1
            else:
                break
        t = np.ceil(c/k**2)
        return k,int(t)
    def set_input_params(self,hi,wi,ci):
        self.hi=hi
        self.wi=wi
        self.ci=ci
        self.ki,self.ti=self.get_k_and_t(self.ci)
    def set_output_params(self,ho,wo,co):
        self.ho=ho
        self.wo=wo
        self.co=co
        self.ko,self.to=self.get_k_and_t(self.co)
    def set_kernel_params(self,fh,fw):
        self.fh=fh
        self.fw=fw
    def set_nslots(self,nslots):
        self.nslots = nslots
    def set_device(self,device):
        self.device=device
    def get_ins(self):
        return self.hi,self.wi,self.ci
    def get_outs(self):
        return self.ho,self.wo,self.co
    def get_kernels(self):
        return self.fh,self.fw
