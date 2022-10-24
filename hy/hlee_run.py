from hlee_cnn import *
from matplotlib import pyplot as plt
from hlee_utils import *
from hlee_multiplexed_lee22e import Multiplexed_lee22e as multiplexed


class CompareConv:
    def __init__(self,fimage,fparam,kernel_size ,co):
        self.fimage = fimage#"./cute.jpg"
        self.fparam = fparam#"./models/simple_model_hlee.pt"
        self.device = 'cpu'
        self.kernel_size = kernel_size
        #self.nslots = 2**16
        ko,to = get_k_and_t(co)
        self.nslots = 32*32*ko*ko#****************************************
        self.hi,self.wi,self.ci=32,32,3
        self.ho,self.wo,self.co=32,32,co#***************************
        #self.ho,self.wo,self.co=32,32,4
        self.fh,self.fw = kernel_size,kernel_size
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog',      'frog',       'horse','ship','truck']
        self.U_multconv=None
        self.torout = None
        self.mpout=None
        self.tot_rots=0
        self.run_torchconv()
        self.run_multconv()

    def run_torchconv(self): 
        print("run torchconv")
        print(self.co)
        tor = TorchConv_infer( num_classes=len(self.classes),
                               activation=F.relu,
                               fn_param=self.fparam,
                               device=self.device,
                               co=self.co)
        self.torout = tor.TorchConv(self.fimage)
        self.U_multconv = tor.U_multconv #with channel last.
    def run_multconv(self):
        mp = multiplexed([self.hi,self.wi,self.ci],
                         [self.ho,self.wo,self.co],
                         [self.fh,self.fw],
                         self.nslots,
                         self.device,self.classes)
        print(f"U_mpconv : {self.U_multconv.shape}")
        #check = get_channel_first(self.U_multconv)
        #print("****************************check")
        #print(check.shape)
        #print(check[8])
        img = cv2.imread(self.fimage)
        img = cv2.resize(img,(self.hi,self.wi))
        print("#######################")
        print(f"image dims = {img.shape}")
        test = mp.tensor_multiplexed_input(img)
        print(f"image dims = {test.shape}")
        test = get_channel_last(test)
        print(test[2])
        print(test[3:])

        mpout,self.tot_rots = mp.MultConv(mp.MultPack(img),self.U_multconv)
        #self.mpout = mp.unpack(mpout,self.ho,self.wo,self.co)
        self.mpout = mp.unpack(mpout,self.ho,self.wo,self.co)
    
#    def compare(self):
#        stack = np.hstack([self.torout[0],self.mpout[0]])
#        cv2.imwrite("out_comp.png",stack)
#        ko,to =get_k_and_t(self.co)
#        for i in range(1,self.co):
#            stack = np.vstack([stack,np.hstack([self.torout[i],self.mpout[i]])])    
#        return stack,self.tot_rots

    def compare(self):
        return self.torout,self.mpout,self.tot_rots


    def MultPack(self,A):
        mp = multiplexed([self.hi,self.wi,self.ci],
                         [self.ho,self.wo,self.co],
                         [self.fh,self.fw],
                         self.nslots,
                         self.device,self.classes)
        return mp.MultPack(A)
    def unpack(self,A,ha,wa,ca):
        mp = multiplexed([self.hi,self.wi,self.ci],
                         [self.ho,self.wo,self.co],
                         [self.fh,self.fw],
                         self.nslots,
                         self.device,self.classes)
        return mp.unpack(A,ha,wa,ca)
    
        
