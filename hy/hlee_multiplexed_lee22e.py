from hlee_utils import *
from hlee_params_lee22e import Parameters_lee22e as dims

class Multiplexed_lee22e(dims):
    def __init__(self,ins,outs,kernels,nslots,device,classes):
        super().__init__(ins,outs,kernels,nslots,device,classes)
        try:
            print("\n\n------------------Parameters--------------------\n")
            print(f"hi,wi,ci (ki,ti) = {self.hi},{self.wi},{self.ci} ({self.ki},{self.ti})")
            print(f"ho,wo,co (ko,to) = {self.ho},{self.wo},{self.co} ({self.ko},{self.to})")
            print(f"fh,fw = {self.fh},{self.fw}")
            print(f"nslots = {self.nslots}")
        except:
            print("Do: Parameters_lee22e() first!!")
        self.nrots = 0

    ################
    #  Vec()       #
    ################

    def Vec(self,A):
        out = np.zeros(self.nslots)
        out[:A.size]=A.reshape(1,A.size)[0]
        return out

    ################
    #  MultPack()  #
    ################

    def tensor_multiplexed_input(self,A):
        ha,wa,ca = A.shape
        ka,ta    = self.get_k_and_t(ca)
        out = np.zeros([ka*ha,ka*wa,ta])
        for i3 in range(ka*ha):
            for i4 in range(ka*wa):
                for i5 in range(int(ta)):
                    cond = (ka**2)*i5 + ka*(i3%ka)+(i4%ka) 
                    if cond<ca:
                        idx_1st = int(np.floor(i3/ka))
                        idx_2nd = int(np.floor(i4/ka))
                        idx_3rd = int((ka**2)*i5+ka*(i3%ka)+i4%ka)
                        out[i3][i4][i5] = A[idx_1st][idx_2nd][idx_3rd]
        return out

    def MultPack(self,A):
        return self.Vec(self.tensor_multiplexed_input(A))

    ################
    #  MultWgt()   #
    ################

    def tensor_multiplexed_shifted_weight(self,U,i1,i2,i):
        hi,wi,ci = self.get_ins()
        ki,ti = self.get_k_and_t(ci)
        fh,fw= self.get_kernels() 
        out = np.zeros([hi*ki, wi*ki,ti])
        for i3 in range(hi*ki):
            for i4 in range(wi*ki):
                for i5 in range(ti):
                    cond1 = i5*ki**2 + ki*(i3%ki) + i4%ki
                    cond2 = np.floor(i3/ki)-(fh-1)/2+i1
                    cond3 = np.floor(i4/ki)-(fw-1)/2+i2
                    if (cond1 >= ci or cond2 not in range(hi) or cond3 not in range(wi)):
                        out[i3][i4][i5] = 0
                    else:                    
                        out[i3][i4][i5] = U[i1][i2][i5*ki**2+ki*(i3%ki)+i4%ki][i]
        return out

    def MultWgt(self,U,i1,i2,i):
        out = np.zeros(self.nslots)
        temp = self.Vec(self.tensor_multiplexed_shifted_weight(U,i1,i2,i))
        out[:temp.size]=temp
        return out

    ################
    #  unpack()    #
    ################
    def unpack(self,ct,ha,wa,ca): 
        ka,ta = self.get_k_and_t(ca)
        #fh,fw= self.get_kernels()
        ch=[]
        for k in range(ka**2):
            mat=[]
            for i in range(ha):
                iflat = i*wa*ka**2
                row = []
                for j in range(wa):
                    jflat = j*ka + k%ka + int(np.floor(k/ka)*ka*wa)
                    row.append(ct[iflat+jflat])
                mat.append(row)
            ch.append(mat) 
        return np.array(ch)
    
    ##################
    #  Selecting()   #
    ##################

    def tensor_multiplexed_selecting(self,ho,wo,co,i):
        ko,to = self.get_k_and_t(co)
        S = np.zeros([ko*ho, ko*wo,to])
        for i3 in range (ko*ho):
            for i4 in range (ko*wo):
                for i5 in range (to):
                    cond = i5*ko**2 + ko*(i3%ko) + (i4%ko)
                    #cond = i5*self.ki**2 + self.ki*(i3%self.ki) + (i4%self.ki)
                    if cond==i:
                        S[i3][i4][i5] = 1
                    #print("************************************i5")
        print(f"*************************************{S[::32,1,0]}")

        return S

    ##################
    #  SumSlots()    #
    ##################

    def SumSlots(self,ct_a,m,p):
        # ckks에서 슬롯갯수는 nslots(N/2) = 2^n이다. ( N = poly_modulus_degree)
        # m : 전체 슬롯중에서 0아닌애들 카운팅 (slot갯수-padded슬롯갯수)
        # p : 0아닌 value나오는 주기 (gap)
        nrots = 0
        n = int(np.floor(np.log2(m)))
        ct_b = []
        ct_b.append(ct_a)
        for j in range(1,n+1):
            lrots = int(-1*p*2**(j-1))
            ct_b.append(ct_b[j-1]+np.roll(ct_b[j-1],lrots))
            if lrots!=0:
                self.nrots=self.nrots+1  #___________________________________________ROTATION
            print(f"[SumSlot][LOOP1] rot [{lrots}] #rotation_count = {self.nrots}")
        ct_c = ct_b[n]
        for j in range(0,n):
            n1 = np.floor((m/(2**j))%2)
            if n1==1:
                n2 =int(np.floor((m/(2**(j+1)))%2))
                lrots = int(-1*p*2**(j+1))*n2
                ct_c = ct_c + np.roll(ct_b[j],lrots)
                if lrots!=0:
                    self.nrots=self.nrots+1#_________________________________________ROTATION
                print(f"[SumSlot][LOOP2] rot [{lrots}] #rotation_count = {self.nrots}")
        return ct_c
    
    ##################
    #  MultConv()    #
    ##################

    def MultConv(self,ct_a,U):
        hi,wi,ci = self.get_ins()
        ki,ti = self.get_k_and_t(ci)
        fh,fw= self.get_kernels()
        ho,wo,co=self.get_outs()
        ko,to=self.get_k_and_t(co)

        print(f"[MultConv] (hi,wi,ci) =({hi},{wi},{ci}),(ho,wo,co)=({ho},{wo},{co}),(fh,fw)=({fh},{fw})")
        print(f"[MultConv] (ki,ti) =({ki},{ti}), (ko,to) =({ko},{to})")
        print("\n\n------------------START MultConv()--------------------\n")
        ct_d = np.zeros(self.nslots)
        ct = []
       
        for i1 in range(fh):
            temp = []
            for i2 in range(fw):
                lrots = int((-(ki**2)*wi*(i1-(fh-1)/2) - ki*(i2-(fw-1)/2))) #both neg in the paper, git -,+
                temp.append(np.roll(ct_a,lrots))
                if lrots!=0:
                    self.nrots = self.nrots+ 1#____________________________________ROTATION
                print(f"[MultConv][(fh,fw) LOOP] rot [{lrots:5}] #rotation_count = {self.nrots}")
            ct.append(temp)
        
        print(f"nrows = {len(ct)}")
        print(f"ncols = {len(ct[0])}")
        print(f"depth = {len(ct[0][0])}")

        for i in range(co):    
            S_mp = self.tensor_multiplexed_selecting(ho,wo,co,i)
            ct_b = np.zeros(self.nslots)
            for i1 in range(fh):
                for i2 in range(fw):
                    ct_b = ct_b + ct[i1][i2]*self.MultWgt(U,i1,i2,i) # 커널에 걸리는 9개 픽셀을 합하는 것

#EXPERIMENT  10.13
            ct_c = self.SumSlots(ct_b, ki,              1 ) # 
            ct_c = self.SumSlots(ct_c, ki,          ki*wi ) # 4개 input 채널을 하나로 합치는 것
            ct_c = self.SumSlots(ct_c, ti, (ki*hi)*(ki*wi))
#EXPERIMENT 10.13

            r1 =  int(np.floor(i/(ko**2))*ko**2*(ho)*(wo)) 
            r2 =  int(np.floor((i%(ko**2))/ko)*ko*(wo))
            r3 =  (i%ko)
            rrots = -(r1+r2+r3)

            rolled =  np.roll(ct_c, -rrots)
            vec_S = self.Vec(S_mp)
            print(rolled[4096:4100])
            print(vec_S[4096:4100])
            ct_d = ct_d +rolled * vec_S
            o=31
            s=o*128
            e=s+128
            print(f"\t\t\t\t\t\t\t______________ {rrots}={r1}+{r2}+{r3}")
            if rrots!=0:
                self.nrots=self.nrots+1 #_________________________________________ROTATION
            print(f"[MultConv][(co) LOOP] rot [{rrots:5}] #rotation_count = {self.nrots}")
        return ct_d,self.nrots


    

