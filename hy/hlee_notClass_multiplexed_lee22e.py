"""
2022.10.19
디버깅중임.
i.e. disecting the code here.
hlee_multiplxed_lee22e.py

"""


from hlee_utils import *
################
#  Vec()       #
################
def Vec(mat,nslots):
    hi,wi,ci = np.shape(mat)
    out = np.zeros(nslots)
    for i in range(hi*wi*ci):
        idx_1st = int(np.floor((i%(hi*wi))/wi))
        idx_2nd = i%wi
        idx_3rd = int(np.floor(i/(hi*wi)))
        out[i]=mat[idx_1st][idx_2nd][idx_3rd]
    return out
        

def Vec_np(mat,nslots):
    out = np.zeros(nslots)
    out[:mat.size]=mat.reshape(1,mat.size)[0]
    return out

################
#  MultPack()  #
################

def tensor_multiplexed_input(mat,dims=[]):
    hi,wi,ci, ki,ti,pi = dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]
    out = np.zeros([ki*hi,ki*wi,ti])
    for i3 in range(ki*hi):
        for i4 in range(ki*wi):
            for i5 in range(ti):
                cond = (ki**2)*i5 + ki*(i3%ki) + (i4%ki)
                if cond<ci:
                    idx_1st = int(np.floor(i3/ki))
                    idx_2nd = int(np.floor(i4/ki))
                    idx_3rd = (ki**2)*i5 + ki*(i3%ki) + i4%ki
                    out[i3][i4][i5] = mat[idx_1st][idx_2nd][idx_3rd]
    return out

def MultPack(mat,dims=[],nslots=2**15):
    return Vec(tensor_multiplexed_input(mat,dims),nslots)
################
#  unpack()    #
################
def unpack(ct,dims=[]):
    fh,fw= 3,3
    ha,wa,ca,ka,ta,pa = dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]
    tsize = ha*wa*ka**2
    ch  = []
    for channel in range(ka**2*ta):
        r = channel%(ka**2)
        idx_start_channel = ta*(r%ka)+int(np.floor(r/ka))*ta*ka*wa +int(np.floor(channel/(ka**2)))
        mat=[]
        for i in range(ha):
            row = []
            for j in range(wa):
                idx = idx_start_channel+i*wa*ta*ka**2+j*ka*ta
                row.append(ct[idx])
            mat.append(row)
        ch.append(mat)
    return np.array(ch)
##################
#  SumSlots()    #
##################

def SumSlots(ct_a,m,p):
    nrots = 0
    n = int(np.floor(np.log2(m)))
    ct_b = []
    ct_b.append(ct_a)
    for j in range(1,n+1):
        lrots = int(-1*p*2**(j-1))
        ct_b.append(ct_b[j-1]+np.roll(ct_b[j-1],lrots))
        if lrots!=0:
            nrots=nrots+1  #______________________________ROTATION
    ct_c = ct_b[n]
    for j in range(0,n):
        n1 = np.floor((m/(2**j))%2)
        if n1==1:
            n2 =int(np.floor((m/(2**(j+1)))%2))
            lrots = int(-1*p*2**(j+1))*n2
            ct_c = ct_c + np.roll(ct_b[j],lrots)
            if lrots!=0:
                nrots=nrots+1#____________________________ROTATION
    return ct_c

##################
#  Selecting()   #
##################

def tensor_multiplexed_selecting(ho,wo,co,ko,to,i):
    S = np.zeros([ko*ho, ko*wo,to])
    for i3 in range (ko*ho):
        for i4 in range (ko*wo):
            for i5 in range (to):
                cond = i5*ko**2 + ko*(i3%ko) + (i4%ko)
                if cond==i:
                    S[i3,i4,i5] = 1
    return S
################
#  MultWgt()   #
################

def tensor_multiplexed_shifted_weight(U,i1,i2,i,ins=[]):
    hi,wi,ci,ki,ti = ins[0],ins[1],ins[2],ins[3],ins[4]
    fh,fw= 3,3
    out = np.zeros([hi*ki, wi*ki,ti])
    for i3 in range(hi*ki):
        for i4 in range(wi*ki):
            for i5 in range(ti):
                cond1 = ki**2*i5 + ki*(i3%ki) + i4%ki
                cond2 = np.floor(i3/ki)-(fh-1)/2+i1
                cond3 = np.floor(i4/ki)-(fw-1)/2+i2
                if (cond1 >= ci or cond2 not in range(hi) or cond3 not in range(wi)):
                    out[i3][i4][i5] = 0
                else:
                    out[i3][i4][i5] = U[i1][i2][ki**2*i5+ki*(i3%ki)+i4%ki][i]
    return out

def MultWgt(U,i1,i2,i,ins=[],nslots=2**15):
    out = np.zeros(nslots)
    temp = Vec(tensor_multiplexed_shifted_weight(U,i1,i2,i,ins),nslots)
    out[:temp.size]=temp
    return out


################
#  MultConv()  #
################
def MultConv(ct_a,U,ins=[],outs=[],kernels=[3,3],nslots=2**15):
    hi,wi,ci,ki,ti = ins[0],ins[1],ins[2],ins[3],ins[4]
    ho,wo,co,ko,to = outs[0],outs[1],outs[2],outs[3],outs[4]
    fh,fw= kernels[0],kernels[1]
    print(f"[MultConv] (hi,wi,ci) =({hi},{wi},{ci}),(ho,wo,co)=({ho},{wo},{co}),(fh,fw)=({fh},{fw})")
    print(f"[MultConv] (ki,ti) =({ki},{ti}), (ko,to) =({ko},{to})")
    ct_d = np.zeros(nslots)
    ct = []
    nrots=0
    for i1 in range(fh):
        temp = []
        for i2 in range(fw):
            lrots = int((-(ki**2)*wi*(i1-(fh-1)/2) - ki*(i2-(fw-1)/2))) #both neg in the paper, git -,+
            temp.append(np.roll(ct_a,lrots))
            if lrots!=0:
                nrots = nrots+ 1#____________________________________ROTATION
        ct.append(temp)
    for i in range(co):
        S_mp = tensor_multiplexed_selecting(ho,wo,co,ko,to,i)
        vec_S = Vec(S_mp,nslots)
        ct_b = np.zeros(nslots)
        for i1 in range(fh):
            for i2 in range(fw):
                ct_b = ct_b + ct[i1][i2]*MultWgt(U,i1,i2,i,ins,nslots)
        ct_c = SumSlots(ct_b, ki,              1 )
        ct_c = SumSlots(ct_c, ki,          ki*wi )
        ct_c = SumSlots(ct_c, ti, (ki**2)*hi*wi)
        r1 =  int(np.floor(i/(ko**2)))*ko**2*(ho)*(wo)
        r2 =  int( np.floor((i%(ko**2))/ko))*ko*(wo)
        r3 =  (i%ko)
        rrots = -r1-r2-r3
        rolled =  np.roll(ct_c, -rrots)
        ct_d = ct_d +rolled*vec_S
        if rrots!=0:
            nrots=nrots+1 #_________________________________________ROTATION
    return ct_d





##############################################################################################10.23 parallel
