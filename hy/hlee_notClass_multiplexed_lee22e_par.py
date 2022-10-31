from hlee_notClass_multiplexed_lee22e import *
from icecream import ic
from hlee_utils import *
from hlee_notClass_multiplexed_lee22e import *

##################################
#          MultParPack()         #
##################################

def MultParPack(A,dims=[],nslots=2**15):
    ha,wa,ca,ka,ta,pa = dims[0],dims[1],dims[2],dims[3],dims[4],dims[5]
    A_mp = MultPack(A,dims,nslots)
    out = np.zeros(nslots)
    if pa !=int(pa):
        print(f"p is not an integer!!!!! p={pa}")
        return
    else:
        pa = int(pa)
        for i in range(0,pa):
            rot =-i*int(np.round((nslots/pa)))
            out = out+ np.roll(A_mp,rot)
        return out


##################################
#          ParMultWgt()          #
##################################

def tensor_multiplexed_shifted_weight_par(U,i1,i2,i3,ins=[],outs=[]):
    fh,fw= 3,3
    hi,wi,ci,ki,ti,pi =  ins[0], ins[1], ins[2], ins[3], ins[4], int(ins[5])
    ho,wo,co,ko,to,po = outs[0],outs[1],outs[2],outs[3],outs[4],int(outs[5])
    out = np.zeros([hi*ki, wi*ki,ti*pi])
    for i5 in range(hi*ki):
        for i6 in range(wi*ki):
            for i7 in range(ti*pi):
                cond0 = np.floor(i7/ti)+pi*i3
                cond1 = ki**2*(i7%ti) + ki*(i5%ki) + (i6%ki)
                cond2 = np.floor(i5/ki)-(fh-1)/2+i1
                cond3 = np.floor(i6/ki)-(fw-1)/2+i2
                if (cond0 >= co or cond1 >= ci or
                    cond2 not in range(hi) or cond3 not in range(wi)):
                    out[i5][i6][i7] = 0
                else:
                    idx_3rd = ki**2*(i7%ti)+ki*(i5%ki)+i6%ki
                    idx_4th = int(np.floor(i7/ti)+pi*i3)
                    out[i5][i6][i7] = U[i1][i2][idx_3rd][idx_4th]
    return out

def ParMultWgt(U,i1,i2,i3,ins=[],outs=[],nslots=2**15):
    out = np.zeros(nslots)
    temp = Vec(tensor_multiplexed_shifted_weight_par(U,i1,i2,i3,ins,outs),nslots)
    out[:temp.size]=temp
    return out



##################################
#          MulParConv()          #
##################################

def MultParConv(ct_a,U,ins=[],outs=[],kernels=[3,3],nslots=2**15):
    hi,wi,ci,ki,ti,pi = ins[0],ins[1],ins[2],ins[3],ins[4],ins[5]
    ho,wo,co,ko,to,po = outs[0],outs[1],outs[2],outs[3],outs[4],outs[5]
    q = get_q(co,pi)
    fh,fw= kernels[0],kernels[1]
    print(f"[MultParConv] (hi,wi,ci,ki,ti,pi) =({hi:2},{wi:2},{ci:2},{ki:2},{ti:2}, {pi:2})")
    print(f"[MultParConv] (ho,wo,co,ko,to,po) =({ho:2},{wo:2},{co:2},{ko:2},{to:2}, {po:2})")
    print(f"[MultParConv] q = {q}")

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
    for i3 in range(int(q)):
        ct_b = np.zeros(nslots)
        for i1 in range(fh):
            for i2 in range(fw):
                ct_b = ct_b + ct[i1][i2]*ParMultWgt(U,i1,i2,i3,ins,outs,nslots)
        ct_c = SumSlots(ct_b, ki,              1 )
        ct_c = SumSlots(ct_c, ki,          ki*wi )
        ct_c = SumSlots(ct_c, ti, (ki**2)*hi*wi)
        #-----diff
        pi = int(pi)
        
        for i4 in range(0,min(pi,co-pi*i3)):
            i = pi*i3 +i4
            S_mp = tensor_multiplexed_selecting(ho,wo,co,ko,to,i)
            vec_S = Vec(S_mp,nslots)
            r0 = int(np.floor(nslots/pi))*(i%pi)
            r1 = int(np.floor(i/(ko**2)))*ko**2*ho*wo
            r2 = int(np.floor((i%(ko**2))/ko))*ko*wo
            r3 = i%ko
            rrots = +r0-r1-r2-r3
            rolled =  np.roll(ct_c, -rrots)
            ct_d = ct_d +rolled*vec_S
            if rrots!=0:
                nrots=nrots+1 #_________________________________________ROTATION
    for j in range(int(np.round(np.log2(po)))):
        r = int(np.round(2**j*(nslots/po)))
        ct_d = ct_d + np.roll(ct_d,-r)
    return ct_d

