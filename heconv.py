import numpy as np
from math import floor, ceil, sqrt

def sumslots(ctxt, m, p):
    # m: number of added slots
    # p: gap size
    ctxt_j =[ctxt]
    for j in range(1, floor(np.log2(m))+1):
        ctxt_j.append(ctxt_j[-1] + np.roll(ctxt_j[-1], 2**(j-1)*p)) # -?
        
    result = ctxt_j[-1].copy()
    for j in range(floor(np.log2(m))):
        if floor(m/2**j) % 2 == 1:
            result += np.roll(ctxt_j[j], floor(m/2**(j+1))*2**(j+1)*p)
    return result

def swap(tensor):
    """(ci,hi,wi) to (hi,wi,ci)"""
    if tensor.ndim == 3:
        swapped = tensor.transpose(1,2,0)
    elif tensor.ndim == 4:
        swapped = np.swapaxes(tensor,0,3)
        swapped = np.swapaxes(swapped,1,2)
    return swapped


def VEC(A, nslots):
    hi, wi, ci = A.shape
    vectorized = np.zeros(nslots) # or int(2**floor(np.log2(ci))) ?
    out = np.zeros(nslots)
    out[:A.size]=A.reshape(1,A.size)[0]
    return out

# def VEC(tensor_img, nslots):
#     """
#     Basically... 
#     new_arr[:nslots] = tensor_img.ravel()

#     But the paper has different axis order than numpy.
#     """
#     hi, wi, ci = tensor_img.shape
#     vectorized = np.zeros(nslots) # or int(2**floor(np.log2(ci))) ?

#     for i in range(wi*hi*ci):
#         vectorized[i] = tensor_img[floor((i%(hi*wi))/wi),
#                                     i % wi, floor(i/(hi*wi))]
#     return vectorized


def multiplex(tensor_data):
    """ Flatten input channel dimension into extended H,W dimension
        
        parameter
        ---------
            tensor_data: (H, W, C)
    """
    hi, wi, ci = tensor_data.shape
    ki = ceil(sqrt(ci))
    ti = ceil(ci/ki**2)

    MP = np.zeros((ki*hi,ki*wi,ti))

    for i3 in range(ki*hi):
        for i4 in range(ki*wi):
            for i5 in range(ti):        
                ind_last = ki**2*i5 + ki*(i3 % ki) + (i4 % ki)
                if ind_last < ci:
                    MP[i3,i4,i5] = tensor_data[floor(i3/ki), floor(i4/ki), ind_last]

    return MP


def get_Up(Uout, Uin, i1, i2, i, ki, fh, fw, hi, wi, ci):
    """Get an intermediate multiplexed representation of (fh,fw,ci,co)-shaped kernel.
        This will be VEC-ed and SIMD-multiplied with the input image
    """
    range_h = np.arange(hi)
    range_w = np.arange(wi)
    n3, n4, n5 = Uout.shape
    #print("n3,n4,n5", n3,n4,n5)
    for i3 in range(n3):
        for i4 in range(n4):
            for i5 in range(n5):
                if  ki**2*i5+ki*(i3 % ki) + (i4 % ki) >= ci or \
                    not floor(i3 / ki) - int((fh -1)/2) +i1 in range_h or \
                    not floor(i4 / ki) - int((fw -1)/2) +i2 in range_w:
                    pass
                    #Uout[i3,i4,i5] = 0
                else:
                    #print("!!!")
                    #print(i1, i2, ki**2*i5+ki*(i3 % ki) + (i4 % ki), i)
                    Uout[i3,i4,i5] = Uin[i1,i2, ki**2*i5+ki*(i3 % ki) + (i4 % ki), i]

## Selecting 
def selecting_tensor(ko, ho, wo, to, i):
    """Gather multiplexed conv output into original form (Am I right??) 
    """
    S = np.zeros((ko*ho,ko*wo,to))
    n3, n4, n5 = S.shape
    for i3 in range(n3):
        for i4 in range(n4):
            for i5 in range(n5):
                if ko**2*i5 + ko*(i3 % ko) + (i4 % ko) == i:
                    S[i3,i4,i5] = 1

    return S