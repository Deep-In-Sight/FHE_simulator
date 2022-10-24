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

def VEC(tensor_img, nslots):
    """
    Basically... 
    new_arr[:nslots] = tensor_img.ravel()

    But the paper has different axis order than numpy.
    """
    hi, wi, ci = tensor_img.shape
    vectorized = np.zeros(nslots) # or int(2**floor(np.log2(ci))) ?

    for i in range(wi*hi*ci):
        vectorized[i] = tensor_img[floor((i%(hi*wi))/wi),
                                    i % wi, floor(i/(hi*wi))]
    return vectorized


def multiplex(tensor_data):
    """ Flatten input channel dimension into extended H,W dimension
        
        parameter
        ---------
            tensor_data: (H, W, C)
    """
    hi, wi, ci = tensor_data.shape
    ki = ceil(sqrt(ci))
    ti = np.ceil(ci/ki**2).astype(int)

    MP = np.zeros((ki*hi,ki*wi,ti))

    for i5 in range(ti):
        for i4 in range(ki*wi):
            for i3 in range(ki*hi):
                ind_last = ki**2*i5 + ki*(i3 % ki) + i4 % ki
                if ind_last < ci:
                    MP[i3,i4,i5] = tensor_data[floor(i3/ki), floor(i4/ki), ind_last]

    return MP