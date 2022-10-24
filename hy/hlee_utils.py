import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image
import cv2
#############
#   Utils   #
#############

def imshow_with_value(data):
    if (len(data.shape))==3:
        h,w,c = data.shape
        channel = {'R':0,'G':1,'B':2}
    else:
        h,w = data.shape
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    # Loop over data dimensions and create text annotations.
    for i in range(h):
        for j in range(w):
            if (len(data.shape)==3):
                text = ax.text(j, i, data[i, j, channel['R']], ha="center", va="center", color="w")
            else:
                text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")  
                
def calculate_nrots(fh,fw,co,ki,ti):
    return fh*fw-1 + co*(2*np.floor(np.log2(ki)) +np.floor(np.log2(ti))+ 1) 
def get_k_and_t(c):
    k = 1
    while(1):
        if c > k**2:
            k = k + 1
        else:
            break
    t = np.ceil(c/k**2)
    return k,int(t)

def print_2d(M):
    h,w = M.shape
    for i in range(h):
        line = ""
        for j in range(w):
            line = line+str(M[i][j])+" "
        print (line)
        
def print_obj(A,channel_last=True):
    ndim = len(A.shape)
    if ndim==2:
        print_2d(A)
    elif ndim==3:
        y,x,z = A.shape
        dtype = A.dtype
        out = np.zeros([z,y,x])
        for k in range(z):
            for i in range(y):
                for j in range(x):
                    out[k][i][j] = A[i][j][k]
        for i in range(z):
            print_2d(out[i])
            print("-------------------------------")


def get_channel_first(U):
    dim=U.shape
    if len(dim)==4:
        h,w,ci,co = U.shape
        out = np.zeros([co,ci,h,w])
        for d in range(co):
            for z in range(ci):
                for i in range(h):
                    for j in range(w):
                        out[d,z,i,j]=U[i,j,z,d]
    elif len(dim)==3:
        h,w,c = U.shape
        out = np.zeros([c,h,w])
        for z in range(c):
            for i in range(h):
                for j in range(w):
                    out[z,i,j] = U[i,j,z]
    return out

def get_channel_last(U):
    dim=U.shape
    if len(dim)==4:
        co,ci,h,w = U.shape
        out = np.zeros([h,w,ci,co])
        for d in range(co):
            for z in range(ci):
                for i in range(h):
                    for j in range(w):
                        out[i,j,z,d]=U[d,z,i,j]
    elif len(dim)==3:
        c,h,w = U.shape
        out = np.zeros([h,w,c])
        for z in range(c):
            for i in range(h):
                for j in range(w):
                    out[i,j,z] = U[z,i,j]
    return out


################
#  unpack()    #
################
def unpack(ct,ha,wa,ca):
    ka,ta = get_k_and_t(ca)
    print(ka,ta)
    fh,fw= 3,3
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
def pack(A):
    A = get_channel_last(A)
    ha,wa,ca = A.shape
    ka,ta    = get_k_and_t(ca)
    print(f"----------------{ha},{wa},{ca}")
    print(f"----------------{ka},{ta}")
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
