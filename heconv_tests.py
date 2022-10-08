from heconv import *
import matplotlib.pyplot as plt 

def plot_anno_heatmap(data):
    nx, ny = data.shape

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # Loop over data dimensions and create text annotations.
    for i in range(nx):
        for j in range(ny):
            text = ax.text(j, i, data[i, j],
                        ha="center", va="center", color="w")

def test_sumslots(N, gap = 2):
    ctxt = np.zeros(N*gap, dtype=int)
    ctxt[::gap] = np.arange(1, N+1)

    result = sumslots(ctxt, N, gap)
    print("input: ", ctxt)
    assert np.all(result[::gap] == np.sum(ctxt))
    print("sumslots test passed!") 
    print(result[:3*gap], "==", np.sum(ctxt))
    

def test_VEC(hi, wi, ci):
    nn = hi*wi*ci
    nslots = 2**ceil(np.log2(nn))
    arr = np.arange(nn)
    swapped = np.swapaxes(arr.reshape(ci,hi,wi),0,1)
    swapped = np.swapaxes(swapped,1,2)
    vectorized = VEC(swapped, nslots)
    assert np.all(vectorized[:nn] == arr)
    print("VEC Test passed")
    print(vectorized)


def test_multiplex():
    ch1 = np.arange(16).reshape(4,4)
    ch2 = 10*np.arange(16).reshape(4,4)
    ch3 = 100*np.arange(16).reshape(4,4)
    #ch4 = np.zeros(16).reshape(4,4)
    tensor_img = np.zeros((4,4,3))
    tensor_img[:,:,0] = ch1
    tensor_img[:,:,1] = ch2
    tensor_img[:,:,2] = ch3
    print(tensor_img.shape)
    print(tensor_img[:,:,0])
    print(tensor_img[:,:,1])
    print(tensor_img[:,:,2])

    MP = multiplex(tensor_img)
    plot_anno_heatmap(MP[:,:,0].astype(int))
    return tensor_img