import torch
from heconv import *
import matplotlib.pyplot as plt 
from PIL import Image
import torchvision.transforms as transforms


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
    swapped = arr.reshape(ci,hi,wi).transpose(1,2,0)
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


def load_img(fn):
    """Load an image for test"""
    img = np.array(Image.open(fn))
    to_tensor = transforms.ToTensor() # [n_channel, nh, nw]
    img_tensor = to_tensor(img).unsqueeze(0) # [n_batch, n_channel, nh, nw]
    return img_tensor
    #n_batch, n_channel, nh, nw = img_tensor.shape

def load_dict(model, fn_param):
    # Load trained parameters
    trained_param = torch.load(fn_param, map_location=torch.device("cpu"))#device))
    trained_param = {k: v.cpu() for k, v in trained_param.items()} # to cpu()
    model.load_state_dict(trained_param)

def compare_conv1(org_model, f_conv, img_tensor, ch=0):

    org_result = org_model.conv_layer1(img_tensor)

    org2d = org_result[0,ch,...].detach().numpy()

    test_result = f_conv(img_tensor[0,ch,:,:].detach().numpy()) 

    fig, axs = plt.subplots(2,2, figsize=(10,10))
    axs[0,0].imshow(img_tensor[0,ch,...].detach().numpy())
    axs[0,0].set_title("Input")

    axs[1,0].imshow(org2d)
    axs[1,0].set_title("Torch Output")

    axs[1,1].imshow(test_result)
    axs[1,1].set_title("FHE Output")

    axs[0,1].imshow((test_result - org2d)/org2d.ptp(), vmin=-1, vmax=1)
    axs[0,1].set_title("Diff")