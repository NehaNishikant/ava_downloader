import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import image

#import h5py
import pickle as pk
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch import Tensor

from sklearn.preprocessing import normalize
from torchvision.utils import save_image
from torchvision import transforms

# attempts to pickle dataset 
# data anaylysis: finds mean diff from avg, historgams

def do(data_type):
    #sequentially read from AVA_dataset/aesthetics_image_lists/fooddrink_<train or test>.jpgl
    fdparsed = r'AVA_dataset/fooddrink_'+data_type+'_parsed.txt'
    savePath = r'fooddrink_imgs_'+data_type+'/'
    f = open(fdparsed, "r")


    valid = 0
    avg = 5.3347913788400545
    if (data_type == "train"):
        valid = 4305
    else:
        valid = 586
    
    size = 224
    
    totaldiff = 0
    
    counter = 0
    X = torch.zeros((valid, 6, size, size))
    Y = torch.zeros((valid, 1))

    for line in f:
        line = line.strip().split(' ')
        imgID = line[0] #imageID

        # load the image
        filename = savePath+imgID+'.jpg'

        # load image as pixel array
        data = image.imread(filename)
        

        #cropped
        #find random top left corner
        H = data.shape[0]
        W = data.shape[1]
        topleftx = int(np.random.rand()*(H-size))
        toplefty = int(np.random.rand()*(W-size))
        cropped = np.matrix.copy(data[topleftx:topleftx+size, toplefty:toplefty+size, :])
        
        #resize
        image_resized = resize(data, (size, size))
        
        #reshape and put together #128x128x3 
        image_resized = np.transpose(image_resized.reshape(size*size, -1)).reshape(-1, size, size) #3x128x128
        cropped = np.transpose(cropped.reshape(size*size, -1)).reshape(-1, size, size) #3x128x128


        #save info
        score = float(line[1])
        Y[counter] = score
        totaldiff += abs(score-avg)
        
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image_resized = torch.from_numpy(image_resized).float()
        image_resized = image_resized/255
        image_resized = transform(image_resized)
        
        cropped = torch.from_numpy(cropped).float()
        cropped = cropped/255
        cropped = transform(cropped)
        
        X[counter, 0:3, :, :] = image_resized
        X[counter, 3:6, :, :] = cropped

        
        
        counter+=1
        print(counter)

    f.close()
    avgdiff = totaldiff/valid
    print("avg diff: ", avgdiff)

    if data_type == "train":
        idx1 = valid//3
        idx2 = 2*valid//3
        X1 = X[:idx1, :, :, :].clone()
        X2 = X[idx1:idx2, :, :, :].clone()
        X3 = X[idx2:, :, :, :].clone()
        
        return (X1, X2, X3, Y)

    return (X, Y)


(trainX1, trainX2, trainX3, trainY) = do("train")
print("train done")

(testX, testY) = do("test")
print("test done")

hist_trainY = torch.histc(trainY, bins = 10, min = 0.0, max = 10.0)
hist_trainY_zoom = torch.histc(trainY, bins = 5, min = 3.0, max = 8.0)
hist_testY = torch.histc(testY, bins = 10, min = 0.0, max = 10.0)
hist_testY_zoom = torch.histc(testY, bins = 5, min = 3.0, max = 8.0)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
ax1.bar(np.linspace(0.0, 10.0, num = 10), hist_trainY.numpy(), color='b')
ax2.bar(np.linspace(3.0, 8.0, num = 5), hist_trainY_zoom.numpy(), color='r')
fig.savefig('trainHistograms.jpg', bbox_inches='tight')

fig2, (ax3, ax4) = plt.subplots(1, 2, sharey = True)
ax3.bar(np.linspace(0.0, 10.0, num = 10), hist_testY.numpy(), color='b')
ax4.bar(np.linspace(3.0, 8.0, num = 5), hist_testY_zoom.numpy(), color='r')
fig2.savefig('testHistograms.jpg', bbox_inches='tight')

torch.save(trainX1, "trainXVGG1.pt")
torch.save(trainX2, "trainXVGG2.pt")
torch.save(trainX3, "trainXVGG3.pt")

torch.save(trainY, "trainYVGG.pt")
torch.save(testX, "testXVGG.pt")
torch.save(testY, "testYVGG.pt")