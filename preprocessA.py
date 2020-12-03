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

#makes histogram (among other things)

def do(data_type):
    #sequentially read from AVA_dataset/aesthetics_image_lists/fooddrink_<train or test>.jpgl
    fdparsed = r'AVA_dataset/fooddrink_'+data_type+'_parsed.txt'
    savePath = r'fooddrink_imgs_'+data_type+'/'
    f = open(fdparsed, "r")


    valid = 0
    if (data_type == "train"):
        valid = 2449
    else:
        valid = 2442
    
    #trainavg = 5.313899933074243
    size = 128
    
    counter = 0
    X = np.zeros((valid, 3, size, size))
    Y = np.zeros((valid, 1))
    for line in f:
        line = line.strip().split(' ')
        imgID = line[0] #imageID

        # load the image
        filename = savePath+imgID+'.jpg'
        #im = Image.open(filename)
        #print(im.format)
        #print(im.mode)
        #print(im.size)
        # show the image
        #im.show()

        # load image as pixel array
        data = image.imread(filename)

        H = data.shape[0]
        W = data.shape[1]
        '''print(H)
        print(W)'''
        #image_resized = resize(data, (data.shape[0] // (H/100), data.shape[1] // (W/100)), anti_aliasing=True)
        data = resize(data, (size, size))

        '''print(image_resized.shape)
        plt.imshow(image_resized)
        plt.show()'''

        #reshape
        data = np.transpose(data.reshape(size*size, -1)).reshape(-1, size, size)
        #pad
        # padSizeH = max(maxT - H, 0)
        # padSizeW = max(maxT - W, 0)
        # data = np.pad(data, [(0, 0), (padSizeH, 0), (padSizeW, 0)], mode='constant')
        #crop
        # data = data[:, 0:maxT, 0:maxT]
        #display the array of pixels as an image
        #pyplot.imshow(data[0])
        #pyplot.show()
        
        # image_downscaled = downscale_local_mean(image, int(H/224), int(W/224)))


        #save info
        score = float(line[1])
        '''if (score > trainavg):
            Y[counter] = 1
        else:
            Y[counter] = 0'''
        Y[counter] = score
        X[counter] = data
        
        print(counter)
        counter+=1

    f.close()
    
    return (X, Y)


(trainX, trainY) = do("train")
print("train done")
(testX, testY) = do("test")
print("test done")

#make histogram
plt.hist(trainY, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.savefig("trainYHist.jpg")


plt.figure()
plt.hist(trainY, [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5])
plt.savefig("trainYHistZoom.jpg")

plt.figure()
plt.hist(testY, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.savefig("testYHist.jpg")

plt.figure()
plt.hist(testY, [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5])
plt.savefig("testYHistZoom.jpg")

trainX = torch.from_numpy(trainX).float()
trainY = torch.from_numpy(trainY).float()


testX = torch.from_numpy(testX).float()
testY = torch.from_numpy(testY).float()


'''torch.save(trainX, "trainX.pt")
torch.save(trainY, "trainY.pt")
torch.save(testX, "testX.pt")
torch.save(testY, "testY.pt")'''
