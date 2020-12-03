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

#attempts to pickle dataset

def do(data_type):
    #sequentially read from AVA_dataset/aesthetics_image_lists/fooddrink_<train or test>.jpgl
    fdparsed = r'AVA_dataset/fooddrink_'+data_type+'_parsed2.txt'
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
    X = np.zeros((valid, 6, size, size))
    Y = np.zeros((valid, 1))
    for line in f:
        line = line.strip().split(' ')

        #get info
        imgID0 = line[0] #imageID
        score0 = line[1]
        imgID1 = line[2]
        score1 = line[3]

        # get the image
        filename0 = savePath+imgID0+'.jpg'
        filename1 = savePath+imgID1+'.jpg'

        # load image as pixel array
        data0 = image.imread(filename0)
        data1 = image.imread(filename1)

        ''' H = data.shape[0]
        W = data.shape[1]
        print(H)
        print(W)'''
        #resize
        data0 = resize(data0, (size, size))
        data1 = resize(data1, (size, size))

        '''print(image_resized.shape)
        plt.imshow(image_resized)
        plt.show()'''

        #reshape
        data0 = np.transpose(data0.reshape(size*size, -1)).reshape(-1, size, size)
        data1 = np.transpose(data1.reshape(size*size, -1)).reshape(-1, size, size)

        #save and interpret info
        rand = np.random.rand()
        if (rand > 0.5):
            temp = data0
            data0 = data1
            data1 = temp
            Y[counter] = 0
        else:
            Y[counter] = 1
            
        X[counter][0:3] = data0
        X[counter][3:6] = data1
        
        print(counter)
        counter+=1

    f.close()
    
    return (X, Y)


(trainX, trainY) = do("train")
print("train done")
(testX, testY) = do("test")
print("test done")

trainX = torch.from_numpy(trainX).float()
trainY = torch.from_numpy(trainY).float()


testX = torch.from_numpy(testX).float()
testY = torch.from_numpy(testY).float()


torch.save(trainX, "trainX2.pt")
torch.save(trainY, "trainY2.pt")
torch.save(testX, "testX2.pt")
torch.save(testY, "testY2.pt")
