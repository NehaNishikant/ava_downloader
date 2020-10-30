import numpy as np
import pickle as pk
from PIL import Image
from matplotlib import image

import h5py
import pickle as pk

def do(data_type):
    #sequentially read from AVA_dataset/aesthetics_image_lists/fooddrink_<train or test>.jpgl
    fdparsed = r'AVA_dataset/fooddrink_'+data_type+'_parsed.txt'
    savePath = r'fooddrink_imgs_'+data_type+'/'
    f = open(fdparsed, "r")

    
    #crop/pad everything else to 600x600
    maxT = 600

    valid = 0
    if (data_type == "train"):
        valid = 2449
    else:
        valid = 2442
    
    trainavg = 5.313899933074243
    
    counter = 0
    X = np.zeros((valid, 3, maxT, maxT))
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

        #reshape
        data = np.transpose(data.reshape(H*W, -1)).reshape(-1, H, W)
        #pad
        padSizeH = max(maxT - H, 0)
        padSizeW = max(maxT - W, 0)
        data = np.pad(data, [(0, 0), (padSizeH, 0), (padSizeW, 0)], mode='constant')
        #crop
        data = data[:, 0:maxT, 0:maxT]
        #display the array of pixels as an image
        #pyplot.imshow(data[0])
        #pyplot.show()

        #save info
        score = float(line[1])
        if (score > trainavg):
            Y[counter] = 1
        else:
            Y[counter] = 0
        X[counter] = data
        
        #print(counter)
        counter+=1

    f.close()
    print(counter)
    return (X, Y)


(trainX, trainY) = do("train")
print("train done")
(testX, testY) = do("test")
print("test done")

'''
with h5py.File("fooddrink.hdf5", "a") as f:
    dset1 = f.create_dataset("trainX", data=trainX)
    dset2 = f.create_dataset("trainY", data=trainY)
    dset3 = f.create_dataset("testX", data=testX)
    dset4 = f.create_dataset("testY", data=testY)


with open('fooddrink_trainX.pk','wb') as f:
    for c in trainX:
        pk.dump(trainX, f)
print("pickled trainX")

with open('fooddrink_trainY.pk','wb') as f:
     pk.dump(trainY, f)
print("pickled trainY")

with open('fooddrink_testX.pk','wb') as f:
     pk.dump(testX, f)
print("pickled testX")

with open('fooddrink_testY.pk','wb') as f:
     pk.dump(testY, f)
print("pickled testY")
'''