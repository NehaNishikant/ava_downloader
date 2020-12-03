from PIL import Image
import sys
import os
from matplotlib import image
from matplotlib import pyplot as plt
import scipy.misc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch import Tensor
#import h5py

import pickle as pk

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


trainloss_txt = 'trainloss.txt'
trainacc_txt = 'trainacc.txt'
testloss_txt = 'testloss.txt'
testacc_txt = 'testacc.txt'

f = open(trainloss_txt, "a")
f2 = open(trainacc_txt, "a")
f3 = open(testloss_txt, "a")
f4 = open(testacc_txt, "a")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #VGG GANG
        # 3 x 128 x 128
        self.conv1a = nn.Conv2d(3, 16, 3, stride=1, padding=1) # 16 x 128 x 128
        self.conv1b = nn.Conv2d(16, 16, 3, stride=1, padding=1) # 16 x 128 x 128
        self.conv1c = nn.Conv2d(16, 16, 3, stride=1, padding=1) # 16 x 128 x 128
        self.pool1 = nn.MaxPool2d(2, 2) # 16 x 64 x 64

        self.conv2a = nn.Conv2d(16, 32, 3, stride=1, padding=1) # 32 x 64 x 64
        self.conv2b = nn.Conv2d(32, 32, 3, stride=1, padding=1) # 32 x 64 x 64
        self.conv2c = nn.Conv2d(32, 32, 3, stride=1, padding=1) # 32 x 64 x 64
        self.pool2 = nn.MaxPool2d(2, 2) # 32 x 32 x 32

        self.conv3a = nn.Conv2d(32, 64, 3, stride=1, padding=1) # 64 x 16 x 16
        self.conv3b = nn.Conv2d(64, 64, 3, stride=1, padding=1) # 64 x 16 x 16
        self.conv3c = nn.Conv2d(64, 64, 3, stride=1, padding=1) # 64 x 16 x 16
        self.pool3 = nn.MaxPool2d(2, 2) # 64 x 16 x 16

        self.fc1 = nn.Linear(64*16*16, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(F.relu(self.conv1c(x)))
        #print("conv1 done")
        
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(F.relu(self.conv2c(x)))
        #print("conv2 done")

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(F.relu(self.conv3c(x)))
        #print("conv3 done")

        x = x.view(-1, 64*16*16)
        x = F.relu(self.fc1(x))
        #print("fc1 done")
        x = torch.sigmoid(self.fc2(x)) * 10 
        #print("fc2 done")
        #x = self.fc2(x)
        return x

  

#all training stuff
#setup
batchsize = 32
net = Net()
  
N_train = 2449
N_test = 2442   

trainX = torch.load("trainX.pt")
trainY = torch.load("trainY.pt")
trainset = TensorDataset(trainX, trainY)

testX = torch.load("testX.pt")
testY = torch.load("testY.pt")
testset = TensorDataset(testX, testY)

print("got all the data")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

print("loaders done")

lossfn = nn.MSELoss()
lossfn2 = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
        
#actual training loop 
epochs = 100
epochList = []
trainLoss = []
testLoss = []
trainAcc = []
testAcc = []
for epoch in range(epochs):  # loop over the dataset multiple times
    print(epoch)
    epochList.append(epoch)
    #print("running an epoch")
    running_loss_train = 0.0
    running_loss_test = 0.0
    running_acc_train = 0
    running_acc_test = 0
    #print(trainloader)
    
    for i, data in enumerate(trainloader):
        #print(i)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #print("inputs: ",inputs.shape)
        #print("labels: ",labels[:, 0])
        #print("running a batch bb!!!!")
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        #print("outputs: ", outputs.shape)
        loss = lossfn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss_train += loss.item()
        running_acc_train += torch.sum(torch.abs(outputs - labels)).item()

        #check difference between true and predicted labels for batch 2
        if i==75:
            print("TRAIN")
            print("EPOCH: ", epoch)
            print("true: ", labels)
            print("PREDICTED: ", outputs)
    
    trainEpochLoss = running_loss_train / (N_train//batchsize)
    trainEpochAcc = running_acc_train / N_train
    print("train loss: ", trainEpochLoss)
    print("train acc: ", trainEpochAcc)
    trainLoss.append(trainEpochLoss)
    trainAcc.append(trainEpochAcc)
    f.write(str(trainEpochLoss))
    f2.write(str(trainEpochAcc))


    for i, data in enumerate(testloader):
        #print(i)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        outputs = net(inputs)
        loss = lossfn2(outputs, labels)
        
        running_loss_test += loss.item()
        running_acc_test += torch.sum(torch.abs(outputs - labels)).item()

        #check difference between true and predicted labels for batch 2
        '''if i==2:
            print("\nTEST")
            print("EPOCH")
            print(epoch)
            print("INPUT")
            print(inputs)
            print("PREDICTED")
            print(outputs)'''
    
    testEpochLoss = running_loss_test / (N_test//batchsize)
    testEpochAcc = running_acc_test / N_test
    print("test loss: ", testEpochLoss)
    print("test acc: ", testEpochAcc)
    testLoss.append(testEpochLoss)
    testAcc.append(testEpochAcc)
    f3.write(str(testEpochLoss))
    f4.write(str(testEpochAcc))


print('Finished Training')
torch.save(net, 'net.pk')

#plot
#loss
figL = plt.figure()
plt.plot(epochList, trainLoss, label="train")
plt.plot(epochList, testLoss, label="test")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
figL.suptitle('Loss vs Epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
figL.savefig('loss.jpg', bbox_inches='tight')
#accuracy
figA = plt.figure()
plt.plot(epochList, trainAcc, label="train")
plt.plot(epochList, testAcc, label="test")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
figA.suptitle('Accuracy vs Epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
figA.savefig('accuracy.jpg', bbox_inches='tight')
print("train loss: ", trainLoss[len(trainLoss)-1])
print("test loss: ", testLoss[len(testLoss)-1])
print("train acc: ", trainAcc[len(trainAcc)-1])
print("test acc: ", testAcc[len(testAcc)-1])