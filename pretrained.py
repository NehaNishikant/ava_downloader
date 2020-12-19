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

import pickle as pk

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=2000)
        self.fc1 = nn.Linear(2000, 128)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(128, 10)
        self.bn3 = nn.BatchNorm1d(num_features=10)
        self.fc3 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.fc3(x)
        return x
        
def train_model(model, model2, lossfn, optimizer, num_epochs=1):

    epochList = []
    trainLoss = []
    testLoss = []
    trainAcc = []
    testAcc = []

    for epoch in range(num_epochs):
        print(epoch)

        running_loss_train = 0.0
        running_loss_test = 0.0
        running_acc_train = 0
        running_acc_test = 0

        # Iterate over data.
        model2.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data

            optimizer.zero_grad()
            # DCNN forward
            x1 = inputs[:, 0:3, :, :]
            x2 = inputs[:, 3:6, :, :]
            
            output1 = model(x1)
            output2 = model(x2)
            hidden = torch.cat((output1, output2), 1)
            outputs = model2(hidden)
                
            #loss and step
            loss = lossfn(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss_train += loss.item() * inputs.size(0) #times batch size since last might != 32
            running_acc_train += (torch.sum(torch.abs(outputs - labels)).item())

        # print statistics
        trainEpochLoss = running_loss_train / N_train
        trainEpochAcc = running_acc_train / N_train
        print("train l2 loss: ", trainEpochLoss)
        print("train l1 loss: ", trainEpochAcc)
        trainLoss.append(trainEpochLoss)
        trainAcc.append(trainEpochAcc)

        model2.eval()
        for i, data in enumerate(testloader):
            inputs, labels = data

            optimizer.zero_grad()
            # DCNN forward
            x1 = inputs[:, 0:3, :, :]
            x2 = inputs[:, 3:6, :, :]
            output1 = model(x1)
            output2 = model(x2)
            hidden = torch.cat((output1, output2), 1)
            outputs = model2(hidden)
                
            #loss
            loss = lossfn(outputs, labels)
        
            running_loss_test += loss.item() * inputs.size(0) #times batch size since last might != 32
            running_acc_test += (torch.sum(torch.abs(outputs - labels)).item())

        # print statistics
        testEpochLoss = running_loss_test / N_test
        testEpochAcc = running_acc_test / N_test
        print("test l2 loss: ", testEpochLoss)
        print("test l1 loss: ", testEpochAcc)
        testLoss.append(testEpochLoss)
        testAcc.append(testEpochAcc)

    # load best model weights
    return model2, trainLoss, testLoss, trainAcc, testAcc, epochList

model_conv = torchvision.models.vgg16_bn(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

myModel = Net()
lossfn = nn.MSELoss()

N_train = 4305
N_test = 586    
batchsize = 32

trainX1 = torch.load("trainXVGG1.pt")
trainX2 = torch.load("trainXVGG2.pt")
trainX3 = torch.load("trainXVGG3.pt")
shape = list(trainX1.shape)
shape[0] = N_train

trainX = torch.zeros(shape)
idx1 = trainX1.shape[0]
idx2 = trainX1.shape[0] + trainX2.shape[0]
trainX[:idx1, :, :, :] = trainX1
trainX[idx1:idx2, :, :, :] = trainX2
trainX[idx2:, :, :, :] = trainX3

trainY = torch.load("trainYVGG.pt")
trainset = TensorDataset(trainX, trainY)

testX = torch.load("testXVGG.pt")
testY = torch.load("testYVGG.pt")
testset = TensorDataset(testX, testY)

print("got all the data")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer = optim.SGD(myModel.parameters(), lr=0.001, momentum=0.9)

myModel, trainLoss, testLoss, trainAcc, testAcc, epochList = train_model(model_conv, myModel, lossfn, optimizer, num_epochs=25)

#plot
#loss
figL = plt.figure()
plt.plot(epochList, trainLoss, label="train")
plt.plot(epochList, testLoss, label="test")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
figL.suptitle('Loss vs Epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
figL.savefig('L2loss.jpg', bbox_inches='tight')
#accuracy
figA = plt.figure()
plt.plot(epochList, trainAcc, label="train")
plt.plot(epochList, testAcc, label="test")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
figA.suptitle('Accuracy vs Epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
figA.savefig('L1loss.jpg', bbox_inches='tight')
print("train L2 loss: ", trainLoss[len(trainLoss)-1])
print("train L1 loss: ", trainAcc[len(trainAcc)-1])
print("test L2 loss: ", testLoss[len(testLoss)-1])
print("test L1 loss: ", testAcc[len(testAcc)-1])

