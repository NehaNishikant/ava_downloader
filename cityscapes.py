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
import h5py

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
        # 3 x 600 x 600
        self.conv1 = nn.Conv2d(3, 6, 10, stride=6, padding=2) # 6 x 100 x 100
        self.pool1 = nn.MaxPool2d(2, 2) # 6 x 50 x 50
        self.conv2 = nn.Conv2d(6, 16, 6, stride=2, padding=3) # 16 x 26 x 26
        self.pool2 = nn.MaxPool2d(2, 2) # 16 x 13 x 13
        self.fc1 = nn.Linear(16*13*13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*13*13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def do(data_type):
    #sequentially read from AVA_dataset/aesthetics_image_lists/fooddrink_<train or test>.jpgl
    fdparsed = r'AVA_dataset/fooddrink_'+data_type+'_parsed.txt'
    savePath = r'fooddrink_imgs_'+data_type+'/'
    f = open(fdparsed, "r")

    #crop/pad everything else to 600x600
    maxT = 600
    #2469 valid images
    valid = 0
    if (data_type == "train"):
        valid = N_train
    else:
        valid = N_test
    
    trainavg = 5.313899933074243
        
    counter = 0
    X = np.zeros((valid, 3, maxT, maxT))
    Y = np.zeros((valid, 1))
    for line in f:
        line = line.strip().split(' ')
        imgID = line[0] #imageID

        filename = savePath+imgID+'.jpg'
        # load image as pixel array
        data = image.imread(filename)

        #reshape
        H = data.shape[0]
        W = data.shape[1]
        data = np.transpose(data.reshape(H*W, -1)).reshape(-1, H, W)
        #pad
        padSizeH = max(maxT - H, 0)
        padSizeW = max(maxT - W, 0)
        data = np.pad(data, [(0, 0), (padSizeH, 0), (padSizeW, 0)], mode='constant')
        #crop
        data = data[:, 0:maxT, 0:maxT]

        #save info
        score = float(line[1])
        if (score > trainavg):
            Y[counter] = 1
        else:
            Y[counter] = 0
        X[counter] = data
        
        counter+=1

    f.close()
    return (X, Y)
    
def getData():
    print("getting data")
    f = h5py.File('fooddrink.hdf5','r')
    testY = np.array(f["testY"])
    testX = np.array(f["testX"])
    trainY = np.array(f["trainY"])
    trainX = np.array(f["trainX"])
    return ((trainX, trainY), (testX, testY))

#all training stuff
#setup
batchsize = 32
net = Net()
  
N_train = 2449
N_test = 2442   

#((trainX, trainY), (testX, testY)) = getData()

(trainX, trainY) = do("train")
(testX, testY) = do("test")

print("got all the data")
trainX = torch.from_numpy(trainX).float()
trainY = torch.from_numpy(trainY).long()
trainset = TensorDataset(trainX, trainY)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=0)

print("trainloader done")

testX = torch.from_numpy(testX).float()
testY = torch.from_numpy(testY).long()
testset = TensorDataset(testX, testY)
print("testset done")

testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)

print("testloader done")

criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
        
#actual training loop 
epochs = 100 
epochList = []
trainLoss = []
testLoss = []
trainAcc = []
testAcc = []
for epoch in range(100):  # loop over the dataset multiple times
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
        loss = criterion(outputs, labels[:, 0])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss_train += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        y = labels[:, 0]
        running_acc_train += (predicted == y).sum().item()
        #print(running_acc_train)
    
    trainEpochLoss = running_loss_train / (N_train/32)
    trainEpochAcc = running_acc_train / N_train
    trainLoss.append(trainEpochLoss)
    trainAcc.append(trainEpochAcc)
    f.write(str(trainEpochLoss))
    f2.write(str(trainEpochAcc))


    for i, data in enumerate(testloader):
        #print(i)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion2(outputs, labels[:, 0])
        running_loss_test += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        y = labels[:, 0]
        running_acc_test += (predicted == y).sum().item()
    
    testEpochLoss = running_loss_test / (N_test/32)
    testEpochAcc = running_acc_test / N_test
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