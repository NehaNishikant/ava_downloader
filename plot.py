from matplotlib import pyplot as plt

def plot(filename):
    #read info
    f = open(filename, "r")
    counter = 0
    
    epochList = []
    trainL2 = []
    trainL1 = []
    testL2 = []
    testL1 = []
    losses = [trainL2, trainL1, testL2, testL1]
    minTestL1 = -1
    matchingTrainL1 = 0
    currTrainL1 = 0
    matchingEpoch = 0
    currEpoch = 0
    for line in f:
        if (counter % 5 == 0):
            #print(line)
            epochList.append(int(line))
            currEpoch = int(line)
        else:
            line = line.strip().split(' ')
            #print(line)
            loss = float(line[4])
            losses[(counter % 5) -1].append(loss)
            if (counter % 5 == 2): #train L1
                currTrainL1 = loss
            if (counter % 5 == 4): #test L1
                if (loss < minTestL1) or (minTestL1 < 0):
                    minTestL1 = loss
                    matchingTrainL1 = currTrainL1
                    matchingEpoch = currEpoch
        counter +=1
    
    #plot
    #l2
    figL = plt.figure()
    plt.plot(epochList, trainL2, label="train")
    plt.plot(epochList, testL2, label="test")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    figL.suptitle('L2 vs Epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    figL.savefig('L2loss_pretrain2.jpg', bbox_inches='tight')
    #l1
    figA = plt.figure()
    plt.plot(epochList, trainL1, label="train")
    plt.plot(epochList, testL1, label="test")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    figA.suptitle('L1 vs Epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    figA.savefig('L1loss_pretrain2.jpg', bbox_inches='tight')
    
    #test only
    figL = plt.figure()
    plt.plot(epochList, testL2, label="test")
    figL.suptitle('L2 vs Epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    figL.savefig('L2loss_pretrain2_test.jpg', bbox_inches='tight')
    
    
    figA = plt.figure()
    plt.plot(epochList, testL1, label="test")
    figA.suptitle('L1 vs Epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    figA.savefig('L1loss_pretrain2_test.jpg', bbox_inches='tight')
    
    print("train L2 loss: ", trainL2[len(trainL2)-1])
    print("train L1 loss: ", trainL1[len(trainL1)-1])
    print("test L2 loss: ", testL2[len(testL2)-1])
    print("test L1 loss: ", testL1[len(testL1)-1])
    
    print("min test L1 loss: ", minTestL1)
    print("epoch: ", matchingEpoch)
    print("corresponding train L1 loss: ", matchingTrainL1)
  

#filename = 'pretrainLosses.txt'  
filename = 'pretrainLosses2.txt'
plot(filename)