import os
from PIL import Image
from matplotlib import image

#creates parsed  text file with image name and score

def getLabel(target):
    avatxt = r'AVA.txt'
    f = open(avatxt)
    for line in f:
        line = line.strip().split(' ')
        imgid = line[1]
        if (int(imgid) == target):
            totscore = 0
            totnum = 0
            for i in range(10):
                num = int(line[i+2])
                totscore += num*(i+1)
                totnum += num
            score = totscore/totnum
            #print(score)
            
            f.close()
            return score

def preprocess(data_type):
    # sequentially read from AVA_dataset/aesthetics_image_lists/fooddrink_train.jpgl
    fdtxt = r'AVA_dataset/aesthetics_image_lists/fooddrink_'+data_type+'.txt'
    savePath = r'fooddrink_imgs_'+data_type+'/'
    f = open(fdtxt, "r")
    

    sortedList = []

    # drop everything under 300x300
    minT = 224
    avgscore = 0
    counter = 0
    for line in f:
        line = line.strip().split('\n')
        imgID = line[0] #imageID
        if os.path.isfile(os.path.join(savePath, imgID + '.jpg')) == True:
            filename = savePath+imgID+'.jpg'
            # load image as pixel array
            data = image.imread(filename)

            #discard bad options
            if(len(data.shape) == 3): #else corrupted/wrong format
                H = data.shape[0]
                W = data.shape[1]
                if (H>= minT) and (W>-minT): # else too small
                    # get labels
                    score = getLabel(int(imgID))
                    avgscore += score
                    
                    sortedList.append(str(score) + ' ' + imgID + '\n')
                    counter +=1
                    print(counter)
        else:
            print("file has been deleted")
            # ignore, file has been deleted from repo
    sortedList.sort()
    print("avg: ", avgscore/counter)
    print("counter: ", counter)
    print("real counter: ", counter)
    f.close()

    print("done reading images. now writing to parsed file")

    fdparsed = r'AVA_dataset/fooddrink_'+data_type+'_parsedSorted2.txt'
    f2 = open(fdparsed, "a")
    minStart = 0 #min index to look for pair
    for i in range(len(sortedList)):
        text1 = sortedList[i].split()
        score1 = text1[0]
        imgID1 = text1[1]
        found = False
        for j in range(minStart, len(sortedList)):
            text2 = sortedList[j].split()
            score2 = text2[0]
            imgID2 = text2[1]
            if (float(score2) >= 2+float(score1)):
                f2.write(imgID1+' '+score1+' ' + imgID2+' '+score2+'\n')
                minStart = j
                found = True
                break
        if not found:
            #too big, end now
            break

    f2.close()

'''
filename = 'images/277832.jpg'
# load image as pixel array
data = image.imread(filename)
im = Image.open(filename)
print(im.format)
print(im.mode)
print(im.size)
# show the image
im.show()
print(data)
H = data.shape[0]'''

preprocess("train")
preprocess("test")