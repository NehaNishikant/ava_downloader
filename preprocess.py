import os
from PIL import Image
from matplotlib import image

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
    fdparsed = r'AVA_dataset/fooddrink_'+data_type+'_parsed.txt'
    savePath = r'fooddrink_imgs_'+data_type+'/'
    f = open(fdtxt, "r")
    f2 = open(fdparsed, "a")

    # drop everything under 300x300
    minT = 300
    avgscore = 0
    counter = 0
    for line in f:
        line = line.strip().split('\n')
        imgID = line[0] #imageID
        if os.path.isfile(os.path.join(savePath, imgID + '.jpg')) == True:
            filename = 'images/'+imgID+'.jpg'
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
                    f2.write(imgID+' '+str(score)+'\n')
                    counter +=1
        else:
            print("file has been deleted")
            # ignore, file has been deleted from repo
    print("avg: ", avgscore/counter)
    print("counter: ", counter)
    f.close()
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