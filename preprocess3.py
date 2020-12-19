import numpy as np
import pickle as pk
from PIL import Image
from matplotlib import image
import shutil, os

#moved necessary images into different folder. 

def move(data_type):
    #sequentially read from AVA_dataset/aesthetics_image_lists/fooddrink_<train or test>.jpgl
    fdparsed = r'AVA_dataset/aesthetics_image_lists/fooddrink_'+ data_type +'.txt'
    f = open(fdparsed, "r")
    destfolder = 'fooddrink_imgs_' +data_type+'/'

    for line in f:
        line = line.strip().split(' ')
        imgID = line[0] #imageID

        filename = 'images/'+imgID+'.jpg'
        shutil.copy(filename, destfolder+imgID+'.jpg')

    f.close()

#moved some from test to train
def move2():
    fdparsed = r'AVA_dataset/aesthetics_image_lists/fooddrink_train2.txt'
    f = open(fdparsed, "r")
    destfolder = 'fooddrink_imgs_train/'
    searchPath = r'fooddrink_imgs_test/'
    counter = 0

    for line in f:
        line = line.strip().split(' ')
        imgID = line[0] #imageID

        if os.path.isfile(os.path.join(searchPath, imgID + '.jpg')) == True:
            filename = searchPath+imgID+'.jpg'
            shutil.move(filename, destfolder+imgID+'.jpg')
            counter+=1
    
    print(counter)
    f.close()

#move("train")
#print("train done")
#move("test")
#print("test done")
move2()