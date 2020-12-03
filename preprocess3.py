import numpy as np
import pickle as pk
from PIL import Image
from matplotlib import image
import shutil, os

#moved necessary iamges into different folder. 

def move(data_type):
    #sequentially read from AVA_dataset/aesthetics_image_lists/fooddrink_<train or test>.jpgl
    fdparsed = r'AVA_dataset/fooddrink_'+data_type+'_parsed.txt'
    f = open(fdparsed, "r")
    destfolder = 'fooddrink_imgs_' +data_type+'/'

    for line in f:
        line = line.strip().split(' ')
        imgID = line[0] #imageID

        filename = 'images/'+imgID+'.jpg'
        shutil.copy(filename, destfolder+imgID+'.jpg')

    f.close()

move("train")
print("train done")
#move("test")
#print("test done")