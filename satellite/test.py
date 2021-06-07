import numpy as np
import os
import cv2
import random

def sp_noise(image,prob):
    output = np.zeros_like(image)
    thres = 1 - prob 
    for i in range(0, image.shape[0]-1):
        for j in range(0, image.shape[1]-1):
            rdn = random.random()
            if rdn < prob:
                output[i+1][j] = 0
                output[i][j+1] = 0
                output[i][j] = 0
                output[i+1][j+1] = 0
            elif rdn > thres:
                output[i][j] = 255
                output[i+1][j] = 255
                output[i][j+1] = 255
                output[i+1][j+1] = 255
            else:
                output[i][j] = image[i][j]
    return output

a=0
path = "C:\\Users\\bhatt\\Desktop\\Jupyter Notebook\\Missing Data\\train"
directory = "C:\\Users\\bhatt\\Desktop\\Jupyter Notebook\\Missing Data\\Noisy_data"
for img in os.listdir(path):
    image = cv2.imread(os.path.join(path,img))
    if (a%2==0):
        print(a)
        a=a+1
        noisy_imgs = sp_noise(image, 0.05)
        #os.chdir(directory)
        cv2.imwrite(img,noisy_imgs)
        cv2.imshow('img', noisy_imgs)
        
    if(a%2==1):
        print(a)
        a=a+1
        #os.chdir(directory)         
        cv2.imwrite(img,image)
        
