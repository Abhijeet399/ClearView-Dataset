import cv2
import numpy as np
import os
import random

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

a=0
path = "C:\\Users\\bhatt\\Desktop\\Jupyter Notebook\\Missing Data\\train"
directory = "C:\\Users\\bhatt\\Desktop\\Jupyter Notebook\\Missing Data\\Noisy_data"
for img in os.listdir(path):
    image = cv2.imread(os.path.join(path,"guatemala-volcano_00000000_pre_disaster.png"))
    if (a%2==0):
        noisy_imgs = sp_noise(image, 0.05)
        os.chdir(directory)
        cv2.imshow('img',noisy_imgs) 
        cv2.imwrite(img,noisy_imgs)
        print(a)
        a=a+1
        break
    if(a%2==1):
        os.chdir(directory)         
        cv2.imwrite(img,image)
        print(a)
        a=a+1
print(a)
a=a+1
    
