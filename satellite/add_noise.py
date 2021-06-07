import cv2
import numpy as np
import os
import random

def sp_noise(image,prob):
    output = np.zeros_like(image)
    thres = 1 - prob 
    for i in range(image.shape[0]-1):
        for j in range(image.shape[1]-1):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
                output[i][j+1] = 0
                output[i+1][j] = 0
                output[i+1][j+1] = 0
            elif rdn > thres:
                output[i][j] = 255
                output[i][j+1] = 255
                output[i+1][j] = 255
                output[i+1][j+1] = 255
            else:
                output[i][j] = image[i][j]
    return output

a=0
path = "C://Users\\bhatt\\Desktop\\Jupyter Notebook\\Missing Data\\train"
directory = "C:\\Users\\bhatt\\Desktop\\Jupyter Notebook\\Missing Data\\Noisy_data"
for img in os.listdir(path):
    image = cv2.imread(os.path.join(path,img))
    if (a>=7041):
        if (a%2==0):
            noisy_imgs = sp_noise(img, 0.05)
            os.chdir(directory)         
            cv2.imwrite(img,noisy_imgs)
            print(a)
            a=a+1
            continue
        elif(a%2==1):
            os.chdir(directory)         
            cv2.imwrite(img,image)
            print(a)
            a=a+1
            continue
        
    if (a%2==0):
        for i in range(13):
            image = cv2.line(image, (0,(i*80)), (1024,50+(i*80)), (0,0,0), 25)
        os.chdir(directory) 
        cv2.imwrite(img,image)
    if (a%2==1):
        os.chdir(directory)         
        cv2.imwrite(img,image)        
    print(a)
    a=a+1
    
