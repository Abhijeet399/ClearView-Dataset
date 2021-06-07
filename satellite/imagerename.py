import cv2
import os

path = "D:/satellitedata/noncloud"
temp=21072

for filename in os.listdir(path):
    src = path + '/' + filename
    dst = path + '/' + str(temp) + '.jpg' 
    
    os.rename(src, dst) 
    temp = int (temp) + 1