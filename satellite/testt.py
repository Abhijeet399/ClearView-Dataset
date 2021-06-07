import cv2
import numpy as np


img2 = cv2.imread('6220_2.png',1)
#img2 = cv2.imread('99.jpeg',1)
rows,cols,channels = img2.shape
#roi = img1[0:rows,0:cols]
img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask=cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY_INV)

sum = np.sum(mask[1024:1024, 1024:1024])
print(sum)
cv2.imshow('mask',mask) 
cv2.waitKey(0)
cv2.destroyAllWindows()
