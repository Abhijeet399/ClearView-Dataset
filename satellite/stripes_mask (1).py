import cv2
import numpy as np
import os
from tqdm import tqdm

path = 'D:/satellitedataCopy/stripes'
dest = 'D:/satellitedataCopy/masks'
i=0
flag=0
for images in tqdm(os.listdir(path)):

	img = os.path.join(path,images)
	name = img.split('/')[2][8:].split('.')[0]
	img2 = cv2.imread(img)

	rows,cols,channels = img2.shape
	#roi = img1[0:rows,0:cols]
	img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	ret,mask=cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
	sum_ = np.sum(mask[:,:])
	if sum_ == 267386880:
		ret,mask=cv2.threshold(img2gray,50,255,cv2.THRESH_BINARY)

	os.chdir(dest)
	cv2.imwrite(str(name)+'.jpg', mask)
	i+=1

	
