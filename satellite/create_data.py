import cv2
import os
import shutil
from tqdm import tqdm

path = "D:/satelitte/train/train/images/"
dest = "E:/DL/satellite/clouds"

contours_list = []
name_list = []
disaster_list = []
temp = 0

for images in tqdm(os.listdir(path)):
	image = os.path.join(path,images)
	img = cv2.imread(image)
	disaster = image[31:49]
	
	if disaster != temp:
		disaster_list.append(disaster)

	temp = disaster

for images in tqdm(os.listdir(path)):
	image = os.path.join(path,images)
	img = cv2.imread(image)
	disaster = image[31:49]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edged = cv2.Canny(img, 30, 200)

	if len(name_list) >= 2:
		name_list.clear()

	name_list.append(image)

	contours, hierarchy = cv2.findContours(edged, 
		cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	if len(contours_list) >= 2:
		contours_list.clear()

	contours_list.append(len(contours))

	if len(contours_list) > 1:
		if disaster == disaster_list[0]:
			if (contours_list[1] - contours_list[0]) > 290:
				shutil.move(name_list[0], dest)

		elif disaster == disaster_list[1]:
			if (contours_list[1] - contours_list[0]) > 500:
				shutil.move(name_list[0], dest)

		elif disaster == disaster_list[2]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)

		elif disaster == disaster_list[3]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[4]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[5]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[6]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[7]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[8]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[9]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[10]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[11]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[12]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[13]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[14]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[15]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[16]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[17]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[18]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[19]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[20]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[21]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[22]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
		elif disaster == disaster_list[23]:
			if (contours_list[1] - contours_list[0]) > 300:
				shutil.move(name_list[0], dest)
		
