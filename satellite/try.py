import numpy as np 
import cv2
import os
from tqdm import tqdm
import shutil

path = "E:/DL/satellite/tier3/images"
cloud_dest = "E:/DL/satellite/clouds"
noncloud_dest = "E:/DL/satellite/nonclouds"

def sum_matrix(block):
	sum = 0
	for i in range(len(block)):
		for j in range(len(block)):
			sum += block[i][j]
	return sum	

name_list = []
max_pixels = []
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
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	disaster = image[31:49]
	social_fire = disaster_list[9]
	hurricane_matthew = disaster_list[3]

	if len(name_list) >= 2:
		name_list.clear()
	name_list.append(image)

	values = []
	j=0
	k=0
	for j in range(0,1024,64):
		for k in range(0,1024,64):
			block = gray[j:64+j,k:64+k]
			values.append(sum_matrix(block))
	
	if len(max_pixels) >= 2:
		max_pixels.clear()

	max_pixels.append(max(values))

	if len(max_pixels) > 1:
		if disaster[1:11] == social_fire[1:11]:
			pass
			# if max_pixels[1] - max_pixels[0] > 18000:
			# 	shutil.move(name_list[1], cloud_dest)
			# 	shutil.move(name_list[0], noncloud_dest)
			# elif max_pixels[0] - max_pixels[1] > 400000:
			# 	shutil.move(name_list[0], cloud_dest)
			# 	shutil.move(name_list[1], noncloud_dest)

		elif disaster[1:17] == hurricane_matthew[1:17]:
			pass

		else:
			if max_pixels[0] - max_pixels[1] > 20000:
				shutil.move(name_list[0], cloud_dest)
				shutil.move(name_list[1], noncloud_dest)
			# elif max_pixels[1] - max_pixels[0] > 400000:
			# 	shutil.move(name_list[0], cloud_dest)
			# 	shutil.move(name_list[1], noncloud_dest)












#(16,16) -> 53737,55155,65280,55003
#(32,32) -> 194856, 261057