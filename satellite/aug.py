import imageio
import os
import cv2
import random
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import shutil

# tf.disable_v2_behavior()

def rot90(image):
	rot = tf.image.rot90(image)
	return rot

def saturation2(image):
	sat = tf.image.random_saturation(image, 3,6, seed=1)
	return sat

def hue(image):
	hue = tf.image.random_hue(image, max_delta=0.2, seed=1)
	return hue

def centrecrop(image):
	cropped = tf.image.central_crop(image, central_fraction=0.75)
	return cropped

def fliphorizontal(image):
	flip_hr_image = tf.image.flip_left_right(image)
	return flip_hr_image

def flipvertical(image):
	flip_vr_image = tf.image.flip_up_down(image)
	return flip_vr_image

def brightness(image):
	contrast_image = tf.image.random_brightness(image,max_delta=0.2, seed=1)
	return contrast_image

def saturation(image):
	sat = tf.image.random_saturation(image, 3, 7, seed=1)
	return sat

def contrast(image):
	contrast = tf.image.random_contrast(image, 2, 3, seed=1)
	return contrast

def brightness2(image):
	bright = tf.image.random_brightness(image,max_delta=0.15, seed=1)
	return bright

# path = 'D:/satelitte/Dataset/Clouds'
# auginputpath = 'D:/satelitte/Dataset/augclouds' 
# movepath = 'D:/satelitte/Dataset/cloud'
# for images in tqdm(os.listdir(path)):
# 	image = os.path.join(path, images)
# 	s = image.split('/')[-1] [7:].split('.')[0]
# 	for i in range(1,11):	
# 		filename = auginputpath + '/' +str(s) + '_' + str(i) + '.jpg'
# 		img = imageio.imread(image)
# 		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 		if i == 1:
# 			rot = rot90(img)
# 			rot = tf.Session().run(rot)
# 			cv2.imwrite(filename, rot)
	
# 		elif i == 2:
# 			sh = saturation2(img)
# 			sh = tf.Session().run(sh)
# 			cv2.imwrite(filename, sh)
	
# 		elif i == 3:
# 			hor = hue(img)
# 			hor = tf.Session().run(hor)
# 			cv2.imwrite(filename, hor)
	
# 		elif i == 4:
# 			ver = centrecrop(img)
# 			ver = tf.Session().run(ver)
# 			cv2.imwrite(filename, ver)
	
# 		elif i == 5:
# 			bri = fliphorizontal(img)
# 			bri = tf.Session().run(bri)
# 			cv2.imwrite(filename, bri)
	
# 		elif i == 6:
# 			rot2 = flipvertical(img)
# 			rot2 = tf.Session().run(rot2)
# 			cv2.imwrite(filename, rot2)
	
# 		elif i == 7:
# 			sh2 = brightness(img)
# 			sh2 = tf.Session().run(sh2)
# 			cv2.imwrite(filename, sh2)
	
# 		elif i == 8:
# 			rot3 = saturation(img)
# 			rot3 = tf.Session().run(rot3)
# 			cv2.imwrite(filename, rot3)
	
# 		elif i == 9:
# 			crop_image = contrast(img)
# 			crop_image = tf.Session().run(crop_image)
# 			cv2.imwrite(filename, crop_image)
	
# 		elif i == 10:
# 			bri2 = brightness2(img)
# 			bri2 = tf.Session().run(bri2)
# 			cv2.imwrite(filename, bri2)
# 	shutil.move(image, movepath)

labelpath = 'D:/satelitte/Dataset/Non-clouds'
auglabelpath = 'D:/satelitte/Dataset/nonaugclouds'
movepath2 = 'D:/satelitte/Dataset/noncloud'
temp = []
for images in tqdm(os.listdir(labelpath)):
	image = os.path.join(labelpath, images)
	s = image.split('/')[-1] [11:].split('.')[0]

	for i in range(1,11):	
		filename = auglabelpath + '/' +str(s) + '_' + str(i) + '.jpg'

		img = imageio.imread(image)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		if i == 1:
			rot = rot90(img)
			rot = tf.Session().run(rot)
			cv2.imwrite(filename, rot)
	
		elif i == 2:
			sh = saturation2(img)
			sh = tf.Session().run(sh)
			cv2.imwrite(filename, sh)
	
		elif i == 3:
			hor = hue(img)
			hor = tf.Session().run(hor)
			cv2.imwrite(filename, hor)
	
		elif i == 4:
			ver = centrecrop(img)
			ver = tf.Session().run(ver)
			cv2.imwrite(filename, ver)
	
		elif i == 5:
			bri = fliphorizontal(img)
			bri = tf.Session().run(bri)
			cv2.imwrite(filename, bri)
	
		elif i == 6:
			rot2 = flipvertical(img)
			rot2 = tf.Session().run(rot2)
			cv2.imwrite(filename, rot2)
	
		elif i == 7:
			sh2 = brightness(img)
			sh2 = tf.Session().run(sh2)
			cv2.imwrite(filename, sh2)
	
		elif i == 8:
			rot3 = saturation(img)
			rot3 = tf.Session().run(rot3)
			cv2.imwrite(filename, rot3)
	
		elif i == 9:
			crop_image = contrast(img)
			crop_image = tf.Session().run(crop_image)
			cv2.imwrite(filename, crop_image)
	
		elif i == 10:
			bri2 = brightness2(img)
			bri2 = tf.Session().run(bri2)
			cv2.imwrite(filename, bri2)
	shutil.move(image, movepath2)