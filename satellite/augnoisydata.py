import imageio
import os
import cv2
import random
# import tensorflow.compat.v1 as tf
from tqdm import tqdm
import random
import numpy as np 
import cv2
import math
import shutil
import pickle

def defvert_flip(image):
    img2 = np.fliplr(image)
    return img2

def defvert_hori(image):
    img2 = np.flipud(image)
    return img2

def rotate90_clock(image):
    img2 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return img2

def rotate90_anticlock(image):
    img2 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img2

def rotate180(image):
    img2 = cv2.rotate(image, cv2.ROTATE_180)
    return img2

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def crop(image):
    crop_img = image[50:990, 50:990]
    # print(np.shape(crop_img))
    return crop_img

def clear_image(image):
    image = np.flipud(image)
    new_image = np.zeros(image.shape, image.dtype)
    ## [basic-linear-transform-output]

    ## [basic-linear-transform-parameters]
    alpha = 1.0 # Simple contrast control
    beta = 0    # Simple brightness control

    # Initialize values
    alpha = 1.3
        #float(input('* Enter the alpha value [1.0-3.0]: '))
    beta = 30
        #int(input('* Enter the beta value [0-100]: '))
    ## [basic-linear-transform-parameters]

    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    ## [basic-linear-transform-operation]
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    return new_image

# choices = [1,2,3,4,5,6,7,8]
# save_choice = []

# path = 'D:/satelitte/Noisy_data'
# auginputpath = 'D:/satelitte/augnoisydata' 
# movepath = 'D:/satelitte/noisydatamove'

# for images in tqdm(os.listdir(path)):
# 	image = os.path.join(path, images)
# 	s = image.split('/')[-1] [11:].split('.')[0]

# 	for i in range(1,3):
# 		temp = random.choice(choices)
# 		save_choice.append(temp)	
# 		filename = auginputpath + '/' +str(s) + '_' + str(i) + '.jpg'
# 		img = imageio.imread(image)
# 		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 		if temp == 1:
# 			rot = defvert_flip(img)
# 			# rot = tf.Session().run(rot)
# 			cv2.imwrite(filename, rot)
	
# 		elif temp == 2:
# 			sh = defvert_hori(img)
# 			# sh = tf.Session().run(sh)
# 			cv2.imwrite(filename, sh)
	
# 		elif temp == 3:
# 			hor = rotate90_clock(img)
# 			# hor = tf.Session().run(hor)
# 			cv2.imwrite(filename, hor)
	
# 		elif temp == 4:
# 			ver = rotate90_anticlock(img)
# 			# ver = tf.Session().run(ver)
# 			cv2.imwrite(filename, ver)
	
# 		elif temp == 5:
# 			bri = rotate180(img)
# 			# bri = tf.Session().run(bri)
# 			cv2.imwrite(filename, bri)
	
# 		elif temp == 6:
# 			rot2 = increase_brightness(img)
# 			# rot2 = tf.Session().run(rot2)
# 			cv2.imwrite(filename, rot2)
	
# 		elif temp == 7:
# 			sh2 = crop(img)
# 			# sh2 = tf.Session().run(sh2)
# 			cv2.imwrite(filename, sh2)
	
# 		elif temp == 8:
# 			rot3 = clear_image(img)
# 			# rot3 = tf.Session().run(rot3)
# 			cv2.imwrite(filename, rot3)

# 	shutil.move(image, movepath)
	# with open('checkpoint.data', 'wb') as filehandle:
	# 	pickle.dump(save_choice, filehandle)

with open('checkpoint.data', 'rb') as filehandle:
    # read the data as binary data stream
    placesList = pickle.load(filehandle)


total = placesList

labelpath = 'D:/satelitte/notnoisydata'
auglabelpath = 'D:/satelitte/augnotnoisydata'
movepath2 = 'D:/satelitte/notnoisydatamove'

j=0
for images in tqdm(os.listdir(labelpath)):
	image = os.path.join(labelpath, images)
	s = image.split('/')[-1] [13:].split('.')[0]

	for i in range(1,3):	
		temp = total[j]
		filename = auglabelpath + '/' +str(s) + '_' + str(i) + '.jpg'
		img = imageio.imread(image)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		if temp == 1:
			rot = defvert_flip(img)
			# rot = tf.Session().run(rot)
			cv2.imwrite(filename, rot)
	
		elif temp == 2:
			sh = defvert_hori(img)
			# sh = tf.Session().run(sh)
			cv2.imwrite(filename, sh)
	
		elif temp == 3:
			hor = rotate90_clock(img)
			# hor = tf.Session().run(hor)
			cv2.imwrite(filename, hor)
	
		elif temp == 4:
			ver = rotate90_anticlock(img)
			# ver = tf.Session().run(ver)
			cv2.imwrite(filename, ver)
	
		elif temp == 5:
			bri = rotate180(img)
			# bri = tf.Session().run(bri)
			cv2.imwrite(filename, bri)
	
		elif temp == 6:
			rot2 = increase_brightness(img)
			# rot2 = tf.Session().run(rot2)
			cv2.imwrite(filename, rot2)
	
		elif temp == 7:
			sh2 = crop(img)
			# sh2 = tf.Session().run(sh2)
			cv2.imwrite(filename, sh2)
	
		elif temp == 8:
			rot3 = clear_image(img)
			# rot3 = tf.Session().run(rot3)
			cv2.imwrite(filename, rot3)

		j+=1
	shutil.move(image, movepath2)
