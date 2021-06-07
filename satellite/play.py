import cv2
import numpy as np
import os
import imageio

# path = 'D:/satellitedataCopy/noncloud'

# def vert_flip(image):
#     img2 = np.fliplr(image)
#     return img2
# i=21064
# for images in os.listdir(path):
# 	image = os.path.join(path, images)
# 	img = imageio.imread(image)
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	rot = vert_flip(img)
# 	filename = path + '/' +str(i) + '_' + str(1) + '.jpg'
# 	cv2.imwrite(filename, rot)
# 	i+=1
import tensorflow as tf 
import pickle
a = tf.constant(3)
b = tf.constant(3)
l=[]
ll=[]
l.append(a)
l.append(b)
with tf.Session() as sess:
	for i in range(5):
		o = sess.run(l)
		with open('%d.data'%i,'wb') as filehandle:
			pickle.dump(o, filehandle)
