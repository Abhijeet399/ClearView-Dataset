import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tqdm import tqdm

tf.reset_default_graph()

training_data = []
training_data_bw=[]
label = []
path = "D:/satellitedata/shufflenoisydata"
path_label = "D:/satellitedata/shufflelabel"

for img in tqdm(os.listdir(path)):
	image = os.path.join(path, img)
	img_array = cv2.imread(image)
	new_array=cv2.resize(img_array,(1024,1024))
	new_array_bw = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
	training_data.append(new_array)
	training_data_bw.append(new_array_bw)

for img in tqdm(os.listdir(path_label)):
	image = os.path.join(path_label, img)
	img_array = cv2.imread(image) 
	new_array=cv2.resize(img_array,(1024,1024))
	label.append(new_array)

def discriminator(img,reuse=None):
	with tf.variable_scope('dis',reuse=reuse):
		flat = tf.contrib.layers.flatten(img)
        # print(np.shape(flat))
		hidden_1 = tf.layers.dense(inputs=flat,units=128, activation=tf.nn.leaky_relu, name='hidden_1')
        #hidden1=tf.layers.dense(inputs=flat,units=128,activation=tf.nn.leaky_relu)
		hidden_2 = tf.layers.dense(inputs=hidden_1,units=128, activation=tf.nn.leaky_relu, name='hidden_2')
        #hidden2=tf.layers.dense(inputs=hidden1,units=128,activation=tf.nn.leaky_relu)
		logits= tf.layers.dense(inputs=hidden_2, units=1)
		output=tf.sigmoid(logits)
	return output,logits

X_noisy = tf.placeholder(tf.float32, (None, 1024, 1024, 3), name='inputs')
X_noisy_bw = tf.placeholder(tf.float32, (None, 1024, 1024, 1), name='inputs')
X_label = tf.placeholder(tf.float32, (None, 1024, 1024, 3), name='targets')
filters = {1:20,2:20,3:60, 4:60, 5:60, 6:60}

def AutoEncoder(X_noisy, X_noisy_bw, reuse=None):
	with tf.variable_scope('gen',reuse=reuse):
        ### Encoder
		conv1 = tf.layers.conv2d(X_noisy, filters[1], (3,3),strides=(1, 1), padding='valid', activation=tf.nn.relu, use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.zeros_initializer())
		conv2 = tf.layers.conv2d(X_noisy_bw, filters[2], (3,3),strides=(1,1),padding='valid',activation=tf.nn.relu, use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.zeros_initializer())
		conv = tf.concat([conv1, conv2], 3)
		conv_shortcut = conv
		conv3 = tf.layers.conv2d(conv, filters[3],(5,5), strides=(2, 2), padding='valid', activation=tf.nn.relu,use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.zeros_initializer())
		maxpool1 = tf.layers.max_pooling2d(conv3,(5,5), (1,1), padding='valid')
		conv4 = tf.layers.conv2d(maxpool1,filters[4],(7,7),strides=(2,2),padding='valid', activation=tf.nn.relu,use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.zeros_initializer())
		maxpool2 = tf.layers.max_pooling2d(conv4,(5,5), (1,1), padding='valid')
		convD1 = tf.layers.conv2d(maxpool2,filters[5],(3,3), strides=(1, 1), padding='valid',dilation_rate=(2, 2), activation=tf.nn.relu,use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.zeros_initializer())
		convD2 = tf.layers.conv2d(convD1,filters[6],(5,5), strides=(1, 1), padding='valid',dilation_rate=(3, 3), activation=tf.nn.relu,use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.zeros_initializer())
		maxpool3 = tf.layers.max_pooling2d(convD2,(7,7), (2,2), padding='valid')
		conv_shortcut1 = tf.layers.conv2d(conv_shortcut, 60, (106,106),strides=(4,4), padding='valid', activation=tf.nn.relu)
    	# print(np.shape(conv_shortcut))
    	# print(np.shape(convD2))
    	# print(np.shape(conv_shortcut1))
		convD2 = tf.math.add(convD2, conv_shortcut1)
    	# print(np.shape(convD2))
        ###Decoder
		upsample1 = tf.image.resize_nearest_neighbor(convD2, (512,512))
        # print(np.shape(upsample1))
		conv4 = tf.layers.conv2d(upsample1, 40, (7,7), padding='same',activation=tf.nn.relu)
		upsample2 = tf.image.resize_nearest_neighbor(conv4, (768,768))
		conv5 = tf.layers.conv2d(upsample2, 20, (5,5), padding='same',activation=tf.nn.relu)
		upsample3 = tf.image.resize_nearest_neighbor(conv5, (1024,1024))
		conv6 = tf.layers.conv2d(upsample3, 3, (7,7), padding='same',activation=tf.nn.relu)
        # print(np.shape(conv6))
		return conv6

conv6 = AutoEncoder(X_noisy, X_noisy_bw)
print(np.shape(conv6))

D_output_real,D_logits_real=discriminator(X_label)
D_output_fake,D_logits_fake=discriminator(conv6,reuse=True)

def loss_func(logits_in,labels_in):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

D_real_loss=loss_func(D_logits_real,tf.ones_like(D_logits_real)*0.9) #Smoothing for generalization
D_fake_loss=loss_func(D_logits_fake,tf.zeros_like(D_logits_real))
D_loss=D_real_loss+D_fake_loss

G_loss= loss_func(D_logits_fake,tf.ones_like(D_logits_fake))

lr=0.001

#Do this when multiple networks interact with each other
tvars=tf.trainable_variables()  #returns all variables created(the two variable scopes) and makes trainable true
d_vars=[var for var in tvars if 'dis' in var.name]
g_vars=[var for var in tvars if 'gen' in var.name]

D_trainer=tf.train.AdamOptimizer(lr).minimize(D_loss,var_list=d_vars)
print(D_trainer)
G_trainer=tf.train.AdamOptimizer(lr).minimize(G_loss,var_list=g_vars)
print(G_trainer)

batch_size=124
totalsize = 21080
epochs=10
init=tf.global_variables_initializer()

Gcost = []
Dcost = []

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):
		num_batches = totalsize/batch_size
		for i in range(num_batches):
			X = training_data[i*batch_size:batch_size+(i*batch_size)]
			X_bw = training_data_bw[i*batch_size:batch_size+(i*batch_size)]
			Y = label[i*batch_size:batch_size+(i*batch_size)]
            
			_, temp_d_loss = sess.run(D_trainer,feed_dict={X_label:Y,X_noisy:X, X_noisy_bw:X_bw})
			_, temp_g_loss =sess.run(G_trainer,feed_dict={X_noisy:X, X_noisy_bw:X_bw})

			minibatch_Dloss += temp_d_loss / num_batches
			minibatch_Gloss += temp_g_loss / num_batches
        
            
		print("on epoch{}".format(epoch))
        # Print the cost every epoch
		print ("Cost after epoch %i: %f" % (epoch, minibatch_Dloss))
		print ("Cost after epoch %i: %f" % (epoch, minibatch_Gloss))
		Gcost.append(minibatch_Gloss)
		Dcost.append(minibatch_Dloss)
        
plt.plot(Gcost)
plt.plot(Dcost)