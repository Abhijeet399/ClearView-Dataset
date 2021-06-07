import keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array, load_img
import os
import cv2
from tqdm import tqdm
# train_datagen=ImageDataGenerator(rotation_range=10,
#                            width_shift_range=4,
#                            height_shift_range=4,
#                            rescale=1/255.0,
#                            horizontal_flip=True,
#                            vertical_flip=True,
#                            fill_mode='nearest')
train_datagen=ImageDataGenerator(zoom_range=[0.5,1.0])



image_folder = "D:/satelitte/Dataset/Clouds"
files=os.listdir(image_folder)
files=list(map(lambda x: os.path.join(image_folder,x),files))
a=(len(files))
for i in tqdm(range(a)):
    im = (files[i])
    s = im.split('/')[-1] [7:].split('.')[0]
    s = int(s)
    img = cv2.imread(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = img.reshape((1,) + img.shape)
    i = 0
    for batch in train_datagen.flow(img, batch_size=1,save_to_dir='D:/satelitte/Dataset/augclouds', save_prefix=s, save_format='jpg'):
        #print(batch.shape)
        i += 1
        if i >0 : ## making 2
            break  # otherwise the generator would loop indefinitely    
    break