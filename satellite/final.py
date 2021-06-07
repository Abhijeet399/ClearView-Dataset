
import os
import random
import pickle
from tqdm import tqdm

path = 'D:/satellitedataCopy/noisydata'
shuffle_noise = 'D:/satellitedataCopy/shufflednoisedata'
labelpath = 'D:/satellitedataCopy/label'
shuffle_label = 'D:/satellitedataCopy/shufflelabel'
maskpath = 'D:/satellitedataCopy/masks'
shufflemaskpath = 'D:/satellitedataCopy/shufflemask'
# temp = []
# totalimages = 21080
# save_choice = []

# for i in range(totalimages):
# 	temp.append(i)

# for images in tqdm(os.listdir(path)):
# 	name = random.choice(temp)
# 	save_choice.append(name)
# 	temp.remove(name)

# 	src = path + '/' + images
# 	dst = shuffle_noise + '/' + str(name) + '.jpg'
# 	os.rename(src, dst) 

# with open('choices.data', 'wb') as filehandle:
# 	pickle.dump(save_choice, filehandle)
with open('choices.data', 'rb') as filehandle:
    # read the data as binary data stream
    choices = pickle.load(filehandle)


j=0
for images in tqdm(os.listdir(maskpath)):
	name = choices[j]

	src = maskpath + '/' + images
	dst = shufflemaskpath + '/' + str(name) + '.jpg'
	os.rename(src, dst) 
	j+=1
