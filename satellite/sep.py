import os
import shutil
from tqdm import tqdm

path = 'D:/satelitte/Noisy_data - Copy'
# noise_data = 'D:/satelitte/noise2'
notnoise_data = 'D:/satelitte/notnoisydata'

for images in tqdm(os.listdir(path)):
	image = os.path.join(path, images)
	name = images.split('_')[2]
	if name == 'post':
		shutil.move(image, notnoise_data)
