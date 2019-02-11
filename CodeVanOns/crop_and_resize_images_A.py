from PIL import Image
import numpy as np
from skimage import transform, filters

read_folder = 'celeba/img_align_celeba_original/'
write_folder = 'celeba/img_align_celeba_resized_and_cropped/'
no_images = 202599
rescale_size = 32

for index in range(1, no_images + 1):
	if index % 1000 == 1:
		print('Processing image: ', '%06d' % index + '.jpg')
	image = Image.open(read_folder + '%06d' % index + '.jpg').crop((40, 75, 178-40, 218-45))
	image = np.array(image)
	scale = image.shape[0] / float(rescale_size)
	sigma = np.sqrt(scale) / float(2)
	image = filters.gaussian(image, sigma=sigma, multichannel=True)
	image = transform.resize(image, (rescale_size, rescale_size, 3), order=3)
	image = (image*255).astype(np.uint8)
	image = Image.fromarray(image)
	image.save(write_folder + '%06d' % index + '.jpg')