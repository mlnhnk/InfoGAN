from PIL import Image

folder = 'celeba/img_align_celeba/'
no_images = 202599
new_sizes = (32, 32)

for index in range(1, no_images + 1):
	image = Image.open(folder + '%06d' % index + '.jpg')
	resized_image = image.resize(new_sizes, Image.ANTIALIAS)
	resized_image.save(folder + '%06d' % index + '.jpg')