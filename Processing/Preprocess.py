import os
import os.path as op
import numpy as np
import skimage
import scipy
import cv2


class Preprocess:
	"""Preprocess the respective dataset"""
	def __init__(self, data_list, data_dir, save_dir, condition):
		"""Initialize the preprocess class for respective dataset.

		:param data_list: Full data list of the respective dataset
		:param data_dir: Directory to the dataset
		:param save_dir: Directory to save the preprocessed dataset
		:param condition: Condition of the dataset (CNV, DME, or Drusen)
		"""
		if save_dir is None:
			self.save_dir = '/home/ubuntu/sl_root/Processed_data'
		else:
			self.save_dir = save_dir
		self.data_list = data_list
		self.condition = condition
		self.data_dir = data_dir

		# Variables to store data
		self.create_save_directories()

	def create_save_directories(self):
		"""Creates the save directories for processed OCT data.

		:return: self
		"""
		print('self.save_dir: ', self.save_dir)
		if not op.exists(self.save_dir):
			os.mkdir(self.save_dir)

		if not op.exists(op.join(self.save_dir, self.condition)):
			os.mkdir(op.join(self.save_dir, self.condition))

		if not op.exists(op.join(self.save_dir, self.condition, 'OCT')):
			os.mkdir(op.join(self.save_dir, self.condition, 'OCT'))

		if not op.exists(op.join(self.save_dir, self.condition, 'Segmentation')):
			os.mkdir(op.join(self.save_dir, self.condition, 'Segmentation'))
		return self

	def resize(self, single_file, size=(512, 512)):
		"""Resize the image to size.

		:param single_file: Single file name to read
		:param size: Resize values (default is (512, 512))
		:return: Resized image
		"""
		# Read the image
		img = skimage.io.imread(op.join(self.data_dir, self.condition, single_file))
		return skimage.transform.resize(image=img, output_shape=size, anti_aliasing=True)

	@staticmethod
	def create_segmentation_mask(image):
		"""Create the segmentation mask for the respective dataset.

		:param image: Single image
		:return: segmentation mask
		"""
		# Create the blurred image of original image
		blurred = cv2.blur(image.astype(np.uint8), ksize=(15, 15))

		# Obtain the background which is the noise data
		noise_data = cv2.addWeighted(src1=image.astype(np.uint8), alpha=1.5, src2=blurred, beta=-4, gamma=64)

		# Remove salt and pepper noise from OCT binary mask
		oct_bin_mask = np.where(noise_data == noise_data.min(), 1, 0)
		oct_bin_mask = scipy.ndimage.median_filter(oct_bin_mask, size=15)

		# Only obtain the largest segmentation 'blob'
		labels_mask = skimage.measure.label(input=oct_bin_mask)
		regions = skimage.measure.regionprops(label_image=labels_mask)
		regions.sort(key=lambda x: x.area, reverse=True)
		if len(regions) > 1:
			for rg in regions[1:]:
				labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
		labels_mask[labels_mask != 0] = 1
		mask = labels_mask

		# Fill in the largest blob to obtain better segmentation effect
		mask = scipy.ndimage.morphology.binary_fill_holes(mask).astype(int)
		return mask

	def save(self, image, file_name, file_type):
		if file_type == 'OCT':
			skimage.io.imsave(op.join(self.save_dir, self.condition, 'OCT', file_name), image)
		elif file_type == 'Segmentation':
			skimage.io.imsave(op.join(self.save_dir, self.condition, 'Segmentation', file_name), image)
		else:
			raise NameError('Invalid file type specified.')
		pass

	def full_data(self):
		"""Preprocess for the full data.

		:return:
		"""
		# Iterate through the full data list as specified in the initialization
		for single_file in self.data_list:
			print('Processing for: {}\n'.format(single_file))
			# Resize the image
			img = self.resize(single_file=single_file)

			# Remove the grainy images and images with white background
			img = img * 255

			# Set pixel values into histogram
			hist, bin_edges = np.histogram(img, bins=np.arange(0, 255), density=False)

			# Obtain the pixel range of [10, 50] and [250 to 255]
			hist_1 = hist[10:51]
			hist_2 = hist[250:]

			# Exclude the grainy images or images with white background using the respective threshold of 5000 and 100
			if any(t > 5000 for t in hist_1) and all(t < 100 for t in hist_2):
				# Create the segmentation mask
				seg_mask = self.create_segmentation_mask(image=img)

				# Convert to uint8
				img = img.astype(np.uint8)

				# Segment the OCT information from the scan using the segmentation mask
				img[seg_mask == seg_mask.min()] = img.min()

				# Save both the segmented OCT scan and the segmentation mask
				self.save(image=img.astype(np.uint8), file_name=single_file, file_type='OCT')
				self.save(image=seg_mask, file_name=single_file, file_type='Segmentation')
		pass








