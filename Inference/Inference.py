import os
import os.path as op
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


class LR_Inference:
	"""Low resolution inference using the GAN model"""
	def __init__(self, name, condition):
		"""Initialize the inference class for respective dataset.

		:param name: Name of the run
		:param condition: Condition of the dataset (CNV, DME, or Drusen)
		"""
		self.name = name
		self.condition = condition

		# Variables to store data
		self.gen_image_dir = None
		self.last_gen_model = None
		self.create_save_directories()

	def create_save_directories(self):
		"""Creates the save directories for generated OCT images.

		:return: self
		"""
		# Set the name for the generated OCT images directory
		self.gen_image_dir = op.join('..\\logs', self.name, 'lr_gen_images')

		if not op.exists(self.gen_image_dir):
			os.mkdir(self.gen_image_dir)
		return self

	def load_gen_model(self):
		"""Loads the latest generator model.

		:return: last generator model
		"""
		# Set the model directory
		model_dir = op.join('..\\logs', self.name, 'models')

		# Obtain the full list generator models
		gen_model_list = glob(op.join(model_dir, '*gen_model*'))
		gen_model_list = [single_model.split('\\')[-1] for single_model in gen_model_list]

		# Check if there are any generator models
		if len(gen_model_list) == 0:
			raise NameError('No existing generator models.')
		else:
			# Obtain the full list of epochs
			epoch_list = [int(single_model.split('_')[-1].split('.')[0]) for single_model in gen_model_list]

			# Obtain the last epoch
			last_epoch = sorted(epoch_list, reverse=True)[0]

			# Obtain the index of the last generator model with the highest epoch
			last_gen_model_idx = [int(single_model.split('_')[-1].split('.')[0]) == int(last_epoch) for single_model in
								  gen_model_list]
			last_gen_model_idx = np.where(last_gen_model_idx)[0]

			# Obtain the last generator model with the highest epoch
			self.last_gen_model = gen_model_list[last_gen_model_idx[0]]

			# Check the condition against the condition as specified in initialization
			condition = self.last_gen_model.split('_')[0]
			assert self.condition == condition, 'Conditions are not the same.'

		# Load the generator model
		gen_model = load_model(op.join(model_dir, self.last_gen_model), compile=False)
		return gen_model

	def generate_images(self, num_images, display=False, save=True):
		"""Generates the low resolution OCT image (128 x 128)

		:param num_images: Number of images to generate
		:param display: Specify to display the image
		:param save: Specify to save the image
		:return: None
		"""
		gen_model = self.load_gen_model()

		print('Generating images using generator model: ', self.last_gen_model)

		# Generate the images according to the number of images specified
		for i in range(num_images):

			# Initialize the noise vector for prediction
			noise = tf.random.normal([1, 100])

			# Predict using the generator
			gen_img = gen_model.predict(noise)
			gen_img = gen_img[0]

			# Save the image in the generated image directory
			plt.figure(figsize=(10, 10))
			plt.imshow(gen_img[:, :, 0], cmap='bone')
			plt.axis('off')
			if save:
				plt.savefig(op.join(self.gen_image_dir, 'gen_img_{num}.png'.format(num=str(i+1))))
			if display:
				plt.show()
				plt.close(plt.gcf())
			else:
				plt.close(plt.gcf())
		pass


class SR_Inference(LR_Inference):
	"""Super resolution inference using the super resolution model"""
	def __init__(self, name, condition):
		"""Initialize the inference class for respective dataset.

		:param name: Name of the run
		:param condition: Condition of the dataset (CNV, DME, or Drusen)
		"""
		super().__init__(name, condition)

		# Variables to store data
		self.gen_image_dir = None
		self.last_sr_model = None
		self.create_save_directories()

	@staticmethod
	def compute_mean(x):
		return tf.math.reduce_mean(x)

	def normalize(self, x):
		return (x - self.compute_mean(x)) / tf.math.reduce_std(x)

	def denormalize(self,x):
		return x * tf.math.reduce_std(x) + self.compute_mean(x)

	@staticmethod
	def pixel_shuffle(scale):
		return lambda x: tf.nn.depth_to_space(x, scale)

	def create_save_directories(self):
		"""Creates the save directories for generated OCT images.

		:return: self
		"""
		# Set the name for the SR generated OCT images directory
		self.gen_image_dir = op.join('..\\logs', self.name, 'sr_gen_images')

		if not op.exists(self.gen_image_dir):
			os.mkdir(self.gen_image_dir)
		return self

	def load_sr_model(self):
		"""Loads the latest super resolution model.

		:return: last super resolution model
		"""
		# Set the model directory
		model_dir = op.join('..\\logs', self.name, 'models')

		# Obtain the full list super resolution models
		sr_model_list = glob(op.join(model_dir, '*edsr*'))
		sr_model_list = [single_model.split('\\')[-1] for single_model in sr_model_list]

		# Check if there are any SR models
		if len(sr_model_list) == 0:
			raise NameError('No existing SR models.')
		else:
			# Obtain the full list of epochs
			epoch_list = [int(single_model.split('_')[-1].split('.')[0]) for single_model in sr_model_list]

			# Obtain the last epoch
			last_epoch = sorted(epoch_list, reverse=True)[0]

			# Obtain the index of the last SR model with the highest epoch
			last_sr_model_idx = [int(single_model.split('_')[-1].split('.')[0]) == int(last_epoch) for single_model in
								  sr_model_list]
			last_sr_model_idx = np.where(last_sr_model_idx)[0]

			# Obtain the last SR model with the highest epoch
			self.last_sr_model = sr_model_list[last_sr_model_idx[0]]

			# Check the condition against the condition as specified in initialization
			condition = self.last_sr_model.split('_')[0]
			assert self.condition == condition, 'Conditions are not the same.'

		# Load the SR model
		sr_model = load_model(op.join(model_dir, self.last_sr_model), custom_objects={'normalize': self.normalize,
																				 'denormalize': self.denormalize,
																				 'pixel_shuffle': self.pixel_shuffle,
																				 'tf': tf})
		return sr_model

	def generate_images(self, num_images, display=False, save=True):
		"""Generates the low and super resolution OCT images (128 x 128 & 512 x 512)

		:param num_images: Number of images to generate
		:param display: Specify to display the image
		:param save: Specify to save the image
		:return: None
		"""
		# Load the generator model and SR model
		gen_model = super().load_gen_model()
		sr_model = self.load_sr_model()

		print('Generating images using SR model: ', self.last_sr_model)

		for i in range(num_images):

			# Initialize the noise vector for prediction
			noise = tf.random.normal([1, 100])

			# Predict using the generator model
			gen_img = gen_model.predict(noise)

			# Convert to [0, 255]
			gen_img = gen_img + 1
			gen_img = gen_img / gen_img.max()
			gen_img = gen_img * 255

			# Predict using the SR model
			sr_img = sr_model.predict(gen_img)

			# Plot the images
			plt.figure(figsize=(20, 10))
			plt.subplot(1, 2, 1)
			plt.imshow(gen_img[0, :, :, 0], cmap='bone')
			plt.title('Low resolution', fontsize=28)
			plt.axis('off')
			plt.subplot(1, 2, 2)
			# Convert to [0, 255]
			sr_img = sr_img + np.abs(sr_img.min())
			sr_img = sr_img / sr_img.max()
			sr_img = sr_img * 255
			plt.imshow(sr_img[0, :, :, 0], cmap='bone')
			plt.title('Super resolution', fontsize=28)
			plt.axis('off')
			if save:
				plt.savefig(op.join(self.gen_image_dir, 'sr_gen_img_{num}.png'.format(num=str(i + 1))))
			if display:
				plt.show()
				plt.close(plt.gcf())
			else:
				plt.close(plt.gcf())
		pass

