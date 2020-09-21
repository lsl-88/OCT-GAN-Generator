from Model import TensorflowModel

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation, Dropout, Flatten
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import os.path as op
import pandas as pd
import time
import skimage
import numpy as np


class DCGAN(TensorflowModel):
	model = 'DCGAN'

	def __init__(self, name, condition):
		super().__init__(name)
		self.condition = condition

		# Variables to store data
		self.generator = None
		self.discriminator = None
		self.generator_optimizer = None
		self.discriminator_optimizer = None
		self.summary_writer = None

		self.model_dir = None
		self.train_summary_dir = None
		self.image_dir = None
		self.create_save_directories()

	def create_save_directories(self):
		"""Creates the save directories for the saved models and training summary.

		:return: self
		"""
		# Set the name for the saved model, images, and training summary directory
		self.model_dir = op.join('../logs', self.name, 'models')
		self.train_summary_dir = op.join('../logs', self.name, 'training_summary')
		self.image_dir = op.join('../logs', self.name, 'images')

		if not op.exists(op.join('../logs', self.name)):
			if not op.exists('../logs'):
				os.mkdir('../logs')
			os.mkdir(op.join('../logs', self.name))

		if not op.exists(self.model_dir):
			os.mkdir(self.model_dir)

		if not op.exists(self.train_summary_dir):
			os.mkdir(self.train_summary_dir)

		if not op.exists(self.image_dir):
			os.mkdir(self.image_dir)
		return self

	def create_summary_writer(self):
		"""Creates the summary_writer.

		:return: self
		"""
		col_names = ['Epoch', 'Step', 'Gen_loss', 'Disc_loss']
		self.summary_writer = pd.DataFrame(columns=col_names)
		return self

	@staticmethod
	def generator_model():
		"""Full generator model.

		:return: generator_model
		"""
		# Input layer (latent dimension of 100)
		inputs = tf.keras.Input(shape=(100,))

		# Convolution layer 1
		layer_1 = Dense(units=4 * 4 * 1024, use_bias=False)(inputs)
		layer_1 = BatchNormalization()(layer_1)
		layer_1 = LeakyReLU(alpha=0.2)(layer_1)
		layer_1 = Reshape(target_shape=(4, 4, 1024))(layer_1)  # Output shape: (4, 4, 1024)

		# Convolution layer 2
		layer_2 = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(
			layer_1)
		layer_2 = BatchNormalization()(layer_2)
		layer_2 = LeakyReLU(alpha=0.2)(layer_2)  # Output shape: (8, 8, 512)

		# Convolution layer 3
		layer_3 = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(
			layer_2)
		layer_3 = BatchNormalization()(layer_3)
		layer_3 = LeakyReLU(alpha=0.2)(layer_3)  # Output shape: (16, 16, 256)

		# Convolution layer 4
		layer_4 = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(
			layer_3)
		layer_4 = BatchNormalization()(layer_4)
		layer_4 = LeakyReLU(alpha=0.2)(layer_4)  # Output shape: (32, 32, 128)

		# Convolution layer 5
		layer_5 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(
			layer_4)
		layer_5 = BatchNormalization()(layer_5)
		layer_5 = LeakyReLU(alpha=0.2)(layer_5)  # Output shape: (64, 64, 64)

		# Output layer
		output = Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False)(
			layer_5)
		output_image = Activation('tanh')(output)  # Output shape: (128, 128, 1)

		generator_model = tf.keras.Model(inputs, output_image)
		return generator_model

	@staticmethod
	def discriminator_model():
		"""Full discriminator model.

		:return: discriminator_model
		"""
		inputs = tf.keras.Input(shape=(128, 128, 1))

		# Convolution layer 1
		layer_1 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same')(inputs)
		layer_1 = LeakyReLU(alpha=0.2)(layer_1)
		layer_1 = Dropout(0.3)(layer_1)  # Output shape: (64, 64, 32)  # RF: 4

		# Convolution layer 2
		layer_2 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(layer_1)
		layer_2 = LeakyReLU(alpha=0.2)(layer_2)
		layer_2 = Dropout(0.3)(layer_2)  # Output shape: (32, 32, 64) # RF: 10

		# Convolution layer 3
		layer_3 = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(layer_2)
		layer_3 = LeakyReLU(alpha=0.2)(layer_3)
		layer_3 = Dropout(0.3)(layer_3)  # Output shape: (16, 16, 128) # RF: 22

		# Convolution layer 4
		layer_4 = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(layer_3)
		layer_4 = LeakyReLU(alpha=0.2)(layer_4)
		layer_4 = Dropout(0.3)(layer_4)  # Output shape: (8, 8, 256) # RF: 46

		# Convolution layer 5
		layer_5 = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(layer_4)
		layer_5 = LeakyReLU(alpha=0.2)(layer_5)
		layer_5 = Dropout(0.3)(layer_5)  # Output shape: (4, 4, 256) # RF: 70

		output = Flatten()(layer_5)
		output_layer = Dense(units=1, activation='linear')(output)

		discriminator_model = tf.keras.Model(inputs, output_layer)
		return discriminator_model

	@staticmethod
	def cross_entropy():
		return tf.keras.losses.BinaryCrossentropy(from_logits=True)

	def discriminator_loss(self, real_output, fake_output):
		"""Discriminator loss for discriminator model.

		:param real_output: Real image
		:param fake_output: Fake image
		:return: discriminator loss
		"""
		loss_func = self.cross_entropy()
		real_loss = loss_func(tf.ones_like(real_output), real_output)
		fake_loss = loss_func(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		return total_loss

	def generator_loss(self, fake_output):
		"""Generator loss for generator model.

		:param fake_output: Fake image
		:return: generator loss
		"""
		loss_func = self.cross_entropy()
		return loss_func(tf.ones_like(fake_output), fake_output)

	def create_model(self):
		"""Creates the generator, discriminator, and summary writer.

		:return: self
		"""
		# Creates the generator and discriminator
		self.generator = self.generator_model()
		self.discriminator = self.discriminator_model()

		# Creates the summary writer
		self.create_summary_writer()

		# Initialize the optimizers
		self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
		self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
		print('Models are created.')
		return self

	def fit(self, train_data, batch_size, epochs=100, save_model=1):
		"""Perform training for the full epochs.

		:param train_data: The full train data list.
		:param batch_size: -
		:param epochs: -
		:param save_model: Specify the model to save per epoch.
		:return: self
		"""
		assert self.generator is not None or self.discriminator is not None, 'Create the models first.'

		# Seed to generate images
		seed = tf.random.normal([4, 100])

		# Perform training to the full epochs
		for epoch in range(epochs):

			# Initialize the start time
			start = time.time()
			print('\nEpoch: ' + str(epoch + 1) + '\n')

			# Shuffle the train data
			train_data = shuffle(train_data)

			# Number of step
			num_steps = len(train_data)

			# Loop through all the training data
			for step in range(0, num_steps, batch_size):

				# Select the label map and CT data
				X_train = train_data[step:step + batch_size]

				# Initialize empty array for batch images
				image_batch = np.array([]).reshape(0, 128, 128, 1)

				# Read the images and stack the images into a batch
				for single_file in X_train:
					single_img = skimage.io.imread(single_file)
					single_img = skimage.transform.resize(image=single_img, output_shape=(128, 128), anti_aliasing=True)
					single_img = (single_img - (1 / 2)) / (1 / 2)

					single_img = single_img[np.newaxis, :, :, np.newaxis]  # Reshape to (batch_size, ht, wt, ch)

					image_batch = np.concatenate((image_batch, single_img), axis=0)

				# Convert to tensors
				image_batch = tf.cast(image_batch, dtype=tf.float32)

				self.train_step(image_batch, batch_size, epoch, step)

			# Produce images for the GIF as we go
			self.generate_and_save_images(self.generator, epoch+1, seed)

			# Save the checkpoint and the model
			if (epoch + 1) % save_model == 0:
				# Save the generator model
				self.save_model(epoch)

			# Create new summary writer
			self.create_summary_writer()

			# Initialize the end time
			end = time.time()
			print('\nTime taken for epoch {} is {:.2f} sec\n'.format(epoch + 1, end - start))

		# Save the model at the end of training
		self.save_model(epoch)
		return self

	def train_step(self, image_batch, batch_size, epoch, step):
		"""Perform training for single step.

		:param image_batch: Image batch array
		:param batch_size: -
		:param epoch: -
		:param step: -
		:return: self
		"""
		# Initialize empty dictionary
		summary_dict = {}

		noise = tf.random.normal([batch_size, 100])

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_images = self.generator(noise, training=True)

			real_output = self.discriminator(image_batch, training=True)
			fake_output = self.discriminator(generated_images, training=True)

			gen_loss = self.generator_loss(fake_output)
			disc_loss = self.discriminator_loss(real_output, fake_output)

		gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
		disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

		self.generator_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
		self.discriminator_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

		print('Epoch:{epoch}, Step:{step}, Disc Loss:{d_loss:.3f}, Gen Loss:{g_loss:.3f}'.
			  format(epoch=epoch+1, step=step+1, d_loss=disc_loss.numpy(), g_loss=gen_loss.numpy()))

		# Store the loss into the dictionary
		summary_dict['Epoch'] = epoch + 1
		summary_dict['Step'] = step + 1
		summary_dict['Gen_loss'] = gen_loss.numpy()
		summary_dict['Disc_loss'] = disc_loss.numpy()
		self.summary_writer = self.summary_writer.append(summary_dict, ignore_index=True)
		return self

	def save_model(self, epoch):
		"""Saves the generator model.

		:param epoch: The epoch to save
		:return: self
		"""
		# Set the name for the model
		gen_filename = '{}_dcgan_gen_model_epoch_{}.h5'.format(self.condition, epoch + 1)
		disc_filename = '{}_dcgan_disc_model_epoch_{}.h5'.format(self.condition, epoch + 1)
		train_summary_filename = '{}_train_summary_epoch_{}.csv'.format(self.condition, epoch + 1)

		# Save the model and train summary
		self.generator.save(op.join(self.model_dir, gen_filename), include_optimizer=True)
		self.discriminator.save(op.join(self.model_dir, disc_filename), include_optimizer=True)
		self.summary_writer.to_csv(op.join(self.train_summary_dir, train_summary_filename))
		return self

	def generate_and_save_images(self, model, epoch, test_input):
		"""Generate image plot at end of epoch

		:param model: generator model
		:param epoch: at each epoch
		:param test_input: the seed
		:return: None
		"""
		# 'training' is set to False so all layers run in inference mode (batchnorm).
		predictions = model(test_input, training=False)

		# Plot the images
		plt.figure(figsize=(10, 10))
		for i in range(predictions.shape[0]):
			plt.subplot(2, 2, i+1)
			plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
			plt.axis('off')

		plt.savefig(op.join(self.image_dir, '{}_images_epoch_{:04d}.png'.format(self.condition, epoch)))
		plt.close(plt.gcf())
		pass
