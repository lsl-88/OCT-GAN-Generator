from Model import TensorflowModel

import os
import os.path as op
import math
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle


class EDSR(TensorflowModel):
	model = 'EDSR'

	def __init__(self, name, condition):
		super().__init__(name)
		self.condition = condition

		# Variables to store data
		self.edsr_model = None
		self.callbacks_list = None

		self.model_dir = None
		self.train_summary_dir = None
		self.image_dir = None
		self.create_save_directories()

	def create_save_directories(self):
		"""Creates the save directories for the saved models and training summary.

		:return: self
		"""
		# Set the name for the saved model and training summary directory
		self.model_dir = op.join('..\\logs', self.name, 'models')
		self.train_summary_dir = op.join('..\\logs', self.name, 'training_summary')

		if not op.exists(op.join('../logs', self.name)):
			if not op.exists('../logs'):
				os.mkdir('../logs')
			os.mkdir(op.join('../logs', self.name))

		if not op.exists(self.model_dir):
			os.mkdir(self.model_dir)

		if not op.exists(self.train_summary_dir):
			os.mkdir(self.train_summary_dir)
		return self

	def edsr(self, num_filters=32, num_res_blocks=8):
		"""Full EDSR model.

		:return: EDSR model
		"""
		x_input = Input(shape=(None, None, 1))

		# Normalize the input image
		x = Lambda(self.normalize)(x_input)

		# Res block
		x = b = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', name='conv_tester')(x)
		for i in range(num_res_blocks):
			b = self.res_block(b, num_filters)
		b = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same')(b)
		x = Add()([x, b])

		# Upsample block
		x = self.upsample(x, num_filters)
		x = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)

		# Denormalize the output layer
		x = Lambda(self.denormalize)(x)

		# Set the optimizer and compile the model
		model = Model(inputs=x_input, outputs=x, name='edsr')
		optimizer = Adam(learning_rate=5e-5)
		model.compile(optimizer=optimizer, loss='mean_absolute_error')
		return model

	def res_block(self, x_input, filters):
		"""Residual block for EDSR model.

		:param x_input: Res block inputs
		:param filters: Number of filters
		:return: Res block outputs
		"""
		x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(x_input)
		x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
		x = Add()([x_input, x])
		return x

	def upsample(self, x, filters):
		"""Upsample block for EDSR model.

		:param x: Res block inputs
		:param filters: Number of filters
		:return: Res block outputs
		"""
		x = Conv2D(filters * (2 ** 2), 3, padding='same', name='conv2d_1_scale_2')(x)
		x = Lambda(self.pixel_shuffle(scale=2))(x)
		x = Conv2D(filters * (2 ** 2), 3, padding='same', name='conv2d_2_scale_2')(x)
		x = Lambda(self.pixel_shuffle(scale=2))(x)
		return x

	@staticmethod
	def pixel_shuffle(scale):
		return lambda x: tf.nn.depth_to_space(x, scale)

	@staticmethod
	def compute_mean(x):
		return tf.math.reduce_mean(x)

	def normalize(self, x):
		return (x - self.compute_mean(x)) / tf.math.reduce_std(x)

	def denormalize(self, x):
		return x * tf.math.reduce_std(x) + self.compute_mean(x)

	def get_callbacks(self):
		"""Creates the callback lists

		:return: callbacks list
		"""
		# Initialize empty callback list
		callbacks_list = []

		# Save the model
		model_filename = self.condition + '_edsr_model_epoch_{epoch:02d}.h5'
		checkpoint = ModelCheckpoint(op.join(self.model_dir, model_filename), monitor='val_loss', verbose=0,
									 save_best_only=True, mode='min', save_weights_only=False)
		callbacks_list.append(checkpoint)

		# Save the logs
		train_summary_filename = 'EDSR_train_summary.csv'
		csv_logger = CSVLogger(op.join(self.train_summary_dir, train_summary_filename), append=True, separator=';')
		callbacks_list.append(csv_logger)
		return callbacks_list

	def create_model(self):
		"""Creates the EDSR model and the callbacks list.

		:return: self
		"""
		# Creates the EDSR model
		self.edsr_model = self.edsr()
		self.callbacks_list = self.get_callbacks()

		print('Models are created.')
		return self

	def save_model(self):
		"""No need to implement this method."""
		pass

	def fit(self, dataset, batch_size, epochs, split_ratio=0.7):
		"""Perform training for the full epochs.

		:param dataset: The full dataset list.
		:param batch_size: -
		:param epochs: -
		:param split_ratio: Ratio to split the dataset into train and validation dataset (Default is 0.7)
		:return: self
		"""
		# Set the train len based on split ratio
		train_len = int(len(dataset) * split_ratio)

		# Initialize the train and validation generator
		train_generator = DataGenerator(x_set=dataset[:train_len], batch_size=batch_size, name=self.name, condition=self.condition)
		val_generator = DataGenerator(x_set=dataset[train_len:], batch_size=batch_size, name=self.name, condition=self.condition)

		self.edsr_model.fit(x=train_generator, epochs=epochs, validation_data=val_generator, callbacks=self.callbacks_list,
							max_queue_size=1, use_multiprocessing=False, workers=1, steps_per_epoch=len(train_generator), verbose=1)
		return self


class DataGenerator(tf.keras.utils.Sequence):
	"""Data generator for training"""
	def __init__(self, x_set, batch_size, name, condition):

		self.x = x_set
		self.batch_size = batch_size

		self.name = name
		self.condition = condition
		self.root_dir = '..\\Processed_data'
		self.dim_x, self.dim_y = 96, 96

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

		# Initialize empty arrays
		x_batch = np.array([]).reshape(0, 96, 96, 1)
		x_batch_resized = np.array([]).reshape(0, 24, 24, 1)

		total_pixels = self.dim_x * self.dim_x
		percentage = 0.85

		# Iterate through the full batch
		for single_x in batch_x:

			# Read single image
			X = cv2.imread(single_x, flags=cv2.IMREAD_GRAYSCALE)

			# Obtain the index where there are non zero pixel values
			idx = np.where(X != X.min())

			# Shuffle the index
			idx_x, idx_y = shuffle(idx[0], idx[1])

			for i, j in zip(idx_x, idx_y):

				# Crop the image to (96, 96)
				X_crop = X[i:i + self.dim_x, j:j + self.dim_y]

				# Obtain the number of pixels that are not background
				num_pixels = len(X_crop[X_crop != X_crop.min()])

				# Obtain the shape of cropped image
				X_crop_shape = X_crop.shape

				# Select only if there are OCT pixels in it
				if num_pixels >= (percentage * total_pixels) and X_crop_shape == (96, 96):
					X_resized_crop = cv2.resize(src=X_crop, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
					break

			# Concatenate to the empty array
			x_batch = np.concatenate((x_batch, X_crop[np.newaxis, :, :, np.newaxis]), axis=0)
			x_batch_resized = np.concatenate((x_batch_resized, X_resized_crop[np.newaxis, :, :, np.newaxis]), axis=0)

		# Cast to tensorflow floats
		X = tf.cast(x_batch, dtype=tf.float32)
		X_resized = tf.cast(x_batch_resized, dtype=tf.float32)
		return X_resized, X
