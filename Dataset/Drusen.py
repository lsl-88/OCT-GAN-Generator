from Dataset import Dataset
import numpy as np
from glob import glob
import os
import os.path as op


class Drusen(Dataset):
	"""Dataset class for Drusen"""
	condition = 'Drusen'

	def __init__(self, data_dir=None, **kwargs):
		super().__init__(**kwargs)
		# Set the data directory
		if data_dir is None:
			if data_dir.split('\\')[-1] != self.__class__.condition:
				self.data_dir = op.join('/home/ubuntu/sl_root/Data/', self.__class__.condition)
			else:
				self.data_dir = '/home/ubuntu/sl_root/Data/Drusen'
		else:
			if data_dir.split('\\')[-1] != self.__class__.condition:
				self.data_dir = op.join(data_dir, self.__class__.condition)
			else:
				self.data_dir = data_dir

		# Variables to store data
		self.data_id = None
		self.total_imgs = None

		# Obtain the full data ID list
		self.data_details()

	def data_details(self):
		"""Obtain the data details.

		:return: self
		"""
		# Obtain the full Drusen files
		data_files = glob(op.join(self.data_dir, '*'))

		# Obtain the data ID and total images
		data_id = [single_file.split('\\')[-1].split('-')[1] for single_file in data_files]
		self.total_imgs = len(data_id)
		self.data_id = np.unique(data_id)
		return self

	def load(self):
		"""Load the data files.

		:return: Drusen data files
		"""
		# Initialize empty list
		data_files = []

		# Append the Drusen files to the list
		for single_file in os.listdir(self.data_dir):
			data_files.append(single_file)
		return data_files
