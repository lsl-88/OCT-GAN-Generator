from abc import ABC, abstractmethod


class DatasetError(Exception):
	pass


class Dataset(ABC):
	"""
    Abstract base class representing a dataset. All datasets should subclass from this baseclass and need to implement
    the 'load' function. Initializing of the dataset and actually loading the data is separated as the latter may
    require significant time, depending on where the data is coming from.
    """
	def __init__(self, **kwargs):
		"""Initialize a dataset."""

	@abstractmethod
	def data_details(self):
		"""Obtain the data details.

        :return: self
        """
		return self

	@abstractmethod
	def load(self, **kwargs):
		"""Load the data list.

	    :param kwargs: Arbitrary keyword arguments
	    :return: self
	    """
		return self

	@property
	def print_stats(self):
		"""This function prints information on the dataset. This method uses the fields that need to be implemented
        when subclassing.
        """
		if self.condition == 'CNV':
			print('Condition: ', self.condition)
			print('Total images: ', self.total_imgs)

		elif self.condition == 'DME':
			print('Condition: ', self.condition)
			print('Total images: ', self.total_imgs)

		elif self.condition == 'Drusen':
			print('Condition: ', self.condition)
			print('Total images: ', self.total_imgs)
		pass
