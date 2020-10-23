"""
data.py - module for efficient data loading
"""

# import dependencies
import os
import tensorflow as tf
from tensorflow.data import TextLineDataset, Dataset

from .utils import process_text
class Dataset:
	"""
	Class Dataset to implement Tensorflow Dataset API for scalable training data pipeline
	"""
	def __init__(self, files, batch_size = 32, name = 'Dataset Loader'):
		"""
		Inputs:
			- files : str or list of str
				List of paths to text files
				Within each file, there are lines of text (sentence or paragraph)
			- batch_size : int
				Size of a batch of samples
		"""
		if isinstance(files, str): # convert to list of files if a single file only
			files = list(files)
		self.files = files
		self.batch_size = batch_size
		self.name = name

	def __call__(self):
		# read text data
		dataset = TextLineDataset(self.files)

		# preprocessing
		dataset = dataset.map(process_text)

		return dataset

