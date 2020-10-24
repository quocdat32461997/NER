"""
data.py - module for efficient data loading
"""

# import dependencies
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.data import TextLineDataset, Dataset

from .utils import process_text

class Dataset:
	"""
	Class Dataset to implement Tensorflow Dataset API for scalable training data pipeline
	"""
	def __init__(self, texts, targets, max_len, batch_size = 32, shuffle = True, buffer_size = None, seed = 1997, name = 'Dataset Loader'):
		"""
		Inputs:
			- texts : str or list of str
				List of paths to text files
				Within each file, there are lines of text (sentence or paragraph)
			- targets : str or list of str
				List of paths to label files
			- max_len : int
				Largest sequenceh length
			- batch_size : int
				Size of a batch of samples
			- shuffle : boolean
				Boolean value to shuffle data
			- buffer_size : int
				By default, None. buffer size to shuffle
			- seed : int
				Random seed
		"""
		if isinstance(texts, str): # convert to list of files if a single file only
			texts = list(text)
		self.texts = texts
		if isinstance(targets, str): # convert to list of files if a single file only
			targets = list(targets)
		self.targets = targets
		self.max_len = max_len
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.seed = seed
		self.name = name

		# buffer_size for shuffling is set to triple the batch_size
		if not buffer_size:
			self.buffer_size = batch_size * 3
		else:
			self.buffer_size = buffer_size 

	def _process_text(self, texts, targets):
		"""
		_process_text - function to process text
		
		Inputs:
			- input : TF Dataset object
		Outputs:
			- input : TF Dataset object
		"""
		

		# processing
		dataset = [data.map(process_text) for data in [texts, targets]]

		# shuffling
		if self.shuffle:
			dataset = [data.shuffle(buffer_size = self.buffer_size, seed = self.seed) for data in dataset]

		# concatenate texts and targets
		dataset = tf.data.Dataset.zip(tuple(dataset))

		# padding and truncating
		padded_shapes = (tf.TensorShape([None]), tf.TensorShape([None])) # unknown length of texts and targets
		padding_values = (0, 0) # 0 and 'O' for padding values
		dataset = dataset.padded_batch(
			batch_size = self.batch_size,
			padded_shapes = padded_shapes,
			padding_values = padding_values)

		return dataset

	def __call__(self):
		# read text data
		texts = TextLineDataset(self.texts)
		targets = TextLineDataset(self.targets)

		# process text data
		dataset = self._process_text(texts, targets)

		return dataset

