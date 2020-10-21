"""
data.py - module for efficient data loading
"""

# import dependencies
import os
import tensorflow as tf
from tensorflow.data import TextLineDataset, Dataset


class Dataset:
	"""
	Class Dataset to implement Tensorflow Dataset API for scalable data pipeline
	"""
	def __init__(self, files, name = 'Dataset Loader'):
		self.files = files
		self.name = name

	def __call__(self):
		return None

