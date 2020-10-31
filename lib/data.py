"""
data.py - module for efficient data loading
"""

# import dependencies
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.data import TextLineDataset, Dataset
from tensorflow.lookup import TextFileInitializer, TextFileIndex, StaticHashTable, StaticVocabularyTable

from .utils import process_text, process_target

class Dataset:
	"""
	Class Dataset to implement Tensorflow Dataset API for scalable training data pipeline
	"""
	def __init__(self, texts, targets, val_texts = None, val_targets = None, word_table = None, tag_table = None, batch_size = 16, shuffle = True, buffer_size = 239, seed = 1997, threads = 4, prefetch = 1, name = 'Dataset Loader'):
		"""
		Inputs:
			- texts : str or list of str
				List of paths to text files
				Within each file, there are lines of text (sentence or paragraph)
			- targets : str or list of str
				List of paths to lael files
			
			- val_texts : str or list of str
                                List of paths to text files
                                Within each file, there are lines of text (sentence or paragraph)
                        - val_targets : str or list of str
                                List of paths to label files
			- word_table : str 
				Text file storing list of words
			- tag_table : str 
				Text file storing list of tags
			- batch_size : int
				Size of a batch of samples
			- shuffle : boolean
				Boolean value to shuffle data
			- buffer_size : int
				By default, None. buffer size to shuffle
			- seed : int
				Random seed
			- threads : int
				Number of threads
			- prefetech : int
				Number of prefetch
			- name : str
				Data loader class
		"""

		# dataset
		if isinstance(texts, str): # convert to list of files if a single file only
			texts = list(text)
		if isinstance(val_texts, str):
			val_texts = list(val_texts)
		self.texts = texts
		self.val_texts = val_texts
		if isinstance(targets, str): # convert to list of files if a single file only
			targets = list(targets)
		if isinstance(val_targets, str):
			val_targets = list(val_targets)
		self.targets = targets
		self.val_targets = val_targets

		# parameters
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.seed = seed
		self.threads = threads
		self.prefetch = prefetch
		self.name = name

		# retrieve word and tag table
		assert type(word_table) == str, "Word table must be string type and not None"
		self.word_table = self.create_lookup_table(word_table)

		assert type(tag_table) == str, "Tag table must be string type and not None"
		self.tag_table = self.create_lookup_table(tag_table)

		# buffer_size for shuffling is set to triple the batch_size
		if not buffer_size:
			self.buffer_size = batch_size * 3
		else:
			self.buffer_size = buffer_size 

	def create_lookup_table(self, file, default = 0):
		"""
		_create_lookup_table - fucntion to createa word/tag lookup table
		Inputs:
			 - file : str
				Name/path to word/tag list file
			- default : int
				Default value for missing value. Be -1 by default

		Outputs:
			- table : Tensorflow lookup table
		"""
		initializer = TextFileInitializer(file, key_dtype = tf.string, key_index = TextFileIndex.WHOLE_LINE,
			value_dtype = tf.int64, value_index = TextFileIndex.LINE_NUMBER, delimiter = '\n')
		
		table = StaticHashTable(initializer = initializer, default_value = default)

		return table

	def _process(self, texts, targets):
		"""
		_process - function to process text
		
		Inputs:
			- texts : TF TextLineDataset object
			- targets : TF TextLineDataset object
		Outputs:
			- dataset : TF Dataset object
				Processed and concatenated dataset object
		"""
		
		# processing
		texts = texts.map(lambda sent: process_text(sent, self.word_table), num_parallel_calls = self.threads)
		targets = targets.map(lambda sent: process_target(inputs = sent, tag_table = self.tag_table), num_parallel_calls = self.threads)

		# concatnenate
		dataset = tf.data.Dataset.zip((texts, targets))

		# shuffling
		if self.shuffle:
			dataset = dataset.shuffle(buffer_size = self.buffer_size, seed = self.seed)

		# padding and truncating
		padded_shapes = (tf.TensorShape([None]), tf.TensorShape([None, None])) # unknown length of texts and targets
		padding_values = (tf.constant(0, dtype = tf.int64), tf.constant(0, dtype = tf.int64)) #vocab_size and tag_size for padding values
		dataset = dataset.padded_batch(
			batch_size = self.batch_size,
			padded_shapes = padded_shapes,
			padding_values = padding_values)

		# prefetch
		dataset = dataset.prefetch(self.prefetch)

		return dataset

	def __call__(self):
		# read text data
		texts = TextLineDataset(self.texts)
		targets = TextLineDataset(self.targets)

		# process text data
		dataset = self._process(texts, targets)

		if self.val_texts and self.val_targets:
			# read text data
			val_texts = TextLineDataset(self.val_texts)
			val_targets = TextLineDataset(self.val_targets)

			# process text data
			val_dataset = self._process(val_texts, val_targets)

			return dataset, val_dataset

		else:
			return dataset

