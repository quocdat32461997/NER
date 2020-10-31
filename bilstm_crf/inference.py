"""
inference.py - mnodule for BiLSTM-CRF inference
"""

# import os
import os
import numpy as np
import tensorflow as tf
from tensorflow.lookup import TextFileInitializer, TextFileIndex, StaticHashTable

from bilstm_crf.models import CRF

class NameEntityRecognizer:
	"""
	NameEntityRecognizer - class to identify name entities
	"""
	def __init__(self, model, tag_table, word_table):
		"""
		Class constructor 
		Inputs:
			- model : str or tf.keras.Model
				If not tf.keras.Model, have to load model itself
			- tag_table : str or Tensorflow lookup table
				Path to table or Tag lookup table
			- word_table : str or Tensorflow lookup table
				Path to table or Word lookup table
		"""

		# if model is string, load model from the given model path
		if isinstance(model, str):
			print("Loading model")
			self.model = tf.keras.models.load_model(model, custom_objects = {'crf_loss' : CRF.loss})
		else:
			self.model = model

		if isinstance(tag_table, str):
			self.tag_table = self._create_lookup_table(tag_table, type = 'idx2str', default = 'UNKNOWN')
		else:
			self.tag_table = tag_table

		if isinstance(word_table, str):
			self.word_table = self._create_lookup_table(word_table)
		else:
			self.word_table = word_table

	def _create_lookup_table(self, file, type = 'str2idx', default = 0):
		"""
		_create_lookup_table - function to create words/tags lookup table
		Inputs:
			- file : str
				Name/path to word/tag list file
			- type : str
				Either str2idx or idx2str
				If str2idx -> word -> index
				If idx2str -> index -> word
			- default : int
				Default value for missing value. By 0 by default
		Outputs:
			- table : Tensorflow lookup table
		"""

		key = {'type' : tf.string, 'index' : TextFileIndex.WHOLE}
		value = {'type' : tf.int64, 'index' : TextFileIndex.LINE_NUMBER}

		# tag_table should lookup index and convert to real tag
		if not type is 'str2idx':
			temp = key
			key = value
			value = temp

		initializer = TextFileInitializer(file, key_dtype = key['type'], key_index = key['index'], value_index = value['index'], value_dtype = value['type'], delimiter = '\n')

		table = StaticHashTable(initializer = initializer, default_value = default)
		return table

	def predict(self, input):
		"""
		predict - function to make predictions
		Input:
			- input : Tensor
				Preprocess input string Tensor in shape of [batch_size, sequence_length] or [sequence_length]
		Output:
			- predictions : Tensor
				Predicted tags. Tensor shape of [batch_size, sequence-length, number-tags]
		"""

		# reshape to [batch_size, sequence_length]
		if tf.shape(input) == 1:
			input = tf.expand_dims(input, axis = 0)

		# conver text to acceptd input format
		input = self._process_text(input)

		# make predictions
		predictions = self.model.predict(input)

		# convert to correct output format
		predictions = self.pred_to_tags(predictions)
		
		return predictions
	
	def pred_to_tags(input):
		"""
		pred_to_tags - function to convert BiLSTM-CRF prediction to tags
		Inputs:
			- input : Tensor
				Tensor of shape [batch_size, sequenc_length, number_tags] or [sequence_length, number_tags]
		Outputs:
			- output : list of tags
		"""

		# reshape input-shape to [batch-size, sequence-length, number-tags]
		if tf.shape(input) == 2: # input has only one sample
			input = tf.expand_dims(input, axis = 0)

		# convert to numpy for CPU-processing
		input = input.numpy()

		# find tag-index with highest probability
		output = np.argmax(input, axis = -1)

		return output

	def _process_text(self, text):
		"""
		_process_text - function to preprocess text for inference
		Inputs:
			- text : str
				Raw text input
		Outputs:
			- text : str
				Processed text
		"""

		text = process_text(text)

		return text
