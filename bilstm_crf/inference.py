"""
inference.py - mnodule for BiLSTM-CRF inference
"""

# import os
import os
import numpy as np
import tensorflow as tf

from bilstm_crf.models import CRF

class NameEntityRecognizer:
	"""
	NameEntityRecognizer - class to identify name entities
	"""
	def __init__(self, model, tags):
		"""
		Class constructor 
		Inputs:
			- model : str or tf.keras.Model
				If not tf.keras.Model, have to load model itself
			- tags : dictionary
				Dictionary of index-tag
		"""

		# if model is string, load model from the given model path
		if isinstance(model, str):
			print("Loading model")
			self.model = tf.keras.models.load_model(model, custom_objects = {'crf_loss' : CRF.loss})
		else:
			self.model = model

		self.tags = tags

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

		# make predictions
		predictions = self.model.predict(input)
		
		return predictions
	
	def pred_to_tags(input, tags):
		"""
		pred_to_tags - function to convert BiLSTM-CRF prediction to tags
		Inputs:
			- input : Tensor
				Tensor of shape [batch_size, sequenc_length, number_tags] or [sequence_length, number_tags]
			- tags : Python dictionary
				Dictionary of index-tag
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

