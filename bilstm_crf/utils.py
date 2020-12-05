"""	
utils.py - module to implement utils for BiLSTM-CRF
"""

# import dependencies
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def process_text(input, word_table, training = True):
	"""
	Function process_text to clean text before Training process:
		- lowercase
		- tokenize

	--Notice--
		Punctuation stripping is not done here the dgiven dataset is clean. Also, punctuation is also considered as features as well.
	Inputs:
		- input : Tensor
			A single text line of sequence of words-tags
		- word_table : Tensorflow table lookup
			Table lookup to convert word to index
		- training : Boolean
			Inference: return tokens and encoded tokens
			Training: return encodedo tokens
	Outputs:
		- input : Tensor
			Tensor of shape [sequence_length]
	"""
	print(input)
	# lowercase string
	input = tf.strings.lower(input)
	print(input)
	# tokenize 
	tokens = tf.strings.split(input)
	print(tokens)
	# convert string to integer
	output = word_table.lookup(tokens)

	if training:
		return output
	else:
		return tokens, output

def process_target(inputs, tag_table):
	"""
	Function process_target to clean/process target for Training process:
		- tokenize
		- convert to full categoricals
	Inputs:
		- inputs : tensor
			A single text line of sequences of words-tags
		- tag_table : Tensor lookup table
			Table lookup to convert tag to index
	Outputs:
		- input: tensor
			Tensor of shape [sequence_length, n_tags]
	"""

	# tokenize
	inputs = tf.strings.split(inputs)

	# convert string to integer
	inputs = tag_table.lookup(inputs)

	# expand targets to number of tags
	inputs = tf.one_hot(inputs, depth = tf.cast(tag_table.size(), dtype = tf.int32), dtype = tf.int64)
	return inputs

class SentenceGetter(Sequence):
	"""
	Inheritted class from tf.keras.utils.Sequence class to efficiently load
	data to Tensorflow/Keras model
	"""
	def __init__(self, data, words, tags, maxlen, batch_size = 16, shuffle = False):
		"""
		__init__ - initializer for SentenceGetter class
		Inputs:
			- data : String or Pandas DataFrame object
				Path string to file or dataframe object
			- words : set
				Set of distinct words
			- tags : set
				Set of distinct tags
		"""

		if isinstance(data, str):
			# load data from Pandas file path
			data = pd.read_csv(data, encoding = 'latin1')
		elif isinstance(data, pd.DataFrame):
			# load data from Pandas DataFrame
			data = data
		else:
			raise Exception('Data is None or not found')
		self.word2dix = {w : i + 1 for i, w in enumerate(words)}
		self.tag2dix = {t : i for i, t in enumerate(tags)}
		self.n_tags = len(tags)
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.maxlen = maxlen
		n_sent = 1
		self.grouped = data.groupby('Sentence #').apply(self.agg_func)\
			.reset_index().rename(columns = {0 : 'sentence'})['sentence']
		self.sentences = [s for s in self.grouped]

	def agg_func(self, input):
		"""
		agg_func - function to group words/tags of sentences together
		"""
		print(input)
		return [(w, p, t) for w, p, t in zip(input['Word'].values.tolist(),
			input['POS'].values.tolist(), input['Tag'].values.tolist())]

	def pad_sentences(self, input):
		input = [[self.word2dix[w[0]] for w in s] for s in input]
		return pad_sequences(maxlen = self.maxlen, sequences = input, padding = 'post', value = 0)
    
	def generate_labels(self, input):
		input = [[self.tag2dix[w[2]] for w in s] for s in input]
		input = pad_sequences(maxlen = self.maxlen, sequences = input, padding = 'post', value = self.tag2dix['O'])
		
		return np.array([to_categorical(x, num_classes = self.n_tags) for x in input])
	def __len__(self):
		"""
		__len__ - function to compute length of SentenceGetter
		"""
		return int(self.grouped.shape[0] // self.batch_size)

	def __getitem__(self, index):
		if index == 0 and self.shuffle:
		# shuffle dataset for every iteration
			self.grouped = shuffle(self.grouped).reset_index()['sentence']

		# get batch of data
		sentences = self.grouped[index * self.batch_size : (index + 1) * self.batch_size]
        
		# generate sentences and labels
		labels = self.generate_labels(sentences)
		sentences = self.pad_sentences(sentences)
        
		return sentences, labels
