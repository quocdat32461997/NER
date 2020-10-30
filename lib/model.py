"""
output.py - module to store BiLSTM-CRF model
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, TimeDistributed, Dropout, Bidirectional, Dense
from tensorflow.keras.initializers import Constant
from tensorflow.lookup import KeyValueTensorInitializer, StaticHashTable

from .loss import CRF

class BiLSTM_CRF:
	"""
	BiLSTM-CRF class to initialize NER (Name-Entity-Recognition) model
	"""
	def __init__(self, max_len, n_tags, embed_dim, n_words = 0, 
		word_table = None, embed_initializer = 'glorot_normal',
		regularizers = ['l2', 'l2', 'l2'], embed_layer = None,
		hidden_units = 64, activations = ['tanh', 'relu'],
		regularizer = 'l2', dropout = 0.1, mask_zero = True,
		trainable = True, pretrained_embed = None):
		"""
		Inputs:
			max_len : int
				Max length of sentences
			n_tags : int
				Number of tags / output_dim
			embed_dim : int
				Size of Embedding layer
			n_words : int
				Number of words (optional for Pre-trained embeddings only)
			word_table : Tensorflow lookup table
				Lookup table for Word and Index
			embed_initializer : str
				Matrix initailizer. If pretrained is valid, then embed_initializer is retreived from Pre-trained embeddings
			regularizers : list of str
				List of regularizers (default: 'l2' for both Embedding, BiLSTM, and Dense layers)
			hidden_units : int
				Size of hidden units
			activations : list of str
				List of activations (default: tanh for LSTM and relu for Dense layer between BiLSTM and CRF
			dropout : float
				Dropout rate applicable for layer in need
			mask_zero : boolean
				Signal to ignore padding tokens
			trainable : boolean
				Initiali setting for layers' trainability
			pretrained_embed : str
				Default : None. Valid string uploads the pretrained embeddings

		Functions:
			- build_embed_layer : function to build Embedding layer
			- __call__ : build BiLSTM-CRF model
		Example:
			# initialize BiLSTM_CRF layer
			bilstm_crf = BiLSTM_CRF(max_len = 128, n_tags = 17, embed_dim = 50)
			# initialize BiLSTM_CRF model
			model = bilstm_crf()
			model.summary()
		"""
		self.max_len = max_len
		self.n_tags = n_tags
		self.embed_layer = embed_layer
		self.hidden_units = hidden_units
		self.embed_dim = embed_dim
		self.n_words = n_words if n_words else word_table.size()
		self.word_table = word_table
		self.embed_initializer = embed_initializer
		self.activations = activations
		self.regularizers = regularizers
		self.dropout = dropout
		self.mask_zero = mask_zero
		self.trainable = trainable
		self.pretrained_embed = pretrained_embed

	def build_embed_layer(self):
		"""
		build_embed_layer - function to initialize Embedding layer either from pretrained word embeddings or inintialy
		"""
		# retrieve pretrained word embeddings
		if self.pretrained_embed:
			# retrieve pretrained word embeddings
			embed_index = {}
			with open(self.pretrained_embed) as file:
				for line in file:
					word, coefs = line.split(maxsplit = 1)
					coefs = np.fromstring(coefs, 'f', sep = ' ')
					if len(coefs) != 100:
						continue
					embed_index[word] = coefs
			
			# initialize new word embedding matrices
			embeds = np.zeros((self.n_words + 1, self.embed_dim))
			# parse words to pretrained word embeddings
			words, indices = self.word_table.export()
			#for word, i in zip(tf.strings.as_string(words), indices.numpy()):
			for text, i in zip(words.numpy(), indices.numpy()):
				word = str(text)[2:-1] # get word from byte-class string
				embed_vector = embed_index.get(word)
				if embed_vector is not None:
					embeds[i] = embed_vector
			self.embed_initializer = Constant(embeds)
			
		return Embedding(input_dim = self.n_words + 1, output_dim = self.embed_dim, input_length = self.max_len, mask_zero = self.mask_zero, trainable = self.trainable, embeddings_initializer = self.embed_initializer, embeddings_regularizer = self.regularizers[0], mask_zero = self.mask_zero)

	def __call__(self):
		# input layer
		input = Input(shape = (self.max_len, ), batch_size = None)
		
		# embed layer
		output = self.embed_layer(input) if self.embed_layer else self.build_embed_layer()(input)

		# bilstm-crf layer
		output = Bidirectional(
			LSTM(units = self.hidden_units, return_sequences = True, 
				activation = self.activations[0], activity_regularizer = self.regularizers[1],
				dropout = self.dropout))(output)
		# dense layer
		output = TimeDistributed(Dense(self.n_tags, activation = self.activations[1], activity_regularizer = self.regularizers[-1]))(output)

		# dense layer
		output = CRF(self.n_tags, name = 'CRF')(output)

		return Model(input, output)
