"""
output.py - module to store BiLSTM-CRF model
"""

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, TimeDistributed, Dropout, Bidirectional, Dense
from tensorflow.keras.initializers import Constant

from .loss import CRF

class BiLSTM_CRF:
	"""
	BiLSTM-CRF class to initialize NER (Name-Entity-Recognition) model
	"""
	def __init__(self, max_len, n_tags, word2dix, embed_dim, n_words, embed_initializer = 'glorot_normal', embed_regularizer = 'l2', embed_layer = None, hidden_units = 64, activations = ['tanh', 'relu'], regularizer = 'l2', dropout = 0.1, mask_zero = True, trainable = True, pretrained = None):
		self.max_len = max_len
		self.n_tags = n_tags
		self.embed_layer = embed_layer
		self.hidden_units = hidden_units
		self.embed_dim = embed_dim
		self.n_words = n_words
		self.word2dix = word2dix
		self.embed_initializer = embed_initializer
		self.embed_regularizer = embed_regularizer
		self.activations = activations
		self.regularizer = regularizer
		self.dropout = dropout
		self.mask_zero = mask_zero
		self.trainable = trainable
		self.pretrained = pretrained

	def build_embed_layer(self):
		"""
		build_embed_layer - function to initialize Embedding layer either from pretrained word embeddings or inintialy
		"""
		# retrieve pretrained word embeddings
		if self.pretrained:
			embed_index = {}
			with open(self.pretrained) as file:
				for line in file:
					word, coefs = line.split(maxsplit = 1)
					coefs = np.fromstring(coefs, 'f', sep = ' ')
					embed_index[word] = coefs
			print("Found %s word vectors.".format(len(embed_index)))

			embeds  = np.zeros((self.n_words + 1, self.embed_dim))
			for word, i in self.word2dix.items():
				embed_vector = embed_index.get(word)
				if embed_vector is not None:
					embeds[i] = embed_vector
			self.embed_initializer = Constant(embeds)
			
		return Embedding(input_dim = self.n_words + 1, output_dim = self.embed_dim, input_length = self.max_len, mask_zero = self.mask_zero, trainable = self.trainable, embeddings_initializer = self.embed_initializer, embeddings_regularizer = self.embed_regularizer)

	def __call__(self):
		# input layer
		input = Input(shape = (self.max_len))
		
		# embed layer
		output = self.embed_layer(input) if self.embed_layer else self.build_embed_layer()(input)

		# bilstm-crf layer
		output = Bidirectional(
			LSTM(units = self.hidden_units, return_sequences = True, 
				activation = self.activations[0], activity_regularizer = self.regularizer,
				dropout = self.dropout))(output)
		# dense layer
		output = TimeDistributed(Dense(self.n_tags, activation = self.activations[1]))(output)

		# dense layer
		output = CRF(self.n_tags, name = 'CRF')(output)

		return Model(input, output)
