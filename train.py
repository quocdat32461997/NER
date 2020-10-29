"""
train.py - module to implement Training functioanlity
"""

# import dependencies
import os
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.compat.v1.enable_eager_execution()

from lib import *

def main():

	"""
	initialize Data pipeline
	"""
	# define paths to text data and lookup tables
	texts = ['data/ner_text_dataset.txt', 'data/wnut17train_conll_text.txt']
	targets = ['data/ner_label_dataset.txt', 'data/wnut17train_conll_label.txt']

	word_table_path = './data/words.txt'
	tag_table_path = './data/tags.txt'

	data_pipeline = Dataset(texts = texts, targets = targets,
		word_table = word_table_path, tag_table = tag_table_path)
	dataset = data_pipeline()

	print("Testing Training Data Pipeline")
	for data in dataset:
		texts, targets = data
		print("Texts shape", texts.shape)
		print("Targets shape", targets.shape)
		break

	"""
	define BiLSTM-CRF model
	"""
	# define parameters
	pretrained_embed = '/Users/datqngo/Desktop/projects/BiLSTM-CRF/glove.twitter.27B/glove.twitter.27B.100d.txt'
	hidden_units = 150
	embed_dim = 100
	max_len = None

	# initialie BiLSTM-CRF class object
	bilstm_crf = BiLSTM_CRF(max_len = max_len, embed_dim = embed_dim, n_tags = data_pipeline.tag_table.size(), word_table = data_pipeline.word_table, hidden_units = hidden_units, pretrained_embed = pretrained_embed)
	model = bilstm_crf()

	# compile model
	LR = 0.001
	optimizer = Adam(learning_rate = LR)
	loss = model.layers[-1].loss
	metrics = [model.layers[-1].accuracy]
	model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

	# inspect data architecture
	print(model.summary())

	"""
	Training
	"""

	# Step 1: freeze Embedding layer for stable-loss training
	for layer in model.layers:
		if layer.name == 'Embedding':
			model.layers[layer.name] = False

	EPOCHS = 20
	SHUFFLE = True
	WORKERS = 4
	QUEUE_SIZE = 10
	STEPS = None
	CALLBACKS = []
	model.fit(dataset, epochs = EPOCHS, verbose = 1, callbacks = CALLBACKS, shuffle = SHUFFLE, steps_per_epoch = STEPS, max_queue_size = QUEUE_SIZE, workers = WORKERS, use_multiprocessing = True)

if __name__ == '__main__':
	main()
