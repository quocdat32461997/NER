"""
train.py - module to implement Training functioanlity
"""

# import dependencies
import os
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

tf.compat.v1.enable_eager_execution()

from lib import *

def main():

	"""
	initialize Data pipeline
	"""
	# define paths to text data and lookup tables
	train_texts = ['data/ner_train_text.txt', 'data/wnut17train_conll_train_text.txt']
	train_targets = ['data/ner_train_label.txt', 'data/wnut17train_conll_val_label.txt']

	val_texts = ['data/ner_val_text.txt', 'data/wnut17train_conll_val_text.txt']
	val_targets = ['data/ner_val_label.txt', 'data/wnut17train_conll_val_label.txt']

	word_table_path = './data/words.txt'
	tag_table_path = './data/tags.txt'

	data_pipeline = Dataset(texts = train_texts, targets = train_targets, val_texts = val_texts, val_targets = val_targets, word_table = word_table_path, tag_table = tag_table_path)
	train_dataset, val_dataset = data_pipeline()


	print("Testing Data Pipeline")
	for train, val in zip(train_dataset, val_dataset):
		txt, labels = train
		val_txt, val_labels = val
		print("Texts shape: train {} and val {}", txt.shape, val_txt.shape)
		print("Targets shape: train {} and val {}", labels.shape, val_labels.shape)
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

	# define callbacks
	log_dir = 'logs'
	logging = TensorBoard(log_dir = log_dir, write_graph = True, write_images = True)
	early_stopping = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1)
	lr_reduce = ReduceLROnPlateau(monitor = 'loss', patience = 5, verbose = 1)
	CALLBACKS = [logging, early_stopping, lr_reduce]
	 #model.fit(dataset, epochs = EPOCHS, verbose = 1, callbacks = CALLBACKS, shuffle = SHUFFLE, steps_per_epoch = STEPS, max_queue_size = QUEUE_SIZE, workers = WORKERS, use_multiprocessing = True)

if __name__ == '__main__':
	main()
