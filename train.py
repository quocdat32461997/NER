"""
train.py - module to implement Training functioanlity
"""

# import dependencies
import os
import argparse
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from bilstm_crf.data import Dataset
from bilstm_crf.models import BiLSTM_CRF

def main():

	"""
	initialize Data pipeline
	"""
	# define paths to text data and lookup tables
	train_texts = ['/resources/data/ner_train_text.txt', './resources/data/wnut17train_conll_train_text.txt']
	train_targets = ['./resources/data/ner_train_label.txt', './resources/data/wnut17train_conll_train_label.txt']

	val_texts = ['./resources/data/ner_val_text.txt', './resources/data/wnut17train_conll_val_text.txt']
	val_targets = ['/resources/data/ner_val_label.txt', './resources/data/wnut17train_conll_val_label.txt']

	word_table_path = './resources/data/words.txt'
	tag_table_path = './resources/data/tags.txt'

	BATCH_SIZE = 32

	data_pipeline = Dataset(texts = train_texts, targets = train_targets, val_texts = val_texts, val_targets = val_targets, word_table = word_table_path, tag_table = tag_table_path, batch_size = BATCH_SIZE)
	train_dataset, val_dataset = data_pipeline()


	print("Testing Data Pipeline")
	for train, val in zip(train_dataset, val_dataset):
		txt, labels = train
		val_txt, val_labels = val
		print("Texts shape: train {} and val {}".format(txt.shape, val_txt.shape))
		print(data_pipeline.word_table.size(), data_pipeline.tag_table.size())
		print("Targets shape: train {} and val {}".format(labels.shape, val_labels.shape))
		print()
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
	bilstm_crf = BiLSTM_CRF(max_len = max_len, embed_dim = embed_dim, n_tags = data_pipeline.tag_table.size(), word_table = data_pipeline.word_table.export(), hidden_units = hidden_units, pretrained_embed = pretrained_embed)
	model = bilstm_crf()

	# compile model
	LR = 0.001
	optimizer = Adam(learning_rate = LR)
	loss = model.layers[-1].loss
	model.compile(optimizer = optimizer, loss = loss)

	"""
	Training
	"""

	# define callbacks
	log_dir = 'logs'
	logging = TensorBoard(log_dir = log_dir, write_graph = True, write_images = True)
	checkpoints = ModelCheckpoint(filepath = 'logs', save_weight_only = True, verbose = 1)
	early_stopping = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1)
	lr_reduce = ReduceLROnPlateau(monitor = 'loss', patience = 5, verbose = 1)
	CALLBACKS = [logging, early_stopping, lr_reduce, checkpoints]

	QUEUE_SIZE = 10
	WORKERS = 4

	# Step 1: freeze Embedding layer for stable-loss training
	print("Phase-1 training: stable loss")
	for idx, layer in zip(range(len(model.layers)), model.layers):
		print(layer.name)
		if layer.name == 'embedding':
			model.layers[idx].trainable = False
	print("Inspect trainable parameters in Phase 1:", model.summary())

	EPOCHS = 20
	SHUFFLE = True
	STEPS = None # entire dataset

	model.fit(train_dataset, epochs = EPOCHS, verbose = 1, callbacks = CALLBACKS, shuffle = SHUFFLE, steps_per_epoch = STEPS, max_queue_size = QUEUE_SIZE, workers = WORKERS, use_multiprocessing = True)

	# Step 2: unfreeze all layers for full-model training
	print("Phase-2 training: full-fine-tuning")
	for idx, layer in zip(range(len(model.layers)), model.layers):
		print(layer.name)
		model.layers[idx].trainable = True
	print("Inspect trainable parameters in Phase 2:", model.summary())

	EPOCHS = 150
	SHUFFLE = True
	STEPS = 512 # by calculation, num_smaples // batch_size ~= 2396
	
	model.fit(train_dataset, validation_data = val_dataset, epochs = EPOCHS, verbose = 1, callbacks = CALLBACKS, shuffle = SHUFFLE, steps_per_epoch = STEPS, max_queue_size = QUEUE_SIZE, workers = WORKERS, use_multiprocessing = True)
	
	# save model
	model_path = 'models/bilstm_crf_model_{}'.format(datetime.utcnow())
	print("Saving model into {}".format(model_path))
	tf.keras.models.save_model(model_path)
if __name__ == '__main__':
	main()
