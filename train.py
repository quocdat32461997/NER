"""
train.py - module to implement Training functioanlity
"""

# import dependencies
import os
import argparse
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from lib import *

def main():

	# define data paths
	texts = ['ner_text_dataset.txt', 'wnut17train_conll_text.txt']
	targets = ['ner_label_dataset.txt', 'wnut17train_conll_label.txt']

	# define path to lookup tables
	word_table_path = './words.txt'
	tag_table_path = './tags.txt'

	# retrieve number of tags
	with open(tag_table_path) as file:
		n_tags = len(file.read().split('\n'))

	data_pipeline = Dataset(texts = texts, targets = targets,
		word_table = word_table_path, tag_table = tag_table_path)
	dataset = data_pipeline()

	print("Testing Training Data Pipeline")
	for data in dataset:
		texts, targets = data
		print("Texts shape", texts.shape)
		print("Targets shape", targets.shape)
		break

	# define BiLSTM-CRF model
	# define parameters
	pretrained_embed = '/Users/datqngo/Desktop/projects/BiLSTM-CRF/glove.twitter.27B/glove.twitter.27B.100d.txt'
	hidden_units = 150
	embed_dim = 100
	max_len = None

	# initialie BiLSTM-CRF class object
	bilstm_crf = BiLSTM_CRF(max_len = max_len, embed_dim = embed_dim, n_tags = data_pipeline.tag_table.size(), word_table = data_pipeline.word_table, hidden_units = hidden_units, pretrained_embed = pretrained_embed)
	model = bilstm_crf()

	# inspect data architecture
	print(model.summary())
if __name__ == '__main__':
	main()
