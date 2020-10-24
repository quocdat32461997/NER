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

	dataset = Dataset(texts = texts, targets = targets,
		word_table = word_table_path, tag_table = tag_table_path , max_len = 64)()

	for data in dataset:
		texts, targets = data
		print("Texts shape", texts.shape)
		print("Targets shape", targets.shape)
		break
if __name__ == '__main__':
	main()
