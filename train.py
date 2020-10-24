"""
train.py - module to implement Training functioanlity
"""

# import dependencies
import os
import argparse

from lib import *

def main():

	# define data paths
	texts = ['ner_text_dataset.txt', 'wnut17train_conll_text.txt']
	targets = ['ner_label_dataset.txt', 'wnut17train_conll_label.txt']

	# define path to word table lookup
	word_table_path = './embeddings/words.lst'
	embedding_table_path = './embeddings/embeddings.txt'

	dataset = Dataset(texts = texts, targets = targets, max_len = 64)()

	for data in dataset:
		print(data)
		break
if __name__ == '__main__':
	main()
