"""
predict.py - module for in-dev prediction
"""

# import dependenceis
import os
import tensorflow as tf

from bilstm_crf import NER

def main():
	# define model path
	model_path = 'logs/bilstm_crf_model_2020-10-31 17:57:42.384736'

	# define path to tag and word table
	tag_path = 'data/tags.txt'
	word_path = 'data/words.txt'

	# initailize NER model for inference
	network = NER(model_path, tag_table = tag_path, word_table = word_path)
	print(network.model.summary())

	# make predictions
	print(network.predict("Who am I"))

if __name__ == '__main__':
	main()
