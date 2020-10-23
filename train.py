"""
train.py - module to implement Training functioanlity
"""

# import dependencies
import os
import argparse

from lib import *

def main():

	files = ['ner_dataset.txt', 'wnut17train_conll.txt']
	dataset = Dataset(files)()

	for data in dataset:
		print(data)
		break
if __name__ == '__main__':
	main()
