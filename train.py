"""
train.py - module to implement Training functioanlity
"""

# import dependencies
import os
import argparse

from lib import *

def main():

	file = 'ner_dataset.csv'
	dataset = Dataset(file)()


if __name__ == '__main__':
	main()
