"""
utils.py - module to implement utils for BiLSTM-CRF
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

def process_text(input):
	"""
	Function process_text to clean text before Training process
	Inputs:
		- input : TBD
	Outputs:
		- input : TBD
	"""
	
	return input
