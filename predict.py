"""
predict.py - module for in-dev prediction
"""

# import dependenceis
import os
import tensorflow as tf

from bilstm_crf import NER

model_path = 'bilstm_crf_model_2020-10-31 17:11:44.593464'

network = NER(model_path, tags = None)
print(network.model.summary())
