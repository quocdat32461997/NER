"""
output.py - module to store BiLSTM-CRF model
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Lambda, Embedding, TimeDistributed, Dropout, Bidirectional, Dense
import tensorflow_addons as tfa
from tensorflow_addons.text import crf_log_likelihood, viterbi_decode

from src.loss import CRF

def embedding_layer(input_dim, output_dim, input_length, mask_zero):
    return Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length, mask_zero = mask_zero)

def bilstm_crf(maxlen, n_tags, embedding_dim, n_words, mask_zero, training = True):
    """
    bilstm_crf - module to build BiLSTM-CRF model
    Inputs:
        - input_shape : tuple
            Tensor shape of inputs, excluding batch size
    Outputs:
        - output : tensorflow.keras.outputs.output
            BiLSTM-CRF output
    """
    input = Input(shape = (maxlen,))
    # Embedding layer
    embeddings = embedding_layer(input_dim = n_words + 1, output_dim = embedding_dim, input_length = maxlen, mask_zero = mask_zero)
    output = embeddings(input)

    # BiLSTM layer
    output = Bidirectional(LSTM(units = 50, return_sequences = True, recurrent_dropout = 0.1))(output)

    # Dense layer
    output = TimeDistributed(Dense(n_tags, activation = 'relu'))(output)

    # CRF layer
    output = CRF(n_tags, name = 'crf_layer')(output)

    return Model(input, output)
