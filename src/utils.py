"""
utils.py - module to implement utils for BiLSTM-CRF
"""

import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentenceGetter(Sequence):
    """
    Inheritted class from tf.keras.utils.Sequence class to efficiently load
    data to Tensorflow/Keras model
    """
    def __init__(self, data, words, tags, maxlen, batch_size = 16, shuffle = False):
        """
        __init__ - initializer for SentenceGetter class
        Inputs:
            - data : String or Pandas DataFrame object
                Path string to file or dataframe object
            - words : set
                Set of distinct words
            - tags : set
                Set of distinct tags
        """

        if isinstance(data, str):
            # load data from Pandas file path
            data = pd.read_csv(data, encoding = 'latin1')
        elif isinstance(data, pd.DataFrame):
            # load data from Pandas DataFrame
            data = data
        else:
            raise Exception('Data is None or not found')
        self.word2dix = {w : i + 1 for i, w in enumerate(words)}
        self.tag2dix = {t : i for i, t in enumerate(tags)}
        self.n_tags = len(tags)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.maxlen = maxlen
        n_sent = 1
        self.grouped = data.groupby('Sentence #').apply(self.agg_func)\
            .reset_index().rename(columns = {0 : 'sentence'})['sentence']
        self.sentences = [s for s in self.grouped]

    def agg_func(self, input):
        """
        agg_func - function to group words/tags of sentences together
        """
        return [(w, p, t) for w, p, t in zip(input['Word'].values.tolist(),
            input['POS'].values.tolist(), input['Tag'].values.tolist())]

    def pad_sentences(self, input):
        input = [[self.word2dix[w[0]] for w in s] for s in input]
        return pad_sequences(maxlen = self.maxlen, sequences = input, padding = 'post', value = 0)
    
    def generate_labels(self, input):
        input = [[self.tag2dix[w[2]] for w in s] for s in input]
        input = pad_sequences(maxlen = self.maxlen, sequences = input, padding = 'post', value = self.tag2dix['O'])
        #return input
        return np.array([to_categorical(x, num_classes = self.n_tags) for x in input])
    def __len__(self):
        """
        __len__ - function to compute length of SentenceGetter
        """
        return int(self.grouped.shape[0] // self.batch_size)

    def __getitem__(self, index):
        if index == 0 and self.shuffle:
            # shuffle dataset for every iteration
            self.grouped = shuffle(self.grouped).reset_index()['sentence']

        # get batch of data
        sentences = self.grouped[index * self.batch_size : (index + 1) * self.batch_size]
        
        # generate sentences and labels
        labels = self.generate_labels(sentences)
        sentences = self.pad_sentences(sentences)
        
        return sentences, labels
