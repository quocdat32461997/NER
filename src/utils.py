"""
utils.py - module to implement utils for BiLSTM-CRF
"""

import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from tensorlfow.keras.utils import Sequence

class SentenceGetter(Sequence):
    """
    Inheritted class from tf.keras.utils.Sequence class to efficiently load
    data to Tensorflow/Keras model
    """
    def __init__(self, data, n_sent = 1, batch_size = 16, shuffle = True, **kwargs):
        """
        __init__ - initializer for SentenceGetter class
        """
        super(self, SentenceGetter).__init__()

        if isinstance(data, str):
            # load data from Pandas file path
            data = pd.read_csv(data, encoding = 'latin1')
        elif istance(data, pd.DataFrame):
            # load data from Pandas DataFrame
            data = data
        else:
            raise Exception('Data is None or not found')

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.grouped = data.groupby('Sentence #').apply(self.agg_func)\
            .reset_index().rename(columns = {0 : 'sentence'})['sentence']
        self.sentences = [s for s in self.grouped]

    def agg_func(self, input):
        return [(w, p, t) for w, p, t in zip(input['Word'].values.tolist(),
            input['POS'].values.tolist(), input['Tag'].values.tolist())]

    def __len__(self):
        """
        __len__ - function to compute length of SentenceGetter
        """
        return math.ceil(self.data.shape[0] / self.batch_size)

    def __getitem__(self, index):
        if index == 0 and self.shuffle:
            # shuffle dataset for every iteration
            self.grouped = shuffle(self.grouped).reset_index()['sentence']

        # get batch of data
        return self.grouped[index * self.batch_size : (index + 1) * self.batch_size]
