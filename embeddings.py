from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import bcolz  # to process the data from Glove File
import pickle  # to dump and load pretrained glove vectors
import copy  # to make deepcopy of python lists and dictionaries
import operator
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids

from torchtext.data import Field
from torchtext.datasets import TranslationDataset
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
glove_path = './'
emb_dim = 50


def glove_data():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

def glove_embedding(target_vocab):
    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0

    print("matrix len", matrix_len)

    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    num_embeddings, embedding_dim = weights_matrix.shape
    return weights_matrix, num_embeddings, embedding_dim

def elmo_embedding():
    options_file = "elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file = "2x1024_128_2048cnn_1xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 2, dropout=0)


if __name__ == "__main__":
    asl = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=False)
    train_data = TranslationDataset(path="data/", exts=["asl_train_processed.txt", "en_train.txt"], fields=[asl, asl])
    asl.build_vocab(train_data, min_freq=2)
    weights_matrix, num_embeddings, embedding_dim = glove_embedding(asl.vocab.itos)
