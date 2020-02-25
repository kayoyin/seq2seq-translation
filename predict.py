import torch

from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator

import random
import math
import time
import sys
from utils import *
from lstm import *

import numpy as np

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

asl = Field(tokenize=tokenize_asl,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=False)


en = Field(tokenize=tokenize_en,
               init_token='<sos>',
               eos_token='<eos>',
               lower=True,
               batch_first=False)


def translate_sentence(sentence):
    tokenized = tokenize_asl(sentence) #tokenize sentence
    numericalized = [asl.vocab.stoi[t] for t in tokenized] #convert tokens into indexes
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device) #convert to tensor and add batch dimension
    translation_tensor_probs = model(tensor, None, 0).squeeze(1) #pass through model to get translation probabilities
    translation_tensor = torch.argmax(translation_tensor_probs, 1) #get translation from highest probabilities
    translation = [en.vocab.itos[t] for t in translation_tensor][1:] #we ignore the first token, just like we do in the training loop
    translation = ' '.join(translation[:translation.index('<eos>')])
    return translation

if __name__ == "__main__":

    model_name = sys.argv[1]
    print(model_name)
    model_path = model_name + '.pt'
    output_path = model_name + '.txt'

    train_data = TranslationDataset(path="data/", exts=["asl_train_processed.txt", "en_train.txt"], fields=[asl, en])
    valid_data = TranslationDataset(path="data/", exts=["asl_val_processed.txt", "en_val.txt"], fields=[asl, en])
    test_data = TranslationDataset(path="data/", exts=["asl_test_processed.txt", "en_test.txt"], fields=[asl, en])

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    print(vars(test_data.examples[1]))

    asl.build_vocab(train_data, min_freq=2)
    en.build_vocab(train_data, min_freq=2)

    print(f"Unique tokens in source (asl) vocabulary: {len(asl.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(en.vocab)}")

    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)

    INPUT_DIM = len(asl.vocab)
    OUTPUT_DIM = len(en.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load(model_path))

    with open('data/asl_test_processed.txt', 'r') as file:
        data = file.readlines()

    with open(output_path, 'w') as file:
        for sent in data:
            print(sent)
            pred = translate_sentence(sent)
            print(pred)
            file.write(pred)



