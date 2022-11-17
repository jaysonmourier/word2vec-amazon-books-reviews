import pandas as pd

from torchtext.data import get_tokenizer

from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import effective_n_jobs

import numpy as np

import time

# PARAMS
TOTAL_EPOCHS = 15
PATH_DATASET = 'dataset/br10.csv'
WINDOW_SIZE=4
# CBOW=0 / Skip-Gram=1
SG=1

def load_dataset(path, header):
    return pd.read_csv(path, header=header)

def tokenize(df, tokenizer, sw):
    listTokens = list()
    dfSize=df.size
    for i in range(df.size):
        tokens = tokenizer(df.iloc[i][0])
        for token in tokens:
            if token in sw:
                tokens.remove(token)
        listTokens.append(tokens)
        if i%int(df.size*.1) == 0:
            print(f'done: {i}/{df.size}')
    return listTokens

# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    https://radimrehurek.com/gensim/models/callbacks.html
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f'Epoch [{self.epoch:2}/{TOTAL_EPOCHS:2}]', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss: {}: {}'.format(self.epoch, loss))
        else:
            print('Loss: {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

if __name__ == "__main__":
    df = load_dataset(PATH_DATASET, header=None)
    tokenizer = get_tokenizer('basic_english')
    listTokens = tokenize(df, tokenizer, set(stopwords.words('english')))
    start_at = time.time()
    model = Word2Vec(listTokens, epochs=TOTAL_EPOCHS, window=WINDOW_SIZE, min_count=1, sg=SG, workers=effective_n_jobs(-1), compute_loss=True, callbacks=[callback()])
    end_at = time.time()
    print(f"time to train: {end_at - start_at}")
    model.save("word2vec.model")