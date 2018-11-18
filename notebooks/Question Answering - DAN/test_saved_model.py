# For running on AWS p2.xlarge (CUDA)
# 1. Create spot instance of "Deep Learning AMI (Ubuntu) Version 18.0 (ami-0484cefb8f48dafe8)"
# 2. SCP json and word embeddings word_vectors into ~/code/data/
#     * scp -i ~/.ssh/keys/aws-leis-personal.pem ./GoogleNews-vectors-negative300.bin ubuntu@ec2-18-207-205-3.compute-1.amazonaws.com:/home/ubuntu/code/data/
# 3. SCP this file into ~/code/
# 4. SSH in and activate pytorch environment `source activate pytorch_p36`
# 5. Double check imports and install missing components
#     * ex: `conda install -c anaconda gensim` `conda install tqdm`
#     * import nltk; nltk.download('punkt')
# 6. File is ready to be executed: `python dan-qa.py`
#     * you may want to use screen or nohup so you can exit ssh session


# Basic Linux Screen Usage
# Below are the most basic steps for getting started with screen:
#
# 1. On the command prompt, type screen.
# 2. Run the desired program.
# 3. Use the key sequence Ctrl-a + Ctrl-d to detach from the screen session.
# 4. Reattach to the screen session by typing screen -r.
# 5. You can list running screen sessions using screen -ls

import time
import json
import random
from pprint import pprint
from collections import namedtuple
from nltk.probability import FreqDist
from pprint import pprint
from nltk import word_tokenize
import pickle

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import trange, tqdm_notebook, tqdm

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_

from gensim.models import word2vec
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings('ignore')

path_prefix = "../../"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#===========================================================================
# Classes and Functions
#===========================================================================

class DanModel(nn.Module):

    def __init__(self, n_classes, n_hidden_units):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units

        self.vocab_size, self.emb_dim = word_vectors.vectors.shape
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(word_vectors.vectors))
        self.embeddings.weight.requires_grad = False

        self.linear1 = nn.Linear(self.emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)
        self.softmax = nn.Softmax()

        self.classifier = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            # nn.Dropout(0.25),
            self.linear2,
            self.softmax
        )

    def forward(self, input_text, text_len):
        """
        Model forward pass

        Keyword arguments:
        input_text : vectorized question text
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer

        """
        # get word embeddings
        text_embed = self.embeddings(input_text).to(device)

        # calculate the mean embeddings
        encoded = text_embed.sum(1)
        encoded /= text_len.view(text_embed.size(0), -1).to(device)

        # run data through the classifier
        return self.classifier(encoded)


#===========================================================================
# Load datasets
#===========================================================================

if __name__ == '__main__':
    stuff = torch.load("lookups.pt")
    word2ind = stuff["word2ind"]
    idx2ans = stuff["idx2ans"]

    train_file = path_prefix + "data/qanta.train.2018.04.18.json"

    def load_data(filename):
        data = list()
        with open(filename) as json_data:
            questions = json.load(json_data)["questions"]
        return questions

    train_data = load_data(train_file)
    data = train_data[:500]

    model = torch.load("dan-qa-aws.pt")
    print(model)

    for i in range(20):
        q = random.choice(data)
        ii = [word2ind[w] for w in q["text"].split() if w in word2ind]
        _, ans_idx = model(torch.LongTensor([ii]),torch.FloatTensor([1])).topk(1)
        ans_idx = ans_idx.data.numpy()[0][0]
        print(idx2ans[ans_idx] == q["page"])
