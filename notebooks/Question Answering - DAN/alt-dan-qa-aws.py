import re
import os
import json
import time
import random
from pathlib import Path
from timer import Timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm_notebook, tqdm
import matplotlib

from nltk import word_tokenize
from gensim.models import KeyedVectors

import torch
import torch.nn as nn
# from torch.autograd import Variable
# from torch.nn import functional as F
# from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_


# from torchtext.data.field import Field
# from torchtext.data.iterator import Iterator

# from qanta import qlogging
# from qanta.util.io import shell, get_tmp_filename
# from qanta.torch.dataset import QuizBowl
# from qanta.config import conf
# from qanta.guesser.abstract import AbstractGuesser
# from qanta.datasets.abstract import QuestionText
# from qanta.torch import (
#     BaseLogger, TerminateOnNaN, EarlyStopping, ModelCheckpoint,
#     MaxEpochStopping, TrainingManager
# )

path_prefix = "../../"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_FILENAME = "alt-dan-qa-aws.pt"


def create_save_model(model):
    def save_model(path):
        torch.save(model.state_dict(), path)
    return save_model


class DanEncoder(nn.Module):
    def __init__(self, embedding_dim, n_hidden_units, dropout_prob):
        super(DanEncoder, self).__init__()
        encoder_layers = [
            nn.Linear(embedding_dim, n_hidden_units),
            nn.BatchNorm1d(n_hidden_units),
            nn.ELU(),
            nn.Dropout(dropout_prob),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x_array):
        return self.encoder(x_array)


class DanModel(nn.Module):
    def __init__(self,
            n_classes,
            n_hidden_units,
            nn_dropout,
            word_vectors):
        super(DanModel, self).__init__()

        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout

        self.vocab_size, self.emb_dim = word_vectors.vectors.shape
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(word_vectors.vectors))
        self.embeddings.weight.requires_grad = False

        self.encoder = DanEncoder(self.emb_dim, self.n_hidden_units, self.nn_dropout)
        self.dropout = nn.Dropout(nn_dropout)

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(self.nn_dropout)
        )

    # def _pool(self, embed, lengths, batch_size):
    #     return embed.sum(1) / lengths.view(batch_size, -1)
    #
    # def forward_old(self, input_, lengths, qanta_ids):
    #     """
    #     :param input_: [batch_size, seq_len] of word indices
    #     :param lengths: Length of each example
    #     :param qanta_ids: QB qanta_id if a qb question, otherwise -1 for wikipedia, used to get domain as source/target
    #     :return:
    #     """
    #     for key in lengths:
    #         if not isinstance(lengths[key], Variable):
    #             lengths[key] = Variable(lengths[key].float(), volatile=not self.training)
    #
    #     embedding_list = []
    #
    #     unigram_input = input_['unigram']
    #     embed = self.embeddings(unigram_input)
    #     embed = self._pool(embed, lengths['unigram'].float, unigram_input.size()[0])
    #     embed = self.dropout(embed)
    #     embedding_list.append(embed)
    #
    #     concat_embed = torch.cat(embedding_list, dim=1)
    #     encoded = self.encoder(concat_embed)
    #     return self.classifier(encoded)

    def forward(self, input_text, text_len):
        """
        Model forward pass

        Keyword arguments:
        input_text : vectorized question text
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer

        """
        # get word embeddings
        text_embed = self.embeddings(input_text).to(DEVICE)

        # calculate the mean embeddings
        encoded = text_embed.sum(1)
        encoded /= text_len.view(text_embed.size(0), -1).to(DEVICE)

        # run data through the classifier
        encoded = self.encoder(encoded)
        return self.classifier(encoded)


class Question_Dataset(Dataset):

    def __init__(self, examples, lookup):
        self.examples = examples
        self.lookup = lookup

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.lookup)

    def __len__(self):
        return len(self.examples)


def vectorize(ex, lookup):
    question_text, question_label = ex
    vec_text = [lookup[token] for token in question_text if token in lookup]
    return vec_text, question_label


def batchify(batch):
    question_len = list()
    label_list = list()

    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list).to(DEVICE)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_().to(DEVICE)

    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text).to(DEVICE)
        x1[i, :len(question_text)].copy_(vec)

    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len).to(DEVICE), 'labels': target_labels}
    return q_batch

def create_loader(data, batch_size, answer_lookup, word_lookup):
    input_tuples = [(word_tokenize(q["text"]), answer_lookup[q["page"]]) for q in data]
    dataset = Question_Dataset(input_tuples, word_lookup)
    return torch.utils.data.DataLoader(dataset,
        batch_size=batch_size,
        shuffle=True,
        # sampler=torch.utils.data.sampler.RandomSampler(dataset),
        collate_fn=batchify
    )


def datasets():
    train_file = path_prefix + "data/qanta.train.2018.04.18.json"
    dev_file = path_prefix + "data/qanta.dev.2018.04.18.json"
    test_file = path_prefix + "data/qanta.test.2018.04.18.json"

    def load_data(filename):
        data = list()
        with open(filename) as json_data:
            questions = json.load(json_data)["questions"]
        return questions

    train_data = load_data(train_file)
    dev_data = load_data(dev_file)
    test_data = load_data(test_file)

    return train_data, dev_data, test_data

def embedding_data():
    EMBEDDING_DATA_FILE = "embedding_data.pt"
    path = Path(EMBEDDING_DATA_FILE)
    if path.is_file():
        return torch.load(EMBEDDING_DATA_FILE)

    path = path_prefix + "data/GoogleNews-vectors-negative300.bin"
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
    word2ind = {k: v.index for k,v in word_vectors.vocab.items()}
    ind2word = {v:k for k,v in word2ind.items()}
    # torch.save((word_vectors, word2ind, ind2word), EMBEDDING_DATA_FILE)
    return word_vectors, word2ind, ind2word

def answer_lookups(dataset):
    ans2idx = {}
    counter = 0
    for idx, q in enumerate(dataset):
        if q["page"] not in ans2idx:
            ans2idx[q["page"]] = counter
            counter += 1
    idx2ans = {v: k for k, v in ans2idx.items()}
    return ans2idx, idx2ans


def train(model, trn, dev, gradient_clip):

    start = time.time()
    iterable = tqdm(range(num_epochs))
    prev_high = 0
    accuracies, losses = [], []

    for epoch in iterable:
        # train and record accuracy/loss
        acc, loss = train_epoch(model, trn, gradient_clip)
        accuracies.append(acc)
        losses.append(loss)
        iterable.write("epoch: {}, accuracy: {}, loss: {}".format(epoch, acc, loss))

        # find accuracy against dev set
        eval_accuracy = round(evaluate(model, dev), 2)

        # report dev accuracy every now and then
        if epoch % 10 == 0:
            iterable.write("Epoch: {}, Dev Accuracy: {}".format(epoch, eval_accuracy))

        if (epoch % 50 == 0 or (epoch + 1) == num_epochs) and epoch > 0 and eval_accuracy > prev_high:
            save_start = time.time()
            iterable.write("saving model at epoch {} ...".format(epoch))
            torch.save(model, MODEL_FILENAME)
            iterable.write("model saved in {} seconds".format(int(time.time() - save_start)))
            prev_high = eval_accuracy

    elapsed = int(time.time() - start)
    print("Training complete in {} seconds\n".format(elapsed))
    return accuracies, losses,  elapsed

def train_epoch(model, trn, gradient_clip):
    model.train()
    epoch_acc, epoch_loss = [], []
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for batch in trn:
        question_text = batch['text'].to(DEVICE)
        question_len = batch['len']
        labels = batch['labels']

        output = model(question_text, question_len)
        loss = criterion(output, labels)

        # record accuracy and loss for this batch
        _, preds = torch.max(output, 1)
        accuracy = torch.mean(torch.eq(preds, labels).float()).data[0]
        epoch_acc.append(accuracy)
        epoch_loss.append(loss.data[0])

        # update weights
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm(model.parameters(), gradient_clip)
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(epoch_acc), np.mean(epoch_loss)



def evaluate(model, data_loader):
    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text']
        question_len = batch['len']
        labels = batch['labels']
        num_examples += question_text.size(0)

        logits = model(question_text, question_len)
        _, preds = logits.topk(1)

        diff = preds.squeeze() - labels
        error += torch.nonzero(diff).size(0)

    accuracy = 1 - error / num_examples
    return accuracy


def create(n_classes, word_vectors, n_hidden_units, nn_dropout):
    model = DanModel(
        n_classes=n_classes,
        n_hidden_units=n_hidden_units,
        nn_dropout=nn_dropout,
        word_vectors=word_vectors,
    )
    model.to(DEVICE)
    return model


def save(model):
    pass


def load(path):
    pass



if __name__ == '__main__':
    num_epochs = 100
    n_hidden_units = 500
    batch_size = 100
    nn_dropout = 0.05
    gradient_clip = None
    dataset_size = 1000

    with Timer() as t:
        train_data, dev_data, test_data = datasets()
        train_data = train_data + dev_data + test_data
        if dataset_size:
            train_data = train_data[:dataset_size]
            dev_data = train_data #random.sample(train_data, int(dataset_size / 2))
            test_data = random.sample(train_data, int(dataset_size / 2))

        ans2idx, idx2ans = answer_lookups(train_data)
        word_vectors, word2ind, ind2word = embedding_data()
    print("INFO: Setup complete in {}".format(t))

    with Timer() as t:
        options = {
            "n_classes": len(ans2idx.keys()) + 1,
            "n_hidden_units": n_hidden_units,
            "nn_dropout": nn_dropout,
            "word_vectors": word_vectors,
        }
        model = create(**options)
        print(model)
    print("INFO: Model created in {}".format(t))

    trn = create_loader(train_data, batch_size, ans2idx, word2ind)
    dev = create_loader(dev_data, batch_size, ans2idx, word2ind)
    tst = create_loader(test_data, batch_size, ans2idx, word2ind)

    acc, loss, train_time = train(model, trn, dev, gradient_clip)

    final_accuracy = evaluate(model, trn)
    print("final accuracy on full training set: {}".format(final_accuracy))

    pd.DataFrame({"acc": acc, "loss": loss}).plot()
    plt.show()
