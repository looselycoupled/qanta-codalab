import re
import os
import json
import time
import random
from pathlib import Path
from timer import Timer
import logging
from pprint import pformat

import numpy as np
# import pandas as pd

import nltk
nltk.data.path += ["fixtures/nltk_data"]

from nltk import word_tokenize

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_


path_prefix = "../../"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_FILENAME = "dan.pt"
LOOKUP_FILENAME = "lookups.pt"
LOGFILE = "run-output.log"

##########################################################################
# Logging
##########################################################################

# remove existing log file
try:
    os.remove(LOGFILE)
except OSError:
    pass

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ])
logger = logging.getLogger()


##########################################################################
# Classes
##########################################################################

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

        vectors = word_vectors.vectors
        if not isinstance(vectors, torch.Tensor):
            vectors = torch.from_numpy(word_vectors.vectors)
        self.embeddings.weight.data.copy_(vectors)
        self.embeddings.weight.requires_grad = False

        self.encoder = DanEncoder(self.emb_dim, self.n_hidden_units, self.nn_dropout)
        self.dropout = nn.Dropout(nn_dropout)

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(self.nn_dropout)
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

##########################################################################
# Helpers
##########################################################################

def vectorize(ex, lookup):
    question_text, question_label = ex
    vec_text = [lookup[token] for token in question_text if token in lookup]
    return vec_text, question_label


def batchify(batch):
    question_len = list()           # list containing size of each question
    label_list = list()             # list containing index of correct answer

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

def embedding_data(emb_source="google"):
    if emb_source == "google":
        from gensim.models import KeyedVectors
        path = path_prefix + "data/GoogleNews-vectors-negative300.bin"
        word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
        word2ind = {k: v.index for k,v in word_vectors.vocab.items()}
        ind2word = {v:k for k,v in word2ind.items()}
    else:
        import torchtext.vocab as vocab
        word_vectors = vocab.GloVe(name='42B', dim=300)
        word2ind = word_vectors.stoi
        ind2word = word_vectors.itos

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

def should_save_model(epoch, trn_accuracies, dev_accuracies):
    SKIP = 2
    MIN_EPOCH = 35
    if epoch < MIN_EPOCH: return False
    if epoch % SKIP != 0: return False
    if max(trn_accuracies[-SKIP:]) > max(trn_accuracies[:-SKIP]): return True
    if max(dev_accuracies[-SKIP:]) > max(dev_accuracies[:-SKIP]): return True
    return False

def should_stop_training(trn_accuracies):
    MIN_STOP = 50
    if len(trn_accuracies) < MIN_STOP: return False

    WINDOW = 20
    curr_max = max(trn_accuracies[-WINDOW:])
    curr_min = min(trn_accuracies[-WINDOW:])
    return curr_max - curr_min < .05


##########################################################################
# Primary Train/Evaluate Functions
##########################################################################

def train(model, trn, dev, gradient_clip):

    start = time.time()
    trn_accuracies, trn_losses, dev_accuracies = [], [], []
    try:
        for epoch in range(num_epochs): #iterable:
            # train and record accuracy/loss
            acc, loss = train_epoch(model, trn, gradient_clip)
            trn_accuracies.append(acc)
            trn_losses.append(loss)

            # find accuracy against dev set
            dev_acc = round(evaluate(model, dev), 2)
            dev_accuracies.append(dev_acc)

            logger.info("epoch: {}, trn_acc: {}, trn_loss: {}, dev_acc: {}".format(
                epoch, acc, loss, dev_acc))

            if should_save_model(epoch, trn_accuracies, dev_accuracies):
                save_start = time.time()
                logger.info("saving model at epoch {} ...".format(epoch))
                torch.save(model, MODEL_FILENAME)
                logger.info("model saved in {} seconds".format(int(time.time() - save_start)))

            if should_stop_training(trn_accuracies):
                logger.info("exiting due to lack of progress")
                break

    except KeyboardInterrupt:
        logger.warn("early exit requested")

    elapsed = int(time.time() - start)
    logger.info("Training complete in {} seconds\n".format(elapsed))
    return trn_accuracies, trn_losses, dev_accuracies

def train_epoch(model, trn, gradient_clip):
    model.train()
    learning_rate = 0.0001
    epoch_acc, epoch_loss = [], []
    optimizer = torch.optim.RMSprop(model.parameters(),  lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for batch in trn:
        question_text = batch['text'].to(DEVICE)
        question_len = batch['len']
        labels = batch['labels']

        output = model(question_text, question_len)
        loss = criterion(output, labels)

        # record accuracy and loss for this batch
        _, preds = torch.max(output, 1)
        accuracy = torch.mean(torch.eq(preds, labels).float()).item()
        epoch_acc.append(accuracy)
        epoch_loss.append(loss.item())

        # update weights
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
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


##########################################################################
# Helper Functions (model management)
##########################################################################

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


class DanGuesser():

    def __init__(self):
        path = "fixtures/dan/" + MODEL_FILENAME
        self.model = torch.load(path, map_location=DEVICE)

        path = "fixtures/dan/" + LOOKUP_FILENAME
        lookups = torch.load(path)
        self.word2ind = lookups["word2ind"]
        self.ind2word = lookups["ind2word"]
        self.idx2ans = lookups["idx2ans"]
        self.ans2idx = lookups["ans2idx"]

    def guess(self, question, num_guesses=5):
        tokenized_text = word_tokenize(question)
        vec = vectorize((tokenized_text, "IGNORE_ME"), self.word2ind)[0]

        logits = self.model(torch.LongTensor([vec]),torch.FloatTensor([len(vec)]))

        ans_logits, ans_idx = logits.topk(num_guesses)
        ans = self.idx2ans[ans_idx[0][0].item()]
        return ans

##########################################################################
# Execution
##########################################################################

if __name__ == '__main__':
    num_epochs = 1000
    n_hidden_units = 14000
    batch_size = 500
    nn_dropout = 0.05       # there is discussion that combining batch norm with dropout is odd
    gradient_clip = .5      # forgot why I set this to .5 - need to test other values
    dataset_size = None
    embedding_source = "google"

    with Timer() as t:
        train_data, dev_data, test_data = datasets()

        ans2idx, idx2ans = answer_lookups(train_data + test_data + dev_data)
        word_vectors, word2ind, ind2word = embedding_data(embedding_source)

        torch.save({
            "word2ind": word2ind,
            "ind2word": ind2word,
            "idx2ans": idx2ans,
            "ans2idx": ans2idx,
            # "train_data": train_data,
        }, LOOKUP_FILENAME)

    logger.info("Setup complete in {}".format(t))

    with Timer() as t:
        n_classes = len(ans2idx.keys()) + 1
        options = {
            "n_classes": n_classes,
            "n_hidden_units": n_hidden_units,
            "nn_dropout": nn_dropout,
            "word_vectors": word_vectors,
        }
        model = create(**options)
        with open("run-details.json", "w") as f:
            details = {
                "num_epochs": num_epochs,
                "train_data": len(train_data),
                "dev_data": len(dev_data),
                "test_data": len(test_data),
                "batch_size": batch_size,
                "gradient_clip": gradient_clip,
            }
            details.update(options)
            details["word_vectors"] = details["word_vectors"].vectors.shape
            f.write(json.dumps(details))

        logger.info(model)
    logger.info("INFO: Model created in {}".format(t))

    trn = create_loader(train_data, batch_size, ans2idx, word2ind)
    dev = create_loader(dev_data, batch_size, ans2idx, word2ind)
    tst = create_loader(test_data, batch_size, ans2idx, word2ind)

    logger.info("INFO: starting training")
    trn_acc, trn_loss, dev_acc = train(model, trn, dev, gradient_clip)
    with open("run-stats.json", "w") as f:
        f.write(json.dumps({
            "trn_accuracies": [ii.item() for ii in trn_acc],
            "trn_losses": [ii.item() for ii in trn_loss],
            "dev_accuracies": dev_acc,
        }))

    final_trn_accuracy = evaluate(model, trn)
    final_tst_accuracy = evaluate(model, tst)

    logger.info("final accuracy on full training set: {}".format(final_trn_accuracy))
    logger.info("final accuracy on test set: {}".format(final_tst_accuracy))
    logger.info("final accuracy on dev set: {}".format(dev_acc[-1]))
    logger.info("max accuracy on dev set: {}".format(max(dev_acc)))

    logger.info(pformat({
        "num_epochs": num_epochs, "n_hidden_units": n_hidden_units, "batch_size": batch_size,
        "max_dev_acc": max(dev_acc), "train_size": len(train_data), "dev_size": len(dev_data),
        "final_acc_train": final_trn_accuracy, "final_acc_test": final_tst_accuracy,
        "dropout": nn_dropout, "gradient_clip": gradient_clip,
        "n_classes": n_classes, "lr": 0.0001
    }))

    # df = pd.DataFrame({"trn_acc": trn_acc, "trn_loss": trn_loss, "dev_acc": dev_acc})
    # logger.info("\n" + df.to_string())
