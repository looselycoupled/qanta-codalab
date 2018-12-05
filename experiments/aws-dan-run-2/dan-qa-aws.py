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


Run = namedtuple("Run", "epochs n_hidden_units training_size elapsed accuracy batch_size accuracies")
RUNS = []

path_prefix = ""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#===========================================================================
# Load word embeddings
#===========================================================================

print("INFO: loading word embeddings...")
path = path_prefix + "data/GoogleNews-vectors-negative300.bin"
word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)
word2ind = {k: v.index for k,v in word_vectors.vocab.items()}
ind2word = {v:k for k,v in word2ind.items()}
print("INFO: word embeddings loaded\n")


#===========================================================================
# Load datasets
#===========================================================================

train_file = path_prefix + "data/qanta.train.2018.04.18.json"
dev_file = path_prefix + "data/qanta.dev.2018.04.18.json"
test_file = path_prefix + "data/qanta.test.2018.04.18.json"

def load_data(filename):
    data = list()
    with open(filename) as json_data:
        questions = json.load(json_data)["questions"]
    return questions

print("INFO: loading data")
train_data = load_data(train_file)
dev_data = load_data(dev_file)
test_data = load_data(test_file)
all_data = train_data + dev_data + test_data
print("INFO: data loaded\n")




#===========================================================================
# Module Variables
#===========================================================================



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



class Question_Dataset(Dataset):

    def __init__(self, examples, lookup):
        self.examples = examples
        self.lookup = lookup

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.lookup)

    def __len__(self):
        return len(self.examples)


def vectorize(ex, lookup):
    """
    vectorize a single example based on the word2ind dict.

    Keyword arguments:
    exs: list of input questions-type pairs
    ex: tokenized question sentence (list)
    label: type of question sentence

    Output:  vectorized sentence(python list) and label(int)
    e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
    """
    vec_text = []
    question_text, question_label = ex

    for idx, token in enumerate(question_text):
        if token in lookup:
            vec_text.append(lookup[token])

    return vec_text, question_label


def batchify(batch):
    """
    Gather a batch of individual examples into one batch,
    which includes the question text, question length and labels

    Keyword arguments:
    batch: list of outputs from vectorize function
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])
    target_labels = torch.LongTensor(label_list).to(device)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_().to(device)
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text).to(device)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len).to(device), 'labels': target_labels}
    return q_batch

GRADIENT_CLIP = 0
def train(model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model

    Keyword arguments:
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """

    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    curr_accuracy = 0
    start = time.time()

    for idx, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        output = model(question_text, question_len)
        loss = criterion(output, labels)

        loss.backward()
        if GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        optimizer.zero_grad()

    return curr_accuracy


def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set

    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """
    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text']
        question_len = batch['len']
        labels = batch['labels']

        logits = model(question_text, question_len)

        top_n, top_i = logits.topk(1)
        num_examples += question_text.size(0)

        diff = top_i.squeeze() - labels
        error += torch.nonzero(diff).size(0)

    accuracy = 1 - error / num_examples
    return accuracy



# Start Training
n_hidden_units = 7000
checkpoint = 2
batch_size = 500
num_epochs = 1000
start = time.time()

accuracy = 0
accuracies = []

data = all_data

ans2idx = {}
counter = 0
for idx, q in enumerate(data):
    if q["page"] not in ans2idx:
        ans2idx[q["page"]] = counter
        counter += 1
idx2ans = {v: k for k, v in ans2idx.items()}

torch.save({
    "word2ind": word2ind,
    "idx2ans": idx2ans,
}, "lookups.pt")


data = [(word_tokenize(q["text"]), ans2idx[q["page"]]) for q in data]
n_classes = len(ans2idx.keys()) + 1

print("INFO: creating DanModel...")
model = DanModel(n_classes, n_hidden_units=n_hidden_units)
model.to(device)
print(model)
print("INFO: Setup finished in {} seconds\n".format(int(time.time() - start)))

# Create testing dataloader
train_dataset = Question_Dataset(data, word2ind)
train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=0, collate_fn=batchify)

dev_size = int(len(data) / 10)
dev_data = data[:dev_size] # random.sample(data, dev_size)
dev_dataset = Question_Dataset(dev_data, word2ind)
dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size,
    sampler=dev_sampler, num_workers=0, collate_fn=batchify)

print("INFO: training_size={}, dev_size={}, batch_size={}, n_classes={}, n_hidden_units={}".format(
    len(data), dev_size, batch_size, n_classes, n_hidden_units
))

# Train / Fit
start = time.time()
iterable = tqdm(range(num_epochs))
prev_high = 0
for epoch in iterable:
    train(model, train_loader, dev_loader, accuracy, device)
    accuracy = round(evaluate(dev_loader, model, device), 2)
    accuracies.append(accuracy)

    if epoch % 10 == 0:
        iterable.write("Epoch: {}, Accuracy: {}".format(epoch, accuracy))

    if (epoch % 50 == 0 or (epoch + 1) == num_epochs) and epoch > 0 and accuracy > prev_high:
        save_start = time.time()
        iterable.write("saving model at epoch {} ...".format(epoch))
        torch.save(model, "dan-qa-aws.model")
        iterable.write("model saved in {} seconds".format(int(time.time() - save_start)))
        prev_high = accuracy

elapsed = int(time.time() - start)
print("Training complete in {} seconds\n".format(elapsed))

# Test
print('\nstart testing:\n')
accuracy = evaluate(train_loader, model, device)
print("Final accuracy on training set: {}".format(accuracy))

print(accuracies)


if False:
    run_data = {
        "epochs": num_epochs,
        "n_hidden_units": n_hidden_units,
        "training_size": len(data),
        "elapsed": elapsed,
        "accuracy": accuracy,
        "accuracies": accuracies,
        "batch_size": batch_size
    }

    torch.save(model, "dan-qa.model")

    with open('run.pickle', 'wb') as f:
        pickle.dump(run_data, f)
