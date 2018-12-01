import os
import random
import json
import numpy as np
from tqdm import tqdm_notebook, trange
from pprint import pprint

import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re


from concurrent import futures

TRAIN_FILE = "../data/qanta.train.2018.04.18.json"
TEST_FILE = "../data/qanta.test.2018.04.18.json"



DEFAULT_PARAMETERS = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1,1),
    "max_df": 1.0,
    "min_df": 1,
    "max_features": None,
    "norm" : "l2",
    "tokenizer": None,
}



PARAMETERS = {
    "lowercase": (False, ),
    "stop_words": ( None, ),
    "ngram_range": ((1,2), (1,3)),
    "max_df": [.75, .85, .95],                  # ignore terms that have a document frequency strictly higher
    "min_df": [.05, .1, .2, 25, 50, 100, 500],  # ignore terms that have a document frequency strictly lower
    "max_features": (5000, 10000, 20000),
    "norm" : ("l1", None),
    "tokenizer": (True, ),
}


def params_generator():
    collection = [DEFAULT_PARAMETERS]
    for k, values in PARAMETERS.items():
        for v in values:
            item = dict(DEFAULT_PARAMETERS)
            item[k] = v
            collection.append(item)
    return collection




# def params_generator(keys, params, data):
#     key = keys.pop(0)
#     updated = []
#
#     # loop through values for my key
#     for val in params[key]:
#         for item in data:
#             ii = dict(item)
#             ii[key] = val
#             updated.append(ii)
#     if len(keys) > 0:
#         result = params_generator(keys, params, updated)
#     else:
#         return updated
#     return result



def load_data(filename):
    data = list()
    with open(filename) as json_data:
        for q in json.load(json_data)["questions"]:
            yield (q['text'], q['page'])


class TFIDF():

    def __init__(self, options):
        self.i_to_ans = {}
        self.options = options
        self.stemmer = PorterStemmer()

    def stemming_tokenizer(self, str_input):
        words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
        words = [self.stemmer.stem(word) for word in words]
        return words

    def train(self, docs, answers):
        tfidf_kwargs = dict(self.options)

        if self.options.get("tokenizer", None):
            tfidf_kwargs["tokenizer"] = self.stemming_tokenizer

        start = time.time()
        self.tfidf_vectorizer = TfidfVectorizer(**tfidf_kwargs).fit(docs)
        elapsed = int(time.time() - start)

        start = time.time()
        self.tfidf_matrix = self.tfidf_vectorizer.transform(docs)
        elapsed = int(time.time() - start)

        for idx, ans in enumerate(answers):
            self.i_to_ans[idx] = ans

    def guess(self, questions, max_n_guesses=2):
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses




def evaluate(options):
    print(f"INFO: {options}")
    start = time.time()
    model = TFIDF(options)

    print("INFO: loading data")
    train_docs, train_answers = zip(*load_data(TRAIN_FILE))
    test_docs, test_answers = zip(*load_data(TEST_FILE))

    # train_docs, train_answers = train_docs[:1000], train_answers

    print("INFO: training")
    model.train(train_docs, train_answers)

    print("INFO: testing")
    guesses = guesses = np.array([ans[0][0] for ans in model.guess(test_docs, 1)])
    num_correct = (guesses == np.array(test_answers)).sum()

    options["num_correct"] = int(num_correct)
    options["accuracy"] = num_correct / len(test_answers)
    options["elapsed"] = int(time.time() - start)
    return options


if __name__ == '__main__':
    results = []
    param_list = params_generator()[:1]
    print("INFO: permutations for search: {}".format(len(param_list)))

    pool = futures.ProcessPoolExecutor(max_workers=1)
    promises = [pool.submit(evaluate, i) for i in param_list]

    for f in futures.as_completed(promises):
        results.append(f.result())
        pprint(results[-1])

    with open("gridsearch_results.json", "w") as f:
        f.write(json.dumps(results))

    df = pd.DataFrame(results).sort_values(by=['accuracy'], ascending=False)
    print("\n\n" + df.to_string())
