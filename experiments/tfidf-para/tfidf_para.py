import json
import os
from glob import glob
from pprint import pprint
import requests
import urllib.parse as urlparse

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def files():
    return sorted(glob("./data/*.json"))


class WikidataIterator(object):
    def __init__(self):
        # self.path = path
        self.i_to_ans = {}

    def _doc_iterator(self):
        files = sorted(glob("*.json"))

        for path in files:
            with open(path, "r") as f:
                print(path)
                for line in f:
                    yield json.loads(line)

    def _para_iterator(self, doc):
        MIN_LENGTH = 300
        text = [ii for ii in doc.strip().split("\n") if ii.strip()]

        too_small = ""
        for para in text:
            if len(para) < MIN_LENGTH:
                too_small = too_small + " " + para
                continue

            yield (too_small + " " + para).strip()
            too_small = ""

    @property
    def docs(self):
        counter = 0
        for doc in self._doc_iterator():
            ans = doc["ans"]

            for para in self._para_iterator(doc["text"]):
                self.i_to_ans[counter] = ans
                counter += 1
                yield para




class TFIDF():

    def __init__(self):
        self.i_to_ans = None

    def train(self, ngram_range=(1, 1), min_df=1, max_df=.95):
        wikidata = WikidataIterator()

        vectorizer_kwargs = {
            'ngram_range': ngram_range,
            'min_df': min_df,
            'max_df': max_df
        }
        start = time.time()
        self.tfidf_vectorizer = TfidfVectorizer(**vectorizer_kwargs).fit(wikidata.docs)
        elapsed = int(time.time() - start)
        print("INFO: fit completed in {} seconds".format(elapsed))

        start = time.time()
        self.tfidf_matrix = self.tfidf_vectorizer.transform(wikidata.docs)
        elapsed = int(time.time() - start)
        print("INFO: transform completed in {} seconds".format(elapsed))

        self.i_to_ans = wikidata.i_to_ans

    def guess(self, questions, max_n_guesses=2):
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses



if __name__ == '__main__':
    import pickle
    model = TFIDF()
    model.train()
    pickle.dump(model, open("tfidf_by_para.pickle", "wb"))
