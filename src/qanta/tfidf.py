from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

import re
from nltk.stem.porter import PorterStemmer

import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

from qanta import util
from qanta.dataset import QuizBowlDataset


MODEL_PATH = 'tfidf.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs


class TfidfGuesser:
    def __init__(self, stem=False):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None
        self.stemmer = PorterStemmer() if stem else None

    def stemming_tokenizer(self, str_input):
        words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
        words = [self.stemmer.stem(word) for word in words]
        return words

    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}

        vectorizer_kwargs = {
            'ngram_range': (1, 3),
            'min_df': 2,
            'max_df': .9
        }
        if self.stemmer:
            print("using stemmer in vectorizer")
            vectorizer_kwargs["tokenizer"] = self.stemming_tokenizer

        self.tfidf_vectorizer = TfidfVectorizer(**vectorizer_kwargs).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self):
        path = self.__class__._get_model_path(stem=self.stemmer)
        with open(path, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

    @staticmethod
    def _get_model_path(stem=False):
        if stem:
            return f"stem-{MODEL_PATH}"
        return MODEL_PATH

    @classmethod
    def load(cls, stem=False):
        path = cls._get_model_path(stem)
        # print(f"Loading {path} guesser")
        with open(path, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            return guesser


def create_app(enable_batch=True, stem=True):
    tfidf_guesser = TfidfGuesser.load(stem=stem)
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(tfidf_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(tfidf_guesser, questions)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
@click.option('--stem', default=False, is_flag=True)
def train(stem):
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    tfidf_guesser = TfidfGuesser(stem=stem)
    tfidf_guesser.train(dataset.training_data())
    tfidf_guesser.save()


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
def download(local_qanta_prefix):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix)


if __name__ == '__main__':
    cli()
