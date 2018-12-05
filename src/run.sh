#!/usr/bin/env bash

pip install torchtext gensim

python -c "import nltk; nltk.download('punkt')"

python -m qanta.server web
