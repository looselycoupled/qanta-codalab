import json
import os
from glob import glob
from pprint import pprint
import requests
import urllib.parse as urlparse

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


PATH = "/Users/allen/Projects/wikiextractor/extracted"
def files():
    return sorted(glob(PATH + "/*/*"))


class WikidataIterator(object):
    def __init__(self, path, group_size, instance):
        self.path = path
        self.i_to_ans = {}
        self.group_size = group_size
        self.instance = instance

    def _doc_iterator(self, path):
        files = sorted(glob(path + "/*/*"))[:100]

        for idx, path in enumerate(files):
            if idx % self.group_size == self.instance:
                with open(path, "r") as f:
                    print(path)
                    for line in f:
                        item = json.loads(line)
                        yield item

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

    def _fetch_title(self, url):
        api_url = "https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids={}&inprop=url&format=json"

        parsed = urlparse.urlparse(url)
        curid = urlparse.parse_qs(parsed.query)["curid"][0]

        response = requests.get(api_url.format(curid))
        data = response.json()
        canonicalurl = data["query"]["pages"][curid]["canonicalurl"]
        return canonicalurl.split("/")[-1]


    @property
    def docs(self):
        for doc in self._doc_iterator(self.path):
            try:
                ans = self._fetch_title(doc["url"])
            except Exception as e:
                print(e)
                continue

            for para in self._para_iterator(doc["text"]):
                yield json.dumps({
                    "ans": ans,
                    "text": para
                })


if __name__ == '__main__':
    import sys
    import time

    group_size = int(sys.argv[1])
    instance = int(sys.argv[2])
    extractor = WikidataIterator(PATH, group_size, instance)

    start = time.time()
    with open("training_set.{}.json".format(instance), "w") as f:
        for item in extractor.docs:
            f.write(item + "\n")


    elapsed = int(time.time() - start)
    print("INFO: finished in {} seconds".format(elapsed))
