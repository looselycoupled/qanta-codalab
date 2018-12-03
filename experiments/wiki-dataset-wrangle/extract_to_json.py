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
        self.group_size = group_size
        self.instance = instance
        self.lookups = {}

    def _doc_iterator(self):
        files = sorted(glob(self.path + "/*/*"))

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

        for _ in range(3):
            response = requests.get(api_url.format(curid))
            if response.status_code == 200:
                break
            print("WARN: _fetch_title encountered status code {}".format(response.status_code))
            time.sleep(2)

        if response.status_code != 200:
            print("WARN: could not retrieve metainfo for curid={}".format(curid))
            raise Exception("Wikimedia API Request Limiter")

        data = response.json()
        canonicalurl = data["query"]["pages"][curid]["canonicalurl"]
        return curid, canonicalurl.split("/")[-1]


    @property
    def docs(self):
        for doc in self._doc_iterator():
            try:
                curid, ans = self._fetch_title(doc["url"])
                self.lookups[curid] = ans
            except Exception as e:
                print("WARN: sleeping for 5 seconds to cool down")
                time.sleep(5)
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

    with open("curid_lookups.{}.json".format(instance), "w") as f:
        f.write(json.dumps(extractor.lookups))
