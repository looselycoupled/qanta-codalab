#===========================================================================
# Imports
#===========================================================================

import random

import torch

from dan import DanModel, DanEncoder


#===========================================================================
# Load datasets
#===========================================================================

if __name__ == '__main__':
    stuff = torch.load("lookups.pt")
    word2ind = stuff["word2ind"]
    idx2ans = stuff["idx2ans"]
    data = stuff["train_data"]

    model = torch.load("alt-dan-qa-aws.pt")
    print(model)

    results = []
    for i in range(10):
        question = random.choice(data)

        ii = [word2ind[w] for w in question["text"].split() if w in word2ind]
        _, ans_idx = model(torch.LongTensor([ii]),torch.FloatTensor([1])).topk(5)

        print("\nAnswer: {}".format(question["page"]))
        print("Guesses: {}".format([idx2ans[ii.item()] for ii in ans_idx[0]]))

        ans_idx = ans_idx.data.numpy()[0][0]
        result = idx2ans[ans_idx] == question["page"]
        results.append(result)


    acc = sum([1 for _ in results if _]) / len(results)
    print("\nAccuracy: {}".format(acc))
