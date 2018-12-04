#===========================================================================
# Imports
#===========================================================================

import random
import torch
from nltk import word_tokenize
from dan import DanModel, DanEncoder, vectorize


#===========================================================================
# Load datasets
#===========================================================================

if __name__ == '__main__':
    stuff = torch.load("lookups.pt")
    word2ind = stuff["word2ind"]
    ind2word = stuff["ind2word"]
    idx2ans = stuff["idx2ans"]
    ans2idx = stuff["ans2idx"]
    data = stuff["train_data"]

    model = torch.load("dan.pt")
    model.eval()
    print(model)

    results = []
    for i in range(50):
        question = random.choice(data)

        tokenized_text = word_tokenize(question["text"])
        vv = vectorize((tokenized_text, question["page"]), word2ind)[0]

        logits = model(torch.LongTensor([vv]),torch.FloatTensor([len(vv)]))
        ans_logits, ans_idx = logits.topk(5)

        print("\nAnswer: {} ({})".format(question["page"], ans2idx[question["page"]]))
        print("Guesses: {}".format([idx2ans[ii.item()] for ii in ans_idx[0]]))
        print("Logits: {}".format(ans_logits[0].data.tolist()))
        print("Indexes: {}".format(ans_idx[0].data.tolist()))

        ans_idx = ans_idx.data.numpy()[0][0]
        result = idx2ans[ans_idx] == question["page"]
        results.append(result)


    acc = sum([1 for _ in results if _]) / len(results)
    print("\nAccuracy: {}".format(acc))
