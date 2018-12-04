import random
import torch
from dan import DanGuesser, DanModel, DanEncoder

if __name__ == '__main__':
    NUM_TESTS = 100
    results = []
    guesser = DanGuesser()

    lookups = torch.load("lookups.pt")
    data = lookups["train_data"]


    for i in range(NUM_TESTS):
        question = random.choice(data)
        ques = question["text"]
        ans = question["page"]

        results.append(guesser.guess(ques) == ans)


    acc = sum([1 for _ in results if _]) / len(results)
    print("Number of tests: {}".format(NUM_TESTS))
    print("Accuracy: {}".format(acc))
