import tqdm
import random
import torch
from dan import DanGuesser, DanModel, DanEncoder, datasets

if __name__ == '__main__':
    NUM_TESTS = 1000
    results = []
    guesser = DanGuesser()
    print("INFO: model loaded ")

    # load lookups, questions
    lookups = torch.load("lookups.pt")
    train_data, dev_data, test_data = datasets()
    print("INFO: test data loaded ")

    # test
    for i in tqdm.tqdm(range(NUM_TESTS)):
        question = random.choice(test_data)
        ques = question["text"]
        ans = question["page"]

        results.append(guesser.guess(ques) == ans)


    acc = sum([1 for _ in results if _]) / len(results)
    print("Number of tests (from test set): {}".format(NUM_TESTS))
    print("Accuracy: {}".format(acc))
