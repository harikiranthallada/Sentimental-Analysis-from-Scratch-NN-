import numpy as np
import random
from RNN import RNN
from Data import train_data, test_data

vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
len_vocab = len(vocab)
word_to_id = {x: i for i, x in enumerate(vocab)}
print(word_to_id)


def create_input(text):
    inputs = []
    for i in text.split(" "):
        v = np.zeros((len(vocab), 1))

        v[word_to_id[i]] = 1
        inputs.append(v)
    return inputs


def softmax(s):
    return np.exp(s) / sum(np.exp(s))


r = RNN(len(vocab), 2)


def process_data(data, backprop=True):
    items = list(data.items())
    random.shuffle(items)
    loss = 0
    num_correct = 0
    for x, c in items:
        text = x
        target = int(c)
        inputs = create_input(text)
        y, _ = r.forward(inputs)
        p = softmax(y)

        if target == np.argmax(p):
            num_correct += 1
        loss -= np.log(p[target])
        p[target] -= 1
        d_y = p
        r.backprop(d_y)
    return loss / len(data), num_correct / len(data)


for i in range(1000):
    x, y = process_data(train_data)
    if i % 100 == 99:
        print("Epoch : ", i+1)
        print("train loss = ", x, " train accuracy = ", y)
        x, y = process_data(test_data, backprop=False)
        print("test loss = ", x, " test accuracy = ", y, "\n")

