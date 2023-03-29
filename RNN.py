import numpy as np
from numpy.random import randn


class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):
        # weights

        self.last_hidden = None
        self.last_inputs = None
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000
        # biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):

        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hidden = {0: h}

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hidden[i + 1] = h

        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn=2e-2):
        n = len(self.last_inputs)

        d_Why = d_y @ self.last_hidden[n].T
        d_h = self.Why.T @ d_y

        d_Wxh = np.zeros(self.Wxh.shape)
        d_Whh = np.zeros(self.Whh.shape)

        d_bh = np.zeros(self.bh.shape)
        d_by = d_y

        for i in reversed(range(n)):
            temp = ((1 - self.last_hidden[i + 1] ** 2) * d_h)
            d_Wxh += temp @ self.last_inputs[i].T
            d_Whh += temp @ self.last_hidden[i].T
            d_bh += temp
            d_h = self.Whh @ temp

        for d in [d_Wxh, d_Whh, d_Why, d_by, d_bh]:
            np.clip(d, -1, 1, out=d)

        self.Wxh -= learn * d_Wxh
        self.Whh -= learn * d_Whh
        self.Why -= learn * d_Why

        self.by -= learn * d_by
        self.bh -= learn * d_bh

        return
