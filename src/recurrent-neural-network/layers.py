import numpy as np

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        # initialize embedding matrix
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
    def forward(self, x):
        # x: (batch, seq_len) of int indices
        return self.W[x]

class RNNCell:
    def __init__(self, input_dim, hidden_dim):
        # Wx, Wh weight matrices and bias
        self.Wx = np.random.randn(input_dim,  hidden_dim) * np.sqrt(1. / input_dim)
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1. / hidden_dim)
        self.b  = np.zeros((hidden_dim,))
    def forward(self, x_t, h_prev):
        # x_t: (batch, input_dim), h_prev: (batch, hidden_dim)
        return tanh(x_t @ self.Wx + h_prev @ self.Wh + self.b)

class RNN:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.cell = RNNCell(input_dim, hidden_dim)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch, seq_len, _ = x.shape
        h = np.zeros((batch, self.hidden_dim))
        for t in range(seq_len):
            h = self.cell.forward(x[:, t, :], h)
        return h

class Dropout:
    def __init__(self, rate):
        self.rate = rate
    def forward(self, x, training=False):
        if not training or self.rate == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
        return x * mask

class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros((output_dim,))
    def forward(self, x):
        # x: (batch, input_dim)
        return x @ self.W + self.b
    
class BiRNN:
    def __init__(self, input_dim, hidden_dim):
        self.fwd = RNN(input_dim, hidden_dim)
        self.bwd = RNN(input_dim, hidden_dim)
    def forward(self, x):
        # x: (batch, seq, dim)
        h_f = self.fwd.forward(x)
        h_b = self.bwd.forward(x[:, ::-1, :])  # reverse seq
        # Gabungkan dua arah
        return np.concatenate([h_f, h_b], axis=-1)