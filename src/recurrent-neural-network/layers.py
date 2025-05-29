import numpy as np

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
    
    def forward(self, x):
        return self.W[x]

    def set_weights(self, W):
        self.W = W

class RNNCell:
    def __init__(self, input_dim, hidden_dim):
        self.Wx = np.random.randn(input_dim,  hidden_dim) * np.sqrt(1. / input_dim)
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1. / hidden_dim)
        self.b  = np.zeros((hidden_dim,))

    def forward(self, x_t, h_prev):
        return tanh(x_t @ self.Wx + h_prev @ self.Wh + self.b)

    def set_weights(self, Wx, Wh, b):
        self.Wx = Wx
        self.Wh = Wh
        self.b = b

class RNN:
    def __init__(self, input_dim, hidden_dim, return_sequences=False): 
        self.hidden_dim = hidden_dim
        self.cell = RNNCell(input_dim, hidden_dim)
        self.return_sequences = return_sequences

    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = np.zeros((batch, self.hidden_dim))
        
        if self.return_sequences:
            outputs = np.zeros((batch, seq_len, self.hidden_dim))
        
        for t in range(seq_len):
            h = self.cell.forward(x[:, t, :], h)
            if self.return_sequences:
                outputs[:, t, :] = h
        
        return outputs if self.return_sequences else h
        
    def set_weights(self, Wx, Wh, b):
        self.cell.set_weights(Wx, Wh, b)

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self._training = True 

    def forward(self, x, training=None): 
        if training is None:
            training = self._training 
            
        if not training or self.rate == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
        return x * mask
    
    def set_training_mode(self, training: bool): 
        self._training = training

class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros((output_dim,))

    def forward(self, x):
        return x @ self.W + self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b
    
class BiRNN:
    def __init__(self, input_dim, hidden_dim, return_sequences=False): 
        self.fwd = RNN(input_dim, hidden_dim, return_sequences=return_sequences)
        self.bwd = RNN(input_dim, hidden_dim, return_sequences=return_sequences)
        self.return_sequences = return_sequences

    def forward(self, x):
        h_f = self.fwd.forward(x)
        h_b = self.bwd.forward(x[:, ::-1, :])  

        if self.return_sequences:
            if h_b.ndim > 2 and h_f.ndim > 2: 
                 return np.concatenate([h_f, h_b[:, ::-1, :]], axis=-1) 
            else: 
                 return np.concatenate([h_f, h_b], axis=-1)
        
        return np.concatenate([h_f, h_b], axis=-1)

    def set_weights(self, Wx_f, Wh_f, b_f, Wx_b, Wh_b, b_b):
        self.fwd.set_weights(Wx_f, Wh_f, b_f)
        self.bwd.set_weights(Wx_b, Wh_b, b_b)