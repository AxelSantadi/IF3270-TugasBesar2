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

    def set_weights(self, W):
        self.W = W

class RNNCell:
    def __init__(self, input_dim, hidden_dim):
        # Wx, Wh weight matrices and bias
        self.Wx = np.random.randn(input_dim,  hidden_dim) * np.sqrt(1. / input_dim)
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1. / hidden_dim)
        self.b  = np.zeros((hidden_dim,))

    def forward(self, x_t, h_prev):
        # x_t: (batch, input_dim), h_prev: (batch, hidden_dim)
        return tanh(x_t @ self.Wx + h_prev @ self.Wh + self.b)

    def set_weights(self, Wx, Wh, b):
        self.Wx = Wx
        self.Wh = Wh
        self.b = b

class RNN:
    def __init__(self, input_dim, hidden_dim, return_sequences=False): # Added return_sequences
        self.hidden_dim = hidden_dim
        self.cell = RNNCell(input_dim, hidden_dim)
        self.return_sequences = return_sequences

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
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
        self._training = True # Default to training mode

    def forward(self, x, training=None): # Allow overriding training mode
        if training is None:
            training = self._training # Use instance's training mode if not specified
            
        if not training or self.rate == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
        return x * mask
    
    def set_training_mode(self, training: bool): # Method to explicitly set training mode
        self._training = training

class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros((output_dim,))

    def forward(self, x):
        # x: (batch, input_dim)
        return x @ self.W + self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b
    
class BiRNN:
    def __init__(self, input_dim, hidden_dim, return_sequences=False): # Added return_sequences
        self.fwd = RNN(input_dim, hidden_dim, return_sequences=return_sequences)
        self.bwd = RNN(input_dim, hidden_dim, return_sequences=return_sequences)
        self.return_sequences = return_sequences

    def forward(self, x):
        # x: (batch, seq, dim)
        h_f = self.fwd.forward(x)
        h_b = self.bwd.forward(x[:, ::-1, :])  # reverse seq

        if self.return_sequences:
            # For BiRNN with return_sequences=True, Keras concatenates along the feature axis for each time step
            # h_b needs to be un-reversed in its sequence dimension if we were to match Keras exactly
            # For now, if return_sequences=True, we might need to adjust h_b's time ordering before concat
            # However, if return_sequences=False, this is fine.
            # Keras BiRNN(return_sequences=True) output shape: (batch, seq_len, hidden_dim*2)
            # Keras BiRNN(return_sequences=False) output shape: (batch, hidden_dim*2)
            # Current h_b is (batch, seq_len, hidden_dim) but time-reversed.
            # If matching Keras for return_sequences=True, h_b should be (batch, seq_len, hidden_dim) with original time order for its hidden states.
            # This typically means the backward RNN processes reversed sequence, but its outputs are stored in natural order.
            # For simplicity with return_sequences=True in custom BiRNN, one might need to reverse h_b back: h_b = h_b[:, ::-1, :]
            # However, the task asked to match the Keras model which uses return_sequences=False for its single RNN layer.
            if h_b.ndim > 2 and h_f.ndim > 2: # if both are sequences
                 return np.concatenate([h_f, h_b[:, ::-1, :]], axis=-1) # reverse backward sequence back for concat
            else: # Should not happen if both are sequences
                 return np.concatenate([h_f, h_b], axis=-1)


        # Gabungkan dua arah (final states if return_sequences=False)
        return np.concatenate([h_f, h_b], axis=-1)

    def set_weights(self, Wx_f, Wh_f, b_f, Wx_b, Wh_b, b_b):
        self.fwd.set_weights(Wx_f, Wh_f, b_f)
        self.bwd.set_weights(Wx_b, Wh_b, b_b)