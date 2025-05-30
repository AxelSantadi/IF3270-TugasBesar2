    # Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    # Bidirectional(LSTM(units=lstm_units)),
    # Dropout(rate=dropout_rate),
    # Dense(units=num_classes, activation='softmax')

import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        # initialize embedding matrix
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
    
    def forward(self, x):
        # x: (batch, seq_len) of int indices
        return self.W[x]

    def set_weights(self, W):
        self.W = W
        
class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, x): # Allow overriding training mode
        if self.rate == 0.0:
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

    def set_weights(self, W, b):
        self.W = W
        self.b = b
        

class LSTM:
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim  = D
        hidden_dim = H
        """
        self.D = input_dim
        self.H = hidden_dim
        self.W = None
        self.U = None
        self.b = None

    def set_weights(self, W, U, b):
        """
        W: shape (D, 4H)    Input weights
        U: shape (H, 4H)    Short term memory weights
        b: shape (4H,)      Bias for all 
        """
        assert W.shape == (self.D, 4*self.H)
        assert U.shape == (self.H, 4*self.H)
        assert b.shape == (4*self.H,)
        self.W = W
        self.U = U
        self.b = b

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x_seq, h0=None, c0=None):
        """
        x_seq: array of shape (T, D)
        h0, c0: optional initial states, each (H,)
        
        Returns:
          h_seq: array (T, H) of hidden states
          (h_T, c_T): final hidden & cell states
        """
        T, D = x_seq.shape
        
        assert D == self.D, "Input dimension mismatch"
        assert self.W is not None, "Weights not set!"
        
        h = np.zeros(self.H) if h0 is None else h0 # Short term memory
        c = np.zeros(self.H) if c0 is None else c0 # Long term memory
        h_seq = np.zeros((T, self.H))
        
        for t in range(T):
            x_t = x_seq[t]
            z = x_t @ self.W + h @ self.U + self.b # Apparently @ is for matrix multiplication
            z_i, z_f, z_c, z_o = np.split(z, 4) # h5 format stores weights liek this : 
            i_t = self._sigmoid(z_i) # input gate coef
            f_t = self._sigmoid(z_f) # forget gate coef
            c_hat = np.tanh(z_c)
            o_t = self._sigmoid(z_o)
            c = f_t * c + i_t * c_hat # new long term memory
            h = o_t * np.tanh(c) # new short term memory
            h_seq[t] = h
        
        return h_seq, (h, c)

class BidirectionalLSTM:
    def __init__(self, input_dim, hidden_dim):
        """
        Wraps two LSTM cells: forward and backward.
        """
        self.forward_lstm = LSTM(input_dim, hidden_dim)
        self.backward_lstm = LSTM(input_dim, hidden_dim)
        self.H = hidden_dim

    def set_weights(self, W_f, U_f, b_f, W_b, U_b, b_b):
        """
        Load weights for forward and backward LSTMs.
        """
        self.forward_lstm.set_weights(W_f, U_f, b_f)
        self.backward_lstm.set_weights(W_b, U_b, b_b)

    def forward(self, x_seq):
        """
        x_seq: array of shape (T, D)
        
        Returns:
          h_seq: array (T, 2H) of concatenated forward+backward hidden states
          (h_final, c_final): final states each of shape (2H,)
        """
        # Forward pass
        h_seq_f, (h_f, c_f) = self.forward_lstm.forward(x_seq)
        # Backward pass on reversed sequence
        x_rev = x_seq[::-1]
        h_seq_b_rev, (h_b_rev, c_b_rev) = self.backward_lstm.forward(x_rev)
        # Reverse backward outputs to align
        h_seq_b = h_seq_b_rev[::-1]
        # Concatenate time-wise
        h_seq = np.concatenate([h_seq_f, h_seq_b], axis=1)  # (T, 2H)
        # Final states concatenation
        h_final = np.concatenate([h_f, h_b_rev], axis=0)    # (2H,)
        c_final = np.concatenate([c_f, c_b_rev], axis=0)
        return h_seq, (h_final, c_final)

