import numpy as np
from layers import Embedding, RNN, Dropout, Dense, softmax, BiRNN

class SimpleRNNModel:
    def __init__(self,
                 vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate=0.5, num_layers=1, bidirectional=False):
        # Embedding layer
        self.embed = Embedding(vocab_size, embed_dim)
        # Dropout
        self.dropout = Dropout(dropout_rate)
        # Build RNN layers list
        self.rnns = []
        for i in range(num_layers):
            input_dim = embed_dim if i == 0 else (hidden_dim * (2 if bidirectional else 1))
            if bidirectional:
                self.rnns.append(BiRNN(input_dim, hidden_dim))
            else:
                self.rnns.append(RNN(input_dim, hidden_dim))
        # Dense layer input dim depends on direction
        final_dim = hidden_dim * (2 if bidirectional else 1)
        self.dense = Dense(final_dim, num_classes)

    def forward(self, x, training=False):
        # x: (batch, seq_len)
        # Embed tokens
        h = self.embed.forward(x)  # (batch, seq_len, embed_dim)
        # Pass through RNN layers
        for rnn in self.rnns:
            # Jika h hanya 2D (batch, features), ubah ke 3D (batch, seq=1, features)
            if h.ndim == 2:
                h = h[:, None, :]
            h = rnn.forward(h)
        # Dropout + Dense
        h = self.dropout.forward(h, training)
        logits = self.dense.forward(h)
        return softmax(logits)