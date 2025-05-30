import h5py
from layers import BidirectionalLSTM, Dense, Dropout, Embedding
import numpy as np

class ManualLSTMModel:
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 lstm_units: int,
                 dropout_rate: float,
                 num_classes: int):
        # 1) Layers
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.bilstm    = BidirectionalLSTM(input_dim=embedding_dim,
                                           hidden_dim=lstm_units)
        self.dropout   = Dropout(rate=dropout_rate)
        # 2*lstm_units because bidirectional concat
        self.dense     = Dense(input_dim=2 * lstm_units,
                               output_dim=num_classes)
        
    # Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    # Bidirectional(LSTM(units=lstm_units)),
    # Dropout(rate=dropout_rate),
    # Dense(units=num_classes, activation='softmax')
    def load_weights(self, filename: str):
        with h5py.File(filename, 'r') as f:
            # Embedding
            W_emb = f['layers']['embedding']['vars']['0'][()]
            self.embedding.set_weights(W_emb)

            # Bidirectional LSTM
            Wf = f['layers']['bidirectional']['forward_layer']['cell']['vars']['0'][()]
            Uf = f['layers']['bidirectional']['forward_layer']['cell']['vars']['1'][()]
            bf = f['layers']['bidirectional']['forward_layer']['cell']['vars']['2'][()]
            Wb = f['layers']['bidirectional']['backward_layer']['cell']['vars']['0'][()]
            Ub = f['layers']['bidirectional']['backward_layer']['cell']['vars']['1'][()]
            bb = f['layers']['bidirectional']['backward_layer']['cell']['vars']['2'][()]
            self.bilstm.set_weights(W_f=Wf, U_f=Uf, b_f=bf,
                                    W_b=Wb, U_b=Ub, b_b=bb)

            # Dense (note: Dropout has no weights)
            W_dense = f['layers']['dense']['vars']['0'][()]
            b_dense = f['layers']['dense']['vars']['1'][()]
            self.dense.set_weights(W_dense, b_dense)
    
    def forward(self, x_batch: np.ndarray, training: bool = False) -> np.ndarray:
        """
        x_batch: shape (batch_size, seq_len), dtype=int
        returns: probabilities, shape (batch_size, num_classes)
        """
        # 1) Embedding: (batch, seq_len, embedding_dim)
        embedded = self.embedding.forward(x_batch)

        # 2) Bidirectional LSTM
        # we need to run time-steps for each example in the batch
        batch_outputs = []
        for seq in embedded:
            # seq: (seq_len, embedding_dim)
            h_seq, (h_final, _) = self.bilstm.forward(seq)
            # h_final: (2*lstm_units,)
            batch_outputs.append(h_final)
        h_stack = np.stack(batch_outputs, axis=0)  # (batch, 2*lstm_units)

        # 3) Dropout
        dropped = self.dropout.forward(h_stack, training=training)

        # 4) Dense + softmax
        logits = self.dense.forward(dropped)      # (batch, num_classes)
        # simple stable softmax
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        return probs


# def make_model(filename):
    

def load_weights(filename):
    with h5py.File(filename,'r') as f:
        print(f['layers']['bidirectional']['forward_layer']['cell']['vars']['2'][()][:4])
        
# load_weights("weightbruh.weights.h5")