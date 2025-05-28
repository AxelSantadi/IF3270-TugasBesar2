import numpy as np
from layers import Embedding, RNN, Dropout, Dense, softmax, BiRNN

class SimpleRNNModel:
    def __init__(self,
                 vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate=0.5, num_layers=1, bidirectional=False,
                 return_sequences_list=None): # return_sequences_list for stacked RNNs
        
        self.embed = Embedding(vocab_size, embed_dim)
        self.dropout = Dropout(dropout_rate)
        self.rnns = []
        
        current_input_dim = embed_dim
        if return_sequences_list is None:
            # Default: last RNN layer has return_sequences=False, others True if num_layers > 1
            return_sequences_list = [True] * (num_layers - 1) + [False] if num_layers > 0 else []

        if len(return_sequences_list) != num_layers:
            raise ValueError("Length of return_sequences_list must match num_layers")

        for i in range(num_layers):
            is_last_rnn = (i == num_layers - 1)
            # Keras: return_sequences for all but last RNN if stacking before Dense
            # For custom: use return_sequences_list
            current_return_sequences = return_sequences_list[i]
            
            if bidirectional:
                rnn_layer = BiRNN(current_input_dim, hidden_dim, return_sequences=current_return_sequences)
                current_input_dim = hidden_dim * 2 # Output of BiRNN is 2 * hidden_dim
            else:
                rnn_layer = RNN(current_input_dim, hidden_dim, return_sequences=current_return_sequences)
                current_input_dim = hidden_dim # Output of RNN is hidden_dim
            
            self.rnns.append(rnn_layer)
            # If the RNN layer does not return sequences, the next layer (if any, or Dense) gets a 2D input
            # If it returns sequences, the next RNN layer gets a 3D input
            if not current_return_sequences: # If output is (batch, features)
                # This input_dim is for the *next* layer if it were an RNN.
                # For Dense layer, it will take this current_input_dim.
                pass


        self.dense = Dense(current_input_dim, num_classes) # Input dim for Dense is the output of the last RNN
        self._training = True # Default for model's training mode

    def forward(self, x, training=None): # Allow overriding training mode
        if training is None:
            training = self._training

        self.dropout.set_training_mode(training) # Set training mode for dropout layer

        h = self.embed.forward(x)  # (batch, seq_len, embed_dim)
        
        for i, rnn_layer in enumerate(self.rnns):
            # Output of previous layer is h
            # For stacked RNNs, if previous RNN had return_sequences=False,
            # h would be (batch, features). The current RNN layer expects (batch, seq_len, features).
            # This was handled by `if h.ndim == 2: h = h[:, None, :]`
            # This logic is tricky if we mix return_sequences=True/False in Keras and try to match.
            # For now, assuming Keras stack: Emb -> RNN(ret_seq=T) -> ... -> RNN(ret_seq=F) -> Dense
            
            # If h became 2D from a previous RNN layer (return_sequences=False)
            # and this is not the first RNN layer, it implies an architectural mismatch
            # or a specific design choice to process final state as a sequence of 1.
            if h.ndim == 2 and i > 0 : # Output from a non-sequence returning RNN
                 # This means the previous RNN layer in the stack had return_sequences=False.
                 # If the current RNN expects a sequence, we reshape.
                 # This matches Keras behavior if you connect RNN(ret_seq=F) to RNN().
                 h = h[:, np.newaxis, :]


            h = rnn_layer.forward(h)
        
        # h is now output of last RNN layer.
        # If last RNN return_sequences=True, h is (batch, seq_len, features)
        # If last RNN return_sequences=False, h is (batch, features)
        # Dense layer expects (batch, features).
        # If h is (batch, seq_len, features), usually only the last time step is taken for Dense,
        # or a Flatten/GlobalPooling layer is used.
        # The Keras models in PDF typically have return_sequences=False on the last RNN before Dense.
        if h.ndim == 3 and self.rnns and not self.rnns[-1].return_sequences:
             # This case should ideally not be hit if logic is correct,
             # as return_sequences=False on last RNN makes h 2D.
             # If it is hit, it implies an issue or specific design.
             # For safety, if h is 3D and last RNN was supposed to be 2D output:
             # h = h[:, -1, :] # Take last time step (not robust for all cases)
             pass


        h = self.dropout.forward(h, training) # Pass training flag to dropout
        logits = self.dense.forward(h)
        return softmax(logits)

    def set_training_mode(self, training: bool):
        self._training = training
        self.dropout.set_training_mode(training) # Also propagate to dropout layer

    def load_keras_weights(self, keras_weights):
        """
        Loads weights from a Keras model.
        The order of weights in keras_weights list must match the Keras model architecture.
        Assumes SimpleRNNModel architecture matches the Keras model structure
        (e.g., number of RNN layers, bidirectional, hidden_dim).
        """
        idx = 0
        # 1. Embedding layer
        self.embed.set_weights(keras_weights[idx])
        idx += 1

        # 2. RNN layers
        for rnn_layer_custom in self.rnns:
            if isinstance(rnn_layer_custom, BiRNN):
                # BiRNN Keras weights: Wx_f, Wh_f, b_f, Wx_b, Wh_b, b_b
                rnn_layer_custom.set_weights(
                    keras_weights[idx],      # Wx_f
                    keras_weights[idx+1],    # Wh_f
                    keras_weights[idx+2],    # b_f
                    keras_weights[idx+3],    # Wx_b
                    keras_weights[idx+4],    # Wh_b
                    keras_weights[idx+5]     # b_b
                )
                idx += 6
            elif isinstance(rnn_layer_custom, RNN):
                # RNN Keras weights: Wx, Wh, b
                rnn_layer_custom.set_weights(
                    keras_weights[idx],      # Wx
                    keras_weights[idx+1],    # Wh
                    keras_weights[idx+2]     # b
                )
                idx += 3
            else:
                raise TypeError("Unknown RNN layer type in custom model during weight loading")

        # 3. Dropout layer (no weights)

        # 4. Dense layer
        self.dense.set_weights(keras_weights[idx], keras_weights[idx+1])
        idx += 2

        if idx != len(keras_weights):
            print(f"Warning: Weight loading consumed {idx} weights, but Keras provided {len(keras_weights)} weights.")