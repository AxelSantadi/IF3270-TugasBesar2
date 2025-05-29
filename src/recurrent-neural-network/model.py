import numpy as np
from layers import Embedding, RNN, Dropout, Dense, softmax, BiRNN

class SimpleRNNModel:
    def __init__(self,
                 vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate=0.5, num_layers=1, bidirectional=False,
                 return_sequences_list=None):
        
        self.embed = Embedding(vocab_size, embed_dim)
        self.dropout = Dropout(dropout_rate)
        self.rnns = []
        
        current_input_dim = embed_dim
        if return_sequences_list is None:
            return_sequences_list = [True] * (num_layers - 1) + [False] if num_layers > 0 else []

        if len(return_sequences_list) != num_layers:
            raise ValueError("Length of return_sequences_list must match num_layers")

        for i in range(num_layers):
            is_last_rnn = (i == num_layers - 1)
            current_return_sequences = return_sequences_list[i]
            
            if bidirectional:
                rnn_layer = BiRNN(current_input_dim, hidden_dim, return_sequences=current_return_sequences)
                current_input_dim = hidden_dim * 2
            else:
                rnn_layer = RNN(current_input_dim, hidden_dim, return_sequences=current_return_sequences)
                current_input_dim = hidden_dim
            
            self.rnns.append(rnn_layer)
            if not current_return_sequences:
                pass


        self.dense = Dense(current_input_dim, num_classes)
        self._training = True

    def forward(self, x, training=None):
        if training is None:
            training = self._training

        self.dropout.set_training_mode(training)

        h = self.embed.forward(x)
        
        for i, rnn_layer in enumerate(self.rnns):
            if h.ndim == 2 and i > 0 :
                h = h[:, np.newaxis, :]

            h = rnn_layer.forward(h)

        if h.ndim == 3 and self.rnns and not self.rnns[-1].return_sequences:
            pass


        h = self.dropout.forward(h, training) 
        logits = self.dense.forward(h)
        return softmax(logits)

    def set_training_mode(self, training: bool):
        self._training = training
        self.dropout.set_training_mode(training) 

    def load_keras_weights(self, keras_weights):
        idx = 0
        self.embed.set_weights(keras_weights[idx])
        idx += 1

        for rnn_layer_custom in self.rnns:
            if isinstance(rnn_layer_custom, BiRNN):
                rnn_layer_custom.set_weights(
                    keras_weights[idx],      
                    keras_weights[idx+1],    
                    keras_weights[idx+2],    
                    keras_weights[idx+3],    
                    keras_weights[idx+4],    
                    keras_weights[idx+5]     
                )
                idx += 6
            elif isinstance(rnn_layer_custom, RNN):
                rnn_layer_custom.set_weights(
                    keras_weights[idx],      
                    keras_weights[idx+1],    
                    keras_weights[idx+2]     
                )
                idx += 3
            else:
                raise TypeError("Unknown RNN layer type in custom model during weight loading")

        self.dropout.set_weights(keras_weights[idx])
        idx += 1

        self.dense.set_weights(keras_weights[idx], keras_weights[idx+1])
        idx += 2

        if idx != len(keras_weights):
            print(f"Warning: Weight loading consumed {idx} weights, but Keras provided {len(keras_weights)} weights.")