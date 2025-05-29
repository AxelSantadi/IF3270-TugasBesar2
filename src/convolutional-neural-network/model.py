import tensorflow as tf
from layers import Conv2DLayer, MaxPooling2DLayer, AveragePooling2DLayer,FlattenLayer, DenseLayer

class CNNFromScratch:
    def __init__(self, keras_model):
        self.layers = []
        self.load_weights_from_keras(keras_model)
    
    def load_weights_from_keras(self, keras_model):
        """
        Load weights dari model Keras yang sudah di-train
        """
        layer_idx = 0
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights = layer.get_weights()
                conv_layer = Conv2DLayer(
                    weights=weights[0],  # kernel weights
                    biases=weights[1],   # biases
                    activation='relu',
                    padding=layer.padding
                )
                self.layers.append(conv_layer)
                
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                pool_layer = MaxPooling2DLayer(
                    pool_size=layer.pool_size[0],
                    strides=layer.strides[0]
                )
                self.layers.append(pool_layer)
                
            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                pool_layer = AveragePooling2DLayer(
                    pool_size=layer.pool_size[0],
                    strides=layer.strides[0]
                )
                self.layers.append(pool_layer)
                
            elif isinstance(layer, tf.keras.layers.Flatten):
                flatten_layer = FlattenLayer()
                self.layers.append(flatten_layer)
                
            elif isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                activation = None
                if hasattr(layer, 'activation'):
                    if layer.activation.__name__ == 'relu':
                        activation = 'relu'
                    elif layer.activation.__name__ == 'softmax':
                        activation = 'softmax'
                
                dense_layer = DenseLayer(
                    weights=weights[0],  # weight matrix
                    biases=weights[1],   # biases
                    activation=activation
                )
                self.layers.append(dense_layer)
    
    def predict(self, x):
        """
        Forward propagation
        """
        output = x
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                output = layer.forward(output)
        return output