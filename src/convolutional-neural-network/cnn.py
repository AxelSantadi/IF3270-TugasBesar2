import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(num_conv_layers=2, filters_per_layers=[32, 64]):
    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))

    for i in range(num_conv_layers):
        model.add(layers.Conv2D(filters=filters_per_layers[i], kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

