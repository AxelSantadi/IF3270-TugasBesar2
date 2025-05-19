import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Bidirectional, Dropout, Dense
from sklearn.metrics import f1_score
from utils import load_nusax_data
from model import SimpleRNNModel

# ----- Hyperparameters -----
MAX_TOKENS    = 20000
SEQ_LENGTH    = 100
EMBED_DIM     = 128
DROPOUT_RATE  = 0.5
BATCH_SIZE    = 50
EPOCHS        = 10

# editable hyperparameters
HIDDEN_DIM    = 32
NUM_LAYERS    = 5  
BIDIRECTIONAL = True

# ----- Load and preprocess data -----
tok_train, y_train, tok_val, y_val, tok_test, y_test, vocab_size, num_classes = \
    load_nusax_data(MAX_TOKENS, SEQ_LENGTH)

# ----- Build Keras model -----
keras_model = Sequential()
# Embedding
keras_model.add(Embedding(
    input_dim=vocab_size,
    output_dim=EMBED_DIM,
    input_length=SEQ_LENGTH,
    name='embedding'
))
# RNN layer (bidirectional or unidirectional)
if BIDIRECTIONAL:
    keras_model.add(Bidirectional(
        SimpleRNN(HIDDEN_DIM, return_sequences=False),
        name='bidir_rnn'
    ))
else:
    keras_model.add(SimpleRNN(HIDDEN_DIM, return_sequences=False, name='rnn'))
# Dropout
keras_model.add(Dropout(DROPOUT_RATE, name='dropout'))
# Output
keras_model.add(Dense(num_classes, activation='softmax', name='output'))

# Compile model
keras_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Create folder for weights and plots
os.makedirs('weights', exist_ok=True)

# ----- Training -----
history = keras_model.fit(
    tok_train, y_train,
    validation_data=(tok_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)
# Save weights
keras_model.save_weights('weights/keras_rnn.weights.h5')

# ----- Evaluation on test set -----
y_pred_prob = keras_model.predict(tok_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_prob, axis=1)
f1_test = f1_score(y_test, y_pred, average='macro')
print(f"[Keras] Test Macro F1-score: {f1_test:.4f}")

# Load and forward with custom model, only if desired
custom = SimpleRNNModel(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_classes,
    dropout_rate=DROPOUT_RATE,
    num_layers=NUM_LAYERS,
    bidirectional=BIDIRECTIONAL
)

custom_weights = keras_model.get_weights()
probs_custom = custom.forward(tok_test)
preds_custom = np.argmax(probs_custom, axis=1)
f1_custom = f1_score(y_test, preds_custom, average='macro')
print(f"[Custom] Test Macro F1-score: {f1_custom:.4f}")