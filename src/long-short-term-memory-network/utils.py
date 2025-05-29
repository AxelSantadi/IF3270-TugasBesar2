import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder

def load_nusax_data(max_tokens=20000, seq_length=100):
    """
    Load and preprocess NusaX Indonesian sentiment dataset.
    Returns tokenized numpy arrays and label arrays.
    """
    BASE = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian"
    df_train = pd.read_csv(f"{BASE}/train.csv")
    df_val   = pd.read_csv(f"{BASE}/valid.csv")
    df_test  = pd.read_csv(f"{BASE}/test.csv")

    # Encode labels from strings to integers
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['label'])
    y_val   = le.transform(df_val['label'])
    y_test  = le.transform(df_test['label'])

    # TextVectorization tokenizer
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_sequence_length=seq_length,
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        output_mode='int'
    )
    vectorizer.adapt(df_train['text'].values)

    # Tokenize and cast to numpy int32
    tok_train = tf.cast(vectorizer(df_train['text'].values), tf.int32).numpy()
    tok_val   = tf.cast(vectorizer(df_val['text'].values),   tf.int32).numpy()
    tok_test  = tf.cast(vectorizer(df_test['text'].values),  tf.int32).numpy()

    num_classes = len(le.classes_)
    vocab_size  = max_tokens
    return tok_train, y_train, tok_val, y_val, tok_test, y_test, vocab_size, num_classes, vectorizer