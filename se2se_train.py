from __future__ import print_function

import os
import sys
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, CuDNNLSTM, Input, Embedding, TimeDistributed, Dropout
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.backend.tensorflow_backend import set_session
from keras.utils.np_utils import to_categorical
from keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix


def read_and_prepare_data(data='node'):
    # Path to the data
    data_path = 'complete_data.tsv'
    # Read complete data
    df = pd.read_csv(data_path, sep="\t")

    # Split in Training and Validation data
    if data == 'node':
        # Training data: NoDe debatepedia all versions without neutral label
        # Validation data: NoDe procon
        dataset = df.loc[~df['org_dataset'].isin(['political'])].loc[df['label'].isin(['attack', 'support'])]
        data_train = dataset.loc[~dataset['org_dataset'].isin(['procon'])]
        data_val = dataset.loc[dataset['org_dataset'].isin(['procon'])]
    elif data == 'political':
        dataset = df.loc[df['org_dataset'].isin(['political'])].loc[df['label'].isin(['attack', 'support'])]
        data_train = dataset.iloc[:-200]
        data_val = dataset.iloc[-200:]
    else:
        print('Invalid dataset')
        sys.exit(-1)

    # Create and fit a Tokenizer for the data
    # (Maybe use another Tokenizer made for Online Discussions, instead of the default Keras Tokenizer?)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(np.concatenate((dataset['org'].values, dataset['response'].values), axis=-1))
    word_index = tokenizer.word_index

    # Create the training and validation data for the encoder and the decoder
    encoder_input_train = pad_sequences(tokenizer.texts_to_sequences(data_train['org'].values))
    decoder_input_train = pad_sequences(tokenizer.texts_to_sequences(data_train['response'].values))
    encoder_input_val = pad_sequences(tokenizer.texts_to_sequences(data_val['org'].values))
    decoder_input_val = pad_sequences(tokenizer.texts_to_sequences(data_val['response'].values))

    # Convert the labels to One-Hot Encoding (for now only attack and support labels are used)
    convert_dict = {"attack": 0, "support": 1, "unrelated": 2}
    decoder_labels_train = np.array([convert_dict[label] for label in data_train['label']])
    decoder_target_train = to_categorical(decoder_labels_train, len(set(data_train['label'])))
    decoder_labels_val = np.array([convert_dict[label] for label in data_val['label']])
    decoder_target_val = to_categorical(decoder_labels_val, len(set(data_train['label'])))

    # Print some information about our data
    print('Number of training samples:', len(encoder_input_train))
    print('Number of unique tokens:', len(word_index) + 1)
    print('Max sequence length for inputs:', encoder_input_train.shape[1])
    print('Max sequence length for outputs:', decoder_input_train.shape[1])
    print('Number of validation samples:', len(encoder_input_val))
    print('Max sequence length for inputs (val):', encoder_input_val.shape[1])
    print('Max sequence length for outputs (val):', decoder_input_val.shape[1])
    return encoder_input_train, decoder_input_train, decoder_target_train, decoder_labels_train, \
           encoder_input_val, decoder_input_val, decoder_target_val, decoder_labels_val, word_index


def read_glove_embeddings():
    # Read GloVe 42B 300d Word Embeddings
    print('Indexing word vectors.')
    embeddings_index = {}
    os.chdir('C:')
    with open('Users/Jannis/Documents/glove.42B.300d/glove.42B.300d.txt', encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    os.chdir('F:/SS19/BA/Twitter/twitter-test')
    return embeddings_index


def prepare_embedding_matrix(word_index, embeddings_index, max_num_words, embedding_dim):
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    num_words = min(max_num_words, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i > max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            print("Word not found: " + word)
    return embedding_matrix, num_words


def create_model(num_words, embedding_matrix, embedding_dim, num_labels, dropout_rate, latent_dim, num_layers=1):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                  embeddings_initializer=Constant(embedding_matrix), trainable=False)(encoder_inputs)
    encoder_dropout = (TimeDistributed(Dropout(rate=dropout_rate)))(encoder_embedding)
    encoder_lstm = CuDNNLSTM(latent_dim, return_sequences=True)(encoder_dropout)

    encoder_lstm2 = CuDNNLSTM(latent_dim, return_state=True)
    if num_layers == 1:
        encoder_outputs, state_h, state_c = encoder_lstm2(encoder_dropout)
    elif num_layers == 2:
        encoder_outputs, state_h, state_c = encoder_lstm2(encoder_lstm)
    else:
        print("Invalid number of layers")
        sys.exit(-1)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    # Input layer of the decoder :
    decoder_inputs = Input(shape=(None,))

    # Hidden layers of the decoder :
    decoder_embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                        embeddings_initializer=Constant(embedding_matrix), trainable=False)
    decoder_embedding = decoder_embedding_layer(decoder_inputs)

    decoder_dropout_layer = (TimeDistributed(Dropout(rate=dropout_rate)))
    decoder_dropout = decoder_dropout_layer(decoder_embedding)

    decoder_lstm_layer = CuDNNLSTM(latent_dim, return_sequences=True)
    decoder_lstm = decoder_lstm_layer(decoder_dropout, initial_state=encoder_states)

    decoder_lstm2 = CuDNNLSTM(latent_dim, return_sequences=False, return_state=True)
    if num_layers == 1:
        decoder_outputs, _, _ = decoder_lstm2(decoder_dropout)
    elif num_layers == 2:
        decoder_outputs, _, _ = decoder_lstm2(decoder_lstm)
    else:
        print("Invalid Number of layers")
        sys.exit(-1)

    decoder_dense = Dense(num_labels, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())
    return model


def create_model_only_dec(num_words, embedding_matrix, embedding_dim, num_labels, dropout_rate, latent_dim, num_layers=1):
    # Set up the decoder, using `encoder_states` as initial state.
    # Input layer of the decoder :
    decoder_inputs = Input(shape=(None,))

    # Hidden layers of the decoder :
    decoder_embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                        embeddings_initializer=Constant(embedding_matrix), trainable=False)
    decoder_embedding = decoder_embedding_layer(decoder_inputs)

    decoder_dropout_layer = (TimeDistributed(Dropout(rate=dropout_rate)))
    decoder_dropout = decoder_dropout_layer(decoder_embedding)

    decoder_lstm_layer = CuDNNLSTM(latent_dim, return_sequences=True)
    decoder_lstm = decoder_lstm_layer(decoder_dropout)

    decoder_lstm2 = CuDNNLSTM(latent_dim, return_sequences=False, return_state=True)

    if num_layers == 1:
        decoder_outputs, _, _ = decoder_lstm2(decoder_dropout)
    elif num_layers == 2:
        decoder_outputs, _, _ = decoder_lstm2(decoder_lstm)
    else:
        print("Invalid number of layers")
        sys.exit(-1)

    decoder_dense = Dense(num_labels, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([decoder_inputs], decoder_outputs)
    print(model.summary())
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Seq&Seq Model or load the Model.')
    parser.add_argument('--mode', default='train', help='One of train and load, default train')
    parser.add_argument('--dataset', default='node', help='One of node and political, default node')
    args = parser.parse_args()

    # TensorFlow Config for GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    # Defining some constants:
    embedding_dim = 300  # Length of the vector that we will get from the embedding layer
    max_num_words = 20000
    latent_dim = 1024  # Hidden layers dimension
    dropout_rate = 0.2  # Rate of the dropout layers
    batch_size = 64  # Batch size
    epochs = 50  # Number of epochs

    # Read and prepare the data
    enc_train, dec_train, target_train, labels_train, enc_val, dec_val, target_val, labels_val, word_index = \
        read_and_prepare_data(args.dataset)

    # Read the word vectors
    embeddings_index = read_glove_embeddings()

    # Convert word_index to embedding_matrix
    embedding_matrix, num_words = prepare_embedding_matrix(word_index, embeddings_index, max_num_words, embedding_dim)

    if args.mode == 'train':
        # Create the model
        model = create_model(num_words, embedding_matrix, embedding_dim, 2, dropout_rate, latent_dim)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit([enc_train, dec_train], target_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2,
                  shuffle=True)
        # Save model
        model.save('s2s.h5')

        # Create the model
        model_only_dec = create_model_only_dec(num_words, embedding_matrix, embedding_dim, 2, dropout_rate, latent_dim)

        # Run training
        model_only_dec.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model_only_dec.fit([dec_train], target_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_split=0.2,
                           shuffle=True)
        # Save model
        model_only_dec.save('s2s_only_dec.h5')
    elif args.mode == 'load':
        # load the model that was saved
        model = load_model("s2s.h5")
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        model_only_dec = load_model("s2s_only_dec.h5")
        model_only_dec.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("Invalid mode! Only train and load available")
        sys.exit(-1)

    # Eval on training data
    print(model.evaluate([enc_train, dec_train], target_train, verbose=0))
    y_pred = np.argmax(model.predict([enc_train, dec_train]), axis=1)
    print("Confusion Matrix:")
    print(confusion_matrix(labels_train, y_pred))
    print("Classification Report:")
    print(classification_report(labels_train, y_pred, target_names=["attack", "support"]))

    # Eval on validation data
    print(model.evaluate([enc_val, dec_val], target_val, verbose=0))
    y_pred = np.argmax(model.predict([enc_val, dec_val]), axis=1)
    print("Confusion Matrix:")
    print(confusion_matrix(labels_val, y_pred))
    print("Classification Report:")
    print(classification_report(labels_val, y_pred, target_names=["attack", "support"]))

    # Eval on training data
    print(model_only_dec.evaluate([dec_train], target_train, verbose=0))
    y_pred = np.argmax(model_only_dec.predict([dec_train]), axis=1)
    print("Confusion Matrix:")
    print(confusion_matrix(labels_train, y_pred))
    print("Classification Report:")
    print(classification_report(labels_train, y_pred, target_names=["attack", "support"]))

    # Eval on validation data
    print(model_only_dec.evaluate([dec_val], target_val, verbose=0))
    y_pred = np.argmax(model_only_dec.predict([dec_val]), axis=1)
    print("Confusion Matrix:")
    print(confusion_matrix(labels_val, y_pred))
    print("Classification Report:")
    print(classification_report(labels_val, y_pred, target_names=["attack", "support"]))



