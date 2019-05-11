'''
#Sequence to sequence example in Keras (character-level).
This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.
**Summary of the algorithm**
- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
**Data download**
[English to French sentence pairs.
](http://www.manythings.org/anki/fra-eng.zip)
[Lots of neat sentence pairs datasets.
](http://www.manythings.org/anki/)
**References**
- [Sequence to Sequence Learning with Neural Networks
   ](https://arxiv.org/abs/1409.3215)
- [Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    ](https://arxiv.org/abs/1406.1078)
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Dense, CuDNNLSTM, Input, Embedding, TimeDistributed, Dropout
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


# Path to the data txt file on disk.
data_path = 'complete_data.tsv'

# Vectorize the data.

df = pd.read_csv(data_path, sep="\t")
data = df.loc[~df['org_dataset'].isin(['political', 'procon'])].loc[df['label'].isin(['attack', 'support'])]

# Convert to lowercase and replace commas and get rid of punctuation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(np.concatenate((data['org'].values, data['response'].values), axis=-1))
encoder_input_data = pad_sequences(tokenizer.texts_to_sequences(data['org'].values))
decoder_input_data = pad_sequences(tokenizer.texts_to_sequences(data['response'].values))

word_index = tokenizer.word_index

convert_dict = {"attack": 0, "support": 1, "unrelated": 2}
decoder_target = np.array([convert_dict[label] for label in data['label']])
decoder_target_data = to_categorical(decoder_target, len(set(data['label'])))

print('Number of samples:', len(encoder_input_data))
print('Number of unique tokens:', len(word_index)+1)
#  print('Max sequence length for inputs:', max_en_words_per_sample)
#  print('Max sequence length for outputs:', max_de_words_per_sample)

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
os.chdir('C:')
print(os.getcwd())
with open('Users/Jannis/Documents/glove.42B.300d/glove.42B.300d.txt', encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))
os.chdir('F:/SS19/BA/Twitter/twitter-test')

# Defining some constants:
EMBEDDING_DIM = 300   # Length of the vector that we will get from the embedding layer
MAX_NUM_WORDS = 20000
latent_dim = 1024  # Hidden layers dimension
dropout_rate = 0.2   # Rate of the dropout layers
batch_size = 64    # Batch size
epochs = 100  # Number of epochs
print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        print("Word not found" + word)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                              embeddings_initializer=Constant(embedding_matrix), trainable=False)(encoder_inputs)
encoder_dropout = (TimeDistributed(Dropout(rate=dropout_rate)))(encoder_embedding)
encoder = CuDNNLSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_dropout)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
# Input layer of the decoder :
decoder_inputs = Input(shape=(None,))

# Hidden layers of the decoder :
decoder_embedding_layer = Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                              embeddings_initializer=Constant(embedding_matrix), trainable=False)
decoder_embedding = decoder_embedding_layer(decoder_inputs)

decoder_dropout_layer = (TimeDistributed(Dropout(rate=dropout_rate)))
decoder_dropout = decoder_dropout_layer(decoder_embedding)

decoder_LSTM_layer = CuDNNLSTM(latent_dim, return_sequences=False, return_state=True)
decoder_outputs, _, _ = decoder_LSTM_layer(decoder_dropout, initial_state=encoder_states)

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.

decoder_dense = Dense(len(set(data['label'])), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          shuffle=True)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Eval
encoder_input_eval = pad_sequences(tokenizer.texts_to_sequences(df['org'].loc[df['org_dataset'] == 'procon'].values))
decoder_input_eval = pad_sequences(tokenizer.texts_to_sequences(df['response'].loc[df['org_dataset'] == 'procon'].values))
decoder_target_eval = np.array([convert_dict[label] for label in df['label'].loc[df['org_dataset'] == 'procon']])
decoder_target_eval_data = to_categorical(decoder_target_eval, len(set(data['label'])))
print(model.evaluate([encoder_input_eval, decoder_input_eval], decoder_target_eval_data, verbose=0))

y_pred = np.argmax(model.predict([encoder_input_eval, decoder_input_eval]), axis=1)
print("Confusion Matrix:")
print(confusion_matrix(decoder_target_eval, y_pred))
print("Classification Report:")
print(classification_report(decoder_target_eval, y_pred, target_names=["attack", "support"]))

encoder_input_eval = pad_sequences(tokenizer.texts_to_sequences(data['org'].values))
decoder_input_eval = pad_sequences(tokenizer.texts_to_sequences(data['response'].values))
decoder_target_eval = np.array([convert_dict[label] for label in data['label']])
decoder_target_eval_data = to_categorical(decoder_target_eval, len(set(data['label'])))
print(model.evaluate([encoder_input_eval, decoder_input_eval], decoder_target_eval_data, verbose=0))
y_pred = np.argmax(model.predict([encoder_input_eval, decoder_input_eval]), axis=1)
print("Confusion Matrix:")
print(confusion_matrix(decoder_target_eval, y_pred))
print("Classification Report:")
print(classification_report(decoder_target_eval, y_pred, target_names=["attack", "support"]))