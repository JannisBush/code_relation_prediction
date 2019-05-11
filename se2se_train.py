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
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string
from keras.backend.tensorflow_backend import set_session
from keras.utils.np_utils import to_categorical
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
data = df.loc[~df['org_dataset'].isin(['political'])].loc[df['label'].isin(['attack', 'support'])]

# Convert to lowercase and replace commas and get rid of punctuation
exclude = set(string.punctuation)
lines_org = data['org'].apply(lambda x: x.lower()).apply(lambda x: re.sub("'", '', x))\
    .apply(lambda x: re.sub(",", ' COMMA', x)).apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines_resp = data['response'].apply(lambda x: x.lower()).apply(lambda x: re.sub("'", '', x))\
    .apply(lambda x: re.sub(",", ' COMMA', x)).apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

# Create word dictionaries :
en_words = set()
for line in lines_org:
    for word in line.split():
        if word not in en_words:
            en_words.add(word)

de_words = set()
for line in lines_resp:
    for word in line.split():
        if word not in de_words:
            de_words.add(word)

# get lengths and sizes :
num_en_words = len(en_words)
num_de_words = len(de_words)

max_en_words_per_sample = max([len(sample.split()) for sample in lines_org]) + 5
max_de_words_per_sample = max([len(sample.split()) for sample in lines_resp]) + 5

num_en_samples = len(lines_org)
num_de_samples = len(lines_resp)

# Get lists of words :
input_words = sorted(list(en_words))
target_words = sorted(list(de_words))

en_token_to_int = dict()
en_int_to_token = dict()

de_token_to_int = dict()
de_int_to_token = dict()

# Tokenizing the words ( Convert them to numbers ) :
for i, token in enumerate(input_words):
    en_token_to_int[token] = i
    en_int_to_token[i] = token

for i, token in enumerate(target_words):
    de_token_to_int[token] = i
    de_int_to_token[i] = token

# initiate numpy arrays to hold the data that our seq2seq model will use:
encoder_input_data = np.zeros(
    (num_en_samples, max_en_words_per_sample),
    dtype='float32')
decoder_input_data = np.zeros(
    (num_de_samples, max_de_words_per_sample),
    dtype='float32')

convert_dict = {"attack": 0, "support": 1, "unrelated": 2}
decoder_target = np.array([convert_dict[label] for label in data['label']])
decoder_target_data = to_categorical(decoder_target, len(set(data['label'])))

print('Number of samples:', num_en_samples)
print('Number of unique input tokens:', num_en_words)
print('Number of unique output tokens:', num_de_words)
print('Max sequence length for inputs:', max_en_words_per_sample)
print('Max sequence length for outputs:', max_de_words_per_sample)


# Process samples, to get input, output, target data:
for i, (input_text, target_text) in enumerate(zip(lines_org, lines_resp)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = en_token_to_int[word]
    for t, word in enumerate(target_text.split()):
        decoder_input_data[i, t] = de_token_to_int[word]


# Defining some constants:
vec_len = 300   # Length of the vector that we will get from the embedding layer
latent_dim = 1024  # Hidden layers dimension
dropout_rate = 0.2   # Rate of the dropout layers
batch_size = 64    # Batch size
epochs = 30  # Number of epochs

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=num_en_words, output_dim=vec_len)(encoder_inputs)
encoder_dropout = (TimeDistributed(Dropout(rate=dropout_rate)))(encoder_embedding)
encoder = CuDNNLSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_dropout)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
# Input layer of the decoder :
decoder_inputs = Input(shape=(None,))

# Hidden layers of the decoder :
decoder_embedding_layer = Embedding(input_dim=num_de_words, output_dim=vec_len)
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

print(model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data, verbose=0))

result = model.predict([encoder_input_data, decoder_input_data])
print(result)
print(decoder_target.shape)
print(np.argmax(result, axis=1).shape)
print(np.count_nonzero(decoder_target == np.argmax(result, axis=1))/num_en_samples)





# # Define sampling models
# encoder_model = Model(encoder_inputs, encoder_states)
#
# decoder_state_input_h = Input(shape=(latent_dim,))
# decoder_state_input_c = Input(shape=(latent_dim,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_outputs, state_h, state_c = decoder_lstm(
#     decoder_inputs, initial_state=decoder_states_inputs)
# decoder_states = [state_h, state_c]
# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)
#
# # Reverse-lookup token index to decode sequences back to
# # something readable.
# reverse_input_char_index = dict(
#     (i, char) for char, i in input_token_index.items())
# reverse_target_char_index = dict(
#     (i, char) for char, i in target_token_index.items())
#
#
# def decode_sequence(input_seq):
#     # Encode the input as state vectors.
#     states_value = encoder_model.predict(input_seq)
#
#     # Generate empty target sequence of length 1.
#     target_seq = np.zeros((1, 1, num_decoder_tokens))
#     # Populate the first character of target sequence with the start character.
#     target_seq[0, 0, target_token_index['\t']] = 1.
#
#     # Sampling loop for a batch of sequences
#     # (to simplify, here we assume a batch of size 1).
#     stop_condition = False
#     decoded_sentence = ''
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict(
#             [target_seq] + states_value)
#
#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_char = reverse_target_char_index[sampled_token_index]
#         decoded_sentence += sampled_char
#
#         # Exit condition: either hit max length
#         # or find stop character.
#         if (sampled_char == '\n' or
#            len(decoded_sentence) > max_decoder_seq_length):
#             stop_condition = True
#
#         # Update the target sequence (of length 1).
#         target_seq = np.zeros((1, 1, num_decoder_tokens))
#         target_seq[0, 0, sampled_token_index] = 1.
#
#         # Update states
#         states_value = [h, c]
#
#     return decoded_sentence
#
#
