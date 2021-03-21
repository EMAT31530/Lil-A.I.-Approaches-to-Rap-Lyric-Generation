# Our library imports
import PipelineV8 as pipeLine

# Allow XLA enhanced training
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Machine Learning and maths imports
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import tensorflow as tf

import numpy as np
import random

# Language imports
import re
import pronouncing
from g2p_en import G2p

# System imports
import pickle
import sys
from time import time
from pathlib import Path

# Plotting imports
import matplotlib.pyplot as plt

# Why is this necessary :(
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Function
def num_there(s):
    return any(i.isdigit() for i in s)


def frequency_count(lyrics):
    """ O(n) frequency count
    :param lyrics: list of strings
    :return dictionary
    """
    a = {}
    for word in lyrics:
        if word in a:
            a[word] += 1
        else:
            a[word] = 1
    return a


def generate_content_vocab(path="AllLyrics_unclean.txt", import_data=True, allow_digits=False):
    """
    Generate content and vocab
    :param path: str
    :param import_data: bool
    :return: list, list
    """
    if path[-4:] != '.txt':
        print('Path {} is not a text file'.format(path))
        raise FileNotFoundError

    # Update our local data
    if import_data:
        pipeLine.import_github(write_type='both')

    # Open the lyrics
    with open(path, "r", encoding='utf-8-sig') as f:
        text = f.read()

    # Text to lower case to reduce vocabulary
    text = text.lower()

    with open(path, "r", encoding='utf-8-sig') as f:
        _content = f.readlines()

    # Defined content in case we need to split each line/bar into strings
    _content = [_content[i].rstrip() for i in range(len(_content)) if _content[i] != '\n' and not num_there(_content[i])]

    if allow_digits:
        _vocabulary = ''.join([i if i != '\n' else ' ' for i in text]).replace("\n", " ").split(' ')
    else:
        _vocabulary = ''.join([i if not i.isdigit() and i != '\n' else ' ' for i in text]).replace("\n", " ").split(' ')
        _vocabulary = [word for word in _vocabulary if word != '' and not num_there(word)]

    return _content, _vocabulary


def generate_words(vocabulary_input):
    """
    # List of all unique vocabulary in alphabetical order
    :param vocabulary_input: list
    :return: list
    """
    return sorted(list(set(vocabulary_input)))


def words_to_integers(bars, words):
    # Need to reverse this at the end to reverse numbers back into words
    word_to_int = {words[i]: i for i in range(len(words))}

    stripped_bar = [word.split() for word in bars]
    encode = [[[word_to_int[word] for word in stripped_bar[i]]] for i in range(len(bars))]

    encode = sum(encode, [])
    return encode


def sentence_to_integer(bar, words_to_int, verbose=False):
    # Need to reverse this at the end to reverse numbers back into words
    stripped_bar = [word.split() for word in [bar]]
    stripped_bar = stripped_bar[0]

    seq = [[words_to_int[word] for word in stripped_bar]]
    seq = sum(seq, [])

    if verbose:
        print('Bar: ', bar)
        print('Stripped bar 1: ', stripped_bar)
        print('Words to int dict: ', words_to_int)
        print('Sequence: ', seq)
        print('-----------------------------------')

    return seq


def generate_chars_dicts(vocabulary_input):
    """
    Generate the chars and int dictionaries
    :param vocabulary_input: list
    :return: dict
    """
    words = generate_words(vocabulary_input)

    # Need to reverse this at the end to reverse numbers back into words
    charsdict = {words[i]: i for i in range(len(words))}

    return charsdict


def generate_sentences_next_words(vocabulary_input, chars_dict_input, seq_len=20):
    """
    nextword will contain all the associated 'next words' in our sentences
    Ex. for the sentence 'I am doing a NLP project' with seq_len = 5, sentences = [I am doing a NLP] and nextword = [project]
    :param vocabulary_input: list
    :param chars_dict_input: dict
    :param seq_len: int
    :return: list, list
    """

    nextword = []
    sentences = []

    # We need training and test examples so we split our text into chunks of 'seq_len' with the next word in the list being our 'test data'
    for i in range(0, len(vocabulary_input) - seq_len):
        train_set = vocabulary_input[i:i + seq_len]
        # Uncomment next line to check it is working
        # print(train_set)
        # Output sequence (will be used as target)
        test_set = vocabulary_input[i + seq_len]
        # print(test_set)
        # We append and convert words to digits
        sentences.append([chars_dict_input[word] for word in train_set])
        # Store targets in data_y'
        nextword.append(chars_dict_input[test_set])

    return sentences, nextword


def generate_rap_lyrics(model_input, bars_input, padding_len, words_int_input, int_words_input):
    # Random seed text from the input
    next_words = 100

    seed_text = random.choice(bars_input)
    lyrics = seed_text

    for _ in range(next_words):
        token_list_local = sentence_to_integer(lyrics, words_int_input)
        token_list_local = pad_sequences([token_list_local], maxlen=padding_len - 1, padding='pre')
        predicted = model_input.predict_classes(token_list_local, verbose=0)
        lyrics += ' ' + int_words_input[predicted[0]]

    lyrics = pipeLine.clean(lyrics.capitalize())

    # Add newlines every 10 tokens
    lyrics = lyrics.split(' ')
    lyrics = [lyrics[index] + '\n' if index % 10 == 0 and index != 0 else lyrics[index] for index in range(len(lyrics))]
    lyrics = ' '.join(lyrics)

    return lyrics


def g2p_lazy(word, g2p):
    input0 = tf.identity(word).numpy().decode("utf-8")
    out = ' '.join(g2p(input0))
    return out


def reverse_syllable_count(true_phones, pred_phones):
    # We don't need to copy the arrays, literally doesn't matter that be edit the original array being pointed to,
    # Since this only run once
    syll1 = true_phones.numpy().split()
    syll2 = pred_phones.numpy().split()

    # print('Syll1: ', syll1)
    # print('syll2: ', syll2)

    # Reverse the syllables tensor
    syll1.reverse()
    syll2.reverse()

    # print('Syll1 (r): ', syll1)
    # print('syll2 (r): ', syll2)

    length = min(len(syll1), len(syll2))
    counter = 0
    for i in range(length):
        if syll1[i] == syll2[i]:
            counter += 1
            i += 1
        else:
            break

    # Normalise the rhyming amount
    counter /= length + 1

    return 1 / (counter + 0.1)


def loop_depth(_phone1, _phone2):
    rhyme_count = reverse_syllable_count(_phone1, _phone2)
    return rhyme_count


def custom_loss(y_true, y_pred, phoneme_lookup):
    # To ensure better near rhymes. Use the pronouncing API package
    # Convert to phonemes and then compare how many same phonemes, similar to previous rhyme func
    # This is the most grim hack, I really don't want to hard code each training iteration

    # # THIS IS A SEMI-BAD and HACK-ISH way
    # true_phones = tf.map_fn(fn=lambda word: g2p_lazy(word, g2p), elems=y_true)
    # pred_phones = tf.map_fn(fn=lambda word: g2p_lazy(word, g2p), elems=y_pred)
    true_phones = phoneme_lookup.lookup(y_true)
    pred_phones = phoneme_lookup.lookup(y_pred)

    # print('True Phones: ', true_phones)
    # print('-' * 50)
    # print('Pred Phones: ', pred_phones)
    # print('-' * 50)
    # want rhyme at end so reverse list of syllables, we count number of matching syllables with line and potential
    # line and stop when we meet a non match

    # https://stackoverflow.com/questions/42892347/can-i-apply-tf-map-fn-to-multiple-inputs-outputs
    loss_c = tf.map_fn(fn=lambda phone1: loop_depth(phone1[0], phone1[1]), elems=(true_phones, pred_phones),
                       fn_output_signature=tf.float32)

    return loss_c


class CustomModel:
    def __init__(self, _x_train, _y_train, int_to_words, verbose=True, path="training_2/cp.ckpt"):
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, 256, input_length=padding_length - 1))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(256)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256))
        self.model.add(Dense(vocab_size, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam', run_eagerly=True)
        # self.model.compile(loss=self.custom_loss_inner, metrics=['accuracy'], optimizer='adam', run_eagerly=True)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                              save_weights_only=True,
                                                              verbose=1)
        # https://github.com/tensorflow/tensorflow/issues/1941
        # https://stackoverflow.com/questions/35316250/tensorflow-dictionary-lookup-with-string-tensor

        if verbose:
            self.model.summary()  # Print model summary to console

        self.path = path
        self.int_to_words = int_to_words
        # Build a lookup hash table for which words correspond to which tokens
        self.hash_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(list(int_to_word.keys())),
                values=tf.constant(list(int_to_word.values())),
            ),
            default_value=tf.constant('#'),
            name="int_word_hash"
        )

        # Build a lookup hash table for which phonemes correspond to which words
        g2p = G2p()
        common_phonemes_lookup = [g2p_lazy(word, g2p) for word in int_to_words.values()]

        self.phoneme_hash_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(int_to_words.values()),
                values=tf.constant(common_phonemes_lookup),
            ),
            default_value=tf.constant(''),
            name="phone_word_hash"
        )

        # For clever manipulations, we will save the previous predictions
        self.previous_predictions = 0
        self.batch_size = 32

    def load(self):
        # Loads the weights
        try:
            self.model.load_weights(self.path)
            print('Restored Model')
        except FileNotFoundError:
            print('Directory {} does not exist'.format(self.path))

    def train(self, _x_train, _y_train, _tensorboard=None, epochs=30, batch_size=512, verbose=1):
        self.batch_size = tf.constant(batch_size)
        checkpoint_Path = Path(self.path + '.index')
        if _tensorboard is not None:
            self.model.fit(_x_train, _y_train, epochs=epochs, validation_split=0.15,
                           batch_size=batch_size, callbacks=[_tensorboard],
                           verbose=verbose)
            return

        if not checkpoint_Path.is_file():
            # Train the model
            self.model.fit(_x_train, _y_train, epochs=epochs, validation_split=0.15,
                                     batch_size=batch_size, callbacks=[self.cp_callback],
                                     verbose=verbose)

    def test(self, _x_test, _y_train):
        self.loss = self.model.evaluate(x_train, y_train)
        print("Model loss on training: {}".format(self.loss))
        return self.loss

    def custom_loss_inner(self, y_true, y_pred):
        # This allows biasing rhyming patterns between batches
        loss_categorical = categorical_crossentropy(y_true, y_pred)

        if type(self.previous_predictions) == int:
            # Save the current predictions
            self.previous_predictions = tf.argmax(tf.identity(y_pred), axis=1, output_type=tf.int32)

            return loss_categorical

        # Only perform the rhyme check if the batch size matches the global batch size
        if self.batch_size != tf.shape(y_pred)[0]:
            return loss_categorical

        # Load the previous predictions
        y_pred2 = self.previous_predictions

        # Convert to prediction integers
        y_pred1 = tf.argmax(tf.identity(y_pred), axis=1, output_type=tf.int32)

        # Save the current predictions IF it matches batch size.
        self.previous_predictions = tf.identity(y_pred1)

        # Hash table lookup the corresponding words
        y_pred1 = self.hash_table.lookup(y_pred1)
        y_pred2 = self.hash_table.lookup(y_pred2)

        loss_custom = custom_loss(y_pred1, y_pred2, self.phoneme_hash_table)

        loss_weighted = tf.add(loss_categorical, tf.scalar_mul(0.1, loss_custom))

        return loss_weighted

    def plot_loss(self):
        loss_train = self.model.history['loss']
        loss_val = self.model.history['val_loss']
        print(loss_train)
        sys.exit()


if __name__ == "__main__":
    # Global params:
    sequence_length = 20
    checkpoint_path = "training_4/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Update our local data (Import data = True to import fresh from GitHub
    content, vocab = generate_content_vocab('AllLyrics_unclean.txt', import_data=False)

    # Bars is a list containing each line in dataset in lowercase
    bars = [x.strip().lower() for x in content]
    stripped_bars = [word.split() for word in bars]
    no_of_bars = len(bars)

    words = sorted(list(set(vocab)))
    int_to_word = {i: words[i] for i in range(len(words))}
    # Need to reverse this at the end to reverse numbers back into words
    word_to_int = {words[i]: i for i in range(len(words))}
    print('Word to int local: ', word_to_int)
    print('Number of unique words: ', len(words))

    word_dict = frequency_count(vocab)
    sort_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)

    vocab_size = len(words) + 1

    # Generate sentences
    sequences = []
    for line in bars:
        token_list = sentence_to_integer(line, word_to_int)
        for i in range(1, len(token_list)):
            n_gram_seq = token_list[:i + 1]
            sequences.append(n_gram_seq)

    padding_length = max([len(line) for line in sequences])

    # Pad the data
    sequences = np.array(pad_sequences(sequences, maxlen=padding_length, padding='pre'))

    # Remove last word from each line
    x_train = sequences[:, :-1]
    # Last word is used as the label
    y_train = sequences[:, -1]

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=vocab_size)

    # Split into testing and training randomly
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

    # Check if the model exists:
    # Need to compile model now

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    lstm = CustomModel(x_train, y_train, int_to_word, path="training_9/cp.ckpt", verbose=False)
    # lstm.train(x_train, y_train, epochs=60, batch_size=512, verbose=1)
    lstm.load()

    generated_lyrics = generate_rap_lyrics(lstm.model, bars, padding_length, word_to_int, int_to_word)
    print('Generated Lyrics: ', generated_lyrics)


