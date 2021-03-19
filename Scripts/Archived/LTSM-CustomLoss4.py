# Our library imports
import PipelineV8 as pipeLine

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
import tensorflow as tf

import numpy as np
import random

# Language imports
import re
import pronouncing
from g2p_en import G2p

# System imports
import os
import sys
from time import time
from pathlib import Path

# Plotting imports
import matplotlib.pyplot as plt

# Why is this necessary :(
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Remove pesky information lines from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Function
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
    _content = [_content[i].rstrip() for i in range(len(_content)) if _content[i] != '\n']

    if allow_digits:
        _vocabulary = ''.join([i if i != '\n' else ' ' for i in text]).replace("\n", " ").split(' ')
    else:
        _vocabulary = ''.join([i if not i.isdigit() and i != '\n' else ' ' for i in text]).replace("\n", " ").split(' ')
        _vocabulary = [word for word in _vocabulary if word != '']

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
        token_list_local = pad_sequences([token_list_local], maxlen=padding_length - 1, padding='pre')
        predicted = model_input.predict_classes(token_list_local, verbose=0)
        lyrics += ' ' + int_words_input[predicted[0]]

    return lyrics.capitalize()


def rev_syllable_count_end(phonemes1, phonemes2):
    # We don't need to copy the arrays, literally doesn't matter that be edit the original array being pointed to,
    # Since this only run once
    syll1 = phonemes1
    syll2 = phonemes2

    # Reverse the syllables
    syll1.reverse()
    syll2.reverse()

    length = min(len(syll1), len(syll2))

    counter = 0
    for i in range(length):
        if syll1[i] == syll2[i]:
            counter += 1
            i += 1
        else:
            break

    # Normalise the rhyming amount
    counter /= length
    return counter


def generate_phoneme(word, g2p):
    print('here with word:', word)
    print('type: ', type(word))
    word_local = pronouncing.phones_for_word(r'{}'.format(word))
    print(word)
    sys.exit(0)
    if not word_local:
        word_local = [' '.join(g2p(word))]
    else:
        word_local = [word_local[0]]
    return word_local[0].split()


def custom_loss(y_true, y_pred, int_to_words, depth=1, verbose=False):
    # To ensure better near rhymes. Use the pronouncing API package
    # Convert to phonemes and then compare how many same phonemes, similar to previous rhyme func
    # This is the most grim hack, I really don't want to hard code each training iteration

    g2p = G2p()
    # Make a copy, to not consume the OG tensor
    y_true_local = tf.identity(tf.argmax(y_true))
    y_pred_local = tf.identity(tf.argmax(y_pred))

    # Bodge this shit HARD by eager tensor -> numpy -> int, consuming the copy
    y_true_local = int(y_true_local.numpy())
    y_pred_local = int(y_pred_local.numpy())

    print('Index True: ', y_true_local)
    print('Index Pred: ', y_pred_local)

    y_true_local = int_to_words[y_true_local]
    y_pred_local = int_to_words[y_pred_local]

    print('True word: ', y_true_local)
    print('Predicted word: ', y_pred_local)

    true_phones = y_true_local.split(' ')
    pred_phones = y_pred_local.split(' ')

    if verbose:
        print('Ground truth and predictions are lengths: {} vs {}'.format(len(y_true_local), len(y_pred_local)))
        print('Input Truth: ', true_phones)
        print('Input Pred: ', pred_phones)

    true_phones = [generate_phoneme(word, g2p) for word in true_phones]
    pred_phones = [generate_phoneme(word, g2p) for word in pred_phones]

    print('True Phones: ', true_phones)
    print('Pred Phones: ', pred_phones)

    # want rhyme at end so reverse list of syllables, we count number of matching syllables with line and potential
    # line and stop when we meet a non match
    maximum_length = max(len(true_phones), len(pred_phones))

    # We can't allow the search depth to exceed the length of the array
    search_depth = depth + 1
    if depth > maximum_length:
        search_depth = maximum_length + 1

    rhyme_counts = [1 / (rev_syllable_count_end(true_phones[-count], pred_phones[-count]) + 0.05)
                    for count in range(1, search_depth)]

    if verbose:
        print('True phonemes: ', true_phones)
        print('Pred phonemes: ', pred_phones)
        print('Rhyme counts (last to first): ', rhyme_counts)

    sys.exit(0)

    return tf.float16(rhyme_counts)


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
        # self.model.compile(loss=custom_loss, metrics=['accuracy'], optimizer='adam', run_eagerly=True)
        # self.model.compile(loss=self.custom_loss_outer, metrics=['accuracy'], optimizer='adam', run_eagerly=True)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                              save_weights_only=True,
                                                              verbose=1)

        if verbose:
            self.model.summary()  # Print model summary to console

        self.history = []
        self.path = path
        self.int_to_words = int_to_words

    def load(self):
        # Loads the weights
        self.model.load_weights(self.path)
        print('Restored Model')

    def train(self, _x_train, _y_train, _tensorboard=None, epochs=30, batch_size=512, verbose=1):
        checkpoint_Path = Path(self.path + '.index')
        if _tensorboard is not None:
            self.model.fit(_x_train, _y_train, epochs=epochs, batch_size=batch_size, callbacks=[_tensorboard],
                           verbose=verbose)
            return

        if not checkpoint_Path.is_file():
            # Train the model
            history = self.model.fit(_x_train, _y_train, epochs=20, batch_size=512, callbacks=[self.cp_callback],
                                     verbose=verbose)
            self.history.append(history)

    def test(self, _x_test, _y_train):
        self.loss = self.model.evaluate(x_train, y_train)
        print("Model loss on training: {}".format(self.loss))
        return self.loss

    def custom_loss_outer(self, y_true, y_pred):

        y_true_pred = tf.keras.activations.softmax(y_true)
        y_pred_pred = tf.keras.activations.softmax(y_pred)

        print('Input True: ', y_true_pred)
        print('Input Pred: ', y_pred_pred)

        my_loss = tf.scan(lambda a, x: custom_loss(x[0], x[1], self.int_to_words, verbose=True), (y_true,
                                                                                    y_pred))

        return categorical_crossentropy(y_true, y_pred)


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

    # Check if the model exists:
    # Need to compile model now

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    lstm = CustomModel(x_train, y_train, int_to_word, path="training_4/cp.ckpt", verbose=False)
    lstm.train(x_train, y_train, epochs=20, verbose=1)

    # generated_lyrics = generate_rap_lyrics(lstm.model, bars, padding_length, word_to_int, int_to_word)
    # print('Generated Lyrics: ', generated_lyrics)


