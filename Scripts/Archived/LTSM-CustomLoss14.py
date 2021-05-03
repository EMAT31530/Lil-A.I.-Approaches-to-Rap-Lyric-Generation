# Our library imports
import PipelineV8 as pipeLine

# Allow XLA enhanced training
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Machine Learning and maths imports
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Dropout, Bidirectional, LSTM, Input
from tensorflow.keras.layers import Flatten, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import LeakyReLU, Activation
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp
from gumbel_softmax import GumbelSoftmax, GumbelNoise
from tensorflow_probability.python.distributions import RelaxedOneHotCategorical
import tensorflow_addons as tfa
from tensorflow.keras import regularizers

from tensorflow.keras.layers import GaussianNoise

import numpy as np
from numpy import exp
import random

# Language imports
import re
import pronouncing
from g2p_en import G2p

# System imports
import sys
import os
import time
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

    with open(path, "r", encoding='utf-8-sig') as f:
        _content = f.readlines()

    # Defined content in case we need to split each line/bar into strings
    _content = [_content[i].rstrip().lower() for i in range(len(_content)) if _content[i] != '\n' and not num_there(_content[i])]

    if allow_digits:
        _vocabulary = ''.join([i if i != '\n' else ' ' for i in text]).replace("\n", " ").split(' ')
    else:
        _vocabulary = ''.join([i if not i.isdigit() and i != '\n' else ' ' for i in text]).replace("\n", " ").split(' ')
        _vocabulary = [word.lower() for word in _vocabulary if word != '' and not num_there(word)]

    return _content, _vocabulary


def generate_words(vocabulary_input):
    """
    # List of all unique vocabulary in alphabetical order
    :param vocabulary_input: list
    :return: list
    """
    return sorted(list(set(vocabulary_input)))


def words_to_integers(bars_input, words_input):
    # Need to reverse this at the end to reverse numbers back into words
    word_to_int_local = {words_input[i]: i for i in range(len(words_input))}

    stripped_bar = [word.split() for word in bars_input]
    encode = [[[word_to_int_local[word] for word in stripped_bar[i]]] for i in range(len(bars_input))]

    encode = sum(encode, [])
    return encode


def sentence_to_integer(bar, words_to_int, verbose=False) -> np.array:
    # Need to reverse this at the end to reverse numbers back into words
    stripped_bar = [word.split() for word in [bar]]
    stripped_bar = stripped_bar[0]

    seq = np.array([words_to_int[word] for word in stripped_bar])

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
    _words = generate_words(vocabulary_input)

    # Need to reverse this at the end to reverse numbers back into words
    charsdict = {_words[i]: i for i in range(len(_words))}

    return charsdict


def remove_first_spaces(line: str) -> str:
    if line[0] != ' ':
        return line

    p = [index for index in range(len(line)) if line[index] != ' ']
    return line[p[0]:]


def choose_random_word(bars_input):
    return random.choice(bars_input).split()[0]


def random_number_custom_probs(probabilities, int_to_words_input):
    random_index = np.random.choice(np.arange(0, len(int_to_words_input)) - 1, p=probabilities)
    return random_index


def generate_rap_lyrics(model_input, bars_input, words_int_input, int_words_input, padding_len, num_lines=10):
    # Random seed text from the input

    stop_index = word_to_int['[stop]']

    seed_texts = [choose_random_word(bars_input) for _ in range(num_lines + 1)]
    lines = []
    lyrics = '[start] ' + seed_texts[0]
    token_list_local = sentence_to_integer(lyrics, words_int_input)
    token_list_local = pad_sequences([token_list_local], maxlen=padding_len - 1, padding='pre')

    i = 0

    while i < num_lines:
        predicted = model_input.predict_classes(token_list_local, verbose=0)

        print('Input to predict next word:', token_list_local)
        print('Predicted:', predicted)
        print('-' * 100)

        if predicted[0] == stop_index:
            i += 1
            # End-Of-Sequence character
            seed_text = seed_texts[i]
            lyrics = lyrics + '\n'

            # Let the model add its newlines
            lyrics = lyrics.replace('[start]', '')

            lines.append(remove_first_spaces(lyrics).capitalize())

            lyrics = '[start] ' + seed_text
            print('STOP FOUND. Next Predict:', token_list_local)
        else:
            lyrics += ' ' + int_words_input[predicted[0]]  # 0 because we only want the *next* word

        token_list_local = sentence_to_integer(lyrics, words_int_input)

        token_list_local = pad_sequences([token_list_local], maxlen=padding_len - 1, padding='pre')

    lyrics = ''.join(lines)

    lyrics = pipeLine.clean(lyrics)

    return lyrics


def g2p_lazy(word, g2p):
    input0 = tf.identity(word).numpy().decode("utf-8")
    out = ' '.join(g2p(input0))
    return out


def reverse_syllable_count(true_phones, pred_phones, normalizing_constant):
    # We don't need to copy the arrays, literally doesn't matter that be edit the original array being pointed to
    syll1 = true_phones.numpy().split()
    syll2 = pred_phones.numpy().split()

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
    counter = length + 1 / (counter + 0.05)
    return 1 / (1 + exp(- counter))


def loop_depth(_phone1, _phone2, normalizing_constant):
    rhyme_count = reverse_syllable_count(_phone1, _phone2, normalizing_constant)
    return rhyme_count


def get_focal_params(y_pred):
    epsilon = tf.constant(1e-9)
    gamma = tf.constant(3.)
    y_pred = y_pred + epsilon
    pinv = 1. / y_pred
    pos_weight_f = (pinv - 1) ** gamma
    weight_f = y_pred ** gamma
    return pos_weight_f, weight_f


def add_gumble_noise_from_logits(y_true, y_pred, temperature=0.5):
    y_pred32 = tf.cast(y_pred, tf.float32)
    dist = RelaxedOneHotCategorical(temperature, logits=K.eval(y_pred32))
    redistributed_pred = tf.nn.softmax(y_pred32 + dist.logits)
    return tf.cast(y_true, tf.float32), redistributed_pred


def add_gumble_noise_logit_to_logit(y_pred, temperature=0.5):
    y_pred32 = tf.cast(y_pred, tf.float32)
    dist = RelaxedOneHotCategorical(temperature, logits=K.eval(y_pred32))
    return y_pred32 + dist.logits


class CustomModel:
    def __init__(self, _x_train, _y_train, int_to_words, path="training_2/cp.ckpt", temperature=0.5):
        self.model = Sequential()
        self.model.add(Embedding(len(int_to_words), 512, input_length=padding_length - 1, mask_zero=True))
        # self.model.add(Dropout(0.1))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Dropout(0.2))
        # ------------------------------
        self.model.add(Bidirectional(LSTM(256)))
        self.model.add(Dropout(0.4))
        # ------------------------------
        # self.model.add(Bidirectional(LSTM(128, kernel_regularizer='l2')))
        # self.model.add(Dropout(0.4))
        # ------------------------------
        # ------------------------------
        # self.model.add(Bidirectional(LSTM(512, return_sequences=True, kernel_regularizer='l2')))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.5))
        # ------------------------------
        # self.model.add(Bidirectional(LSTM(512, kernel_regularizer='l2')))
        # # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.5))
        # ------------------------------
        # leaky_relu = LeakyReLU(alpha=0.1)
        # self.model.add(Dense(256, kernel_regularizer='l2'))
        # self.model.add(Dropout(0.5))
        # ------------------------------
        # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedOneHotCategorical
        self.model.add(Dense(len(int_to_words), activation='softmax'))
        self.model.add(GumbelSoftmax())
        # radam = tfa.optimizers.RectifiedAdam(beta_2=0.999, beta_1=0.999, learning_rate=0.1)
        # ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        # loss_recall = self.recall_loss(0.9, 0.1, 2, temperature=100)
        # loss_focall_tf = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=tf.keras.losses.Reduction.AUTO)
        # adam_optimiser = Adam(learning_rate=0.001)
        # sgd_optimiser = SGD()
        categorical_crossentropy_loss = categorical_crossentropy(from_logits=False)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam', run_eagerly=True)
        # self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam_optimiser, run_eagerly=True)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                              save_weights_only=True,
                                                              verbose=1)

        # https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
        # kernel_regularizer=regularizers.l1(l1=1e-5)
        # https://www.kdnuggets.com/2020/08/tensorflow-model-regularization-techniques.html
        # https://github.com/tensorflow/tensorflow/issues/1941
        # https://stackoverflow.com/questions/35316250/tensorflow-dictionary-lookup-with-string-tensor
        # https://stats.stackexchange.com/questions/383310/what-is-the-difference-between-kernel-bias-and-activity-regulizers-and-when-t

        self.path = path
        self.int_to_words = int_to_words

    def summary(self):
        self.model.summary()

    def load(self):
        # Loads the weights
        self.model.load_weights(self.path)
        print('Restored Model')

    def train(self, _x_train, _y_train, _tensorboard=None, epochs=30, batch_size=512, verbose=1):
        self.batch_size = tf.constant(batch_size)

        if _tensorboard is not None:
            self.model.fit(_x_train, _y_train, epochs=epochs, validation_split=0.15,
                           batch_size=batch_size, callbacks=[_tensorboard, self.cp_callback],
                           verbose=verbose)
            return
        else:
            # Train the model
            self.model.fit(_x_train, _y_train, epochs=epochs, validation_split=0.15,
                                     batch_size=batch_size, callbacks=[],
                                     verbose=verbose)

    def test(self, _x_test, _y_train):
        self.loss = self.model.evaluate(x_train, y_train)
        print("Model loss on training: {}".format(self.loss))
        return self.loss

    # Our custom loss' wrapper
    def recall_loss(self, recall_weight, spec_weight, focal_weight, temperature):
        def recall_spec_loss(y_true, y_pred):
            return self.binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight, focal_weight, temperature)

        # Returns the (y_true, y_pred) loss function
        return recall_spec_loss

    def binary_recall_specificity(self, y_true, y_pred, recall_weight, spec_weight, focal_weight, temperature):

        # y_true32 = tf.cast(y_true, tf.float32)
        # y_pred32 = tf.cast(y_pred, tf.float32)

        # # y_true32, y_pred32 = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
        #
        # TN = tf.cast(tf.logical_and(K.eval(y_true32) == 0, K.eval(y_pred32) == 0), tf.float32)
        # TP = tf.cast(tf.logical_and(K.eval(y_true32) == 1, K.eval(y_pred32) == 1), tf.float32)
        #
        # FP = tf.cast(tf.logical_and(K.eval(y_true32) == 0, K.eval(y_pred32) == 1), tf.float32)
        # FN = tf.cast(tf.logical_and(K.eval(y_true32) == 1, K.eval(y_pred32) == 0), tf.float32)
        #
        # # Converted as Keras Tensors
        # TN = K.sum(TN)
        # FP = K.sum(FP)

        # specificity = tf.divide(TN, (TN + FP + K.epsilon()))
        # recall = tf.divide(TP, (TP + FN + K.epsilon()))
        #
        # self.weight = tf.cast(tf.constant(1.0, dtype=tf.float32) - (recall_weight * recall + spec_weight * specificity),
        #                       tf.float32)

        # self.weight * (tf.constant(focal_weight, tf.float32) * self.focal_loss(y_true32, y_pred32) + cce)
        # tf.constant(focal_weight, tf.float32) * focal_losses + cce_losses

        # cce_losses = categorical_crossentropy
        focal_losses = self.focal_loss(y_true, y_pred)
        return focal_losses

    def focal_loss(self, y_true, y_pred):
        y_pred_prob = tf.keras.backend.sigmoid(y_pred)
        pos_weight_f, weight_f = get_focal_params(y_pred_prob)
        alpha = tf.constant(.35)
        alpha_ = 1 - alpha
        alpha_div = alpha / alpha_
        pos_weight = pos_weight_f * alpha_div
        weight = weight_f * alpha_

        l2 = tf.reduce_mean(weight * tf.nn.weighted_cross_entropy_with_logits(labels=y_true,
                                                                              logits=y_pred,
                                                                              pos_weight=pos_weight), 1)
        return l2

    def gcce_loss_from_logits(self, y_true, y_pred, temperature):
        y_pred32 = tf.cast(y_pred, tf.float32)
        dist = RelaxedOneHotCategorical(temperature, logits=K.eval(y_pred32))
        redistributed_pred = y_pred32 + dist.logits
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=redistributed_pred)
        return loss


if __name__ == "__main__":
    # Global params:
    sequence_length = 20
    checkpoint_path = "training_4/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Update our local data (Import data = True to import fresh from GitHub
    content, vocab = generate_content_vocab('AllLyrics_uncleanOLD.txt', import_data=False)

    # Bars is a list containing each line in dataset
    bars_no_start = [x.strip() for x in content]
    bars = ['[start] ' + x.strip() + ' [stop]' for x in content]
    stripped_bars = [word.split() for word in bars]
    stripped_bars = [subitem for item in stripped_bars for subitem in item]

    no_of_bars = len(bars)

    words = sorted(list(set(stripped_bars)))
    vocab_size = len(words)

    int_to_word = {i: words[i] for i in range(len(words))}
    word_to_int = {words[i]: i for i in range(len(words))}

    print('Words dictionary: ', word_to_int)
    print('Number of unique words: ', len(words))

    word_dict = frequency_count(vocab)

    # Generate sentences
    sequences = [sentence_to_integer(line, word_to_int)[:i + 1] for line in bars
                 for i in range(1, len(sentence_to_integer(line, word_to_int)))]

    padding_length = max(len(line) for line in sequences)

    # Pad the data
    sequences = np.array(pad_sequences(sequences, maxlen=padding_length, padding='pre'))

    # Remove last word from each line
    x_train = sequences[:, :-1]

    # Last word is used as the label
    y_train = sequences[:, -1]
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=vocab_size)

    # Split into testing and training randomly
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=1, shuffle=True)

    # Check if the model exists:
    # Need to compile model now
    time_stamp = time.time()
    tensorboard = TensorBoard(log_dir="logs/{}".format(time_stamp))

    lstm = CustomModel(x_train, y_train, int_to_word, path="training_61/cp.ckpt".format(time_stamp))
    lstm.summary()
    # lstm.train(x_train, y_train, epochs=200, _tensorboard=tensorboard, batch_size=256, verbose=1)
    lstm.load()
    #
    # lstm.test(x_test, y_test)

    generated_lyrics = generate_rap_lyrics(lstm.model, bars_no_start, word_to_int, int_to_word, padding_length)
    print('Generated Lyrics:')
    print(generated_lyrics)


