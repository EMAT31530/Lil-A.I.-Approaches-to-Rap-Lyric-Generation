# Our library imports
import PipelineV8 as pipeLine

# Allow XLA enhanced training
# import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Machine Learning and maths imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Dropout, Bidirectional, LSTM, Input
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow_probability.python.distributions import RelaxedOneHotCategorical
from collections import Counter

import numpy as np
from numpy import exp
import random
import matplotlib.pyplot as plt

# Language imports
# import re
# import pronouncing
# from g2p_en import G2p

# System imports
import sys
import os
import time
# from pathlib import Path

# Plotting imports
# import matplotlib.pyplot as plt

# Why is this necessary :(
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.python.summary.summary_iterator import summary_iterator


# Functions
def remove_elements(input_list, k):
    """
    Josh's Function
    """
    counted = Counter(input_list)
    kept_words = [word for word in input_list if counted[word] >= k]
    removed_words = [word for word in input_list if counted[word] < k]
    return kept_words, removed_words


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


# def add_gumble_noise_from_logits(y_true, y_pred, temperature=0.5):
#     y_pred32 = tf.cast(y_pred, tf.float32)
#     dist = RelaxedOneHotCategorical(temperature, logits=K.eval(y_pred32))
#     redistributed_pred = tf.nn.softmax(y_pred32 + dist.logits)
#     return tf.cast(y_true, tf.float32), redistributed_pred
#
#
# def add_gumble_noise_logit_to_logit(y_pred, temperature=0.5):
#     y_pred32 = tf.cast(y_pred, tf.float32)
#     dist = RelaxedOneHotCategorical(temperature, logits=K.eval(y_pred32))
#     return y_pred32 + dist.logits


class CustomModel:
    def __init__(self, _x_train, _y_train, int_to_words, path="training_2/cp.ckpt", temperature=0.5):
        self.path = path

        self.model = Sequential()
        self.model.add(Embedding(len(int_to_words), 25, input_length=padding_length - 1, mask_zero=True))
        self.model.add(LayerNormalization())
        # ------------------------------
        # self.model.add(Dropout(0.1))
        # ------------------------------
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Dropout(0.3))
        # ------------------------------
        self.model.add(Bidirectional(LSTM(256)))
        self.model.add(Dropout(0.5))
        # ------------------------------
        # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedOneHotCategorical
        self.model.add(Dense(len(int_to_words), activation='softmax'))
        # self.model.add(GumbelSoftmax())
        # radam = tfa.optimizers.RectifiedAdam(beta_2=0.999, beta_1=0.999, learning_rate=0.1)
        # ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        # loss_recall = self.recall_loss(0.9, 0.1, 2, temperature=100)
        # loss_focall_tf = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, reduction=tf.keras.losses.Reduction.AUTO)
        # adam_optimiser = Adam(learning_rate=0.0001)
        # sgd_optimiser = SGD()
        # categorical_crossentropy_loss = categorical_crossentropy(from_logits=True)
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam', run_eagerly=True)
        # self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam_optimiser, run_eagerly=True)
        self.cp_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.cp_callback_save = tf.keras.callbacks.ModelCheckpoint(filepath=self.path,
                                                                   save_weights_only=True,
                                                                   verbose=1)

        # https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
        # kernel_regularizer=regularizers.l1(l1=1e-5)
        # https://www.kdnuggets.com/2020/08/tensorflow-model-regularization-techniques.html
        # https://github.com/tensorflow/tensorflow/issues/1941
        # https://stackoverflow.com/questions/35316250/tensorflow-dictionary-lookup-with-string-tensor
        # https://stats.stackexchange.com/questions/383310/what-is-the-difference-between-kernel-bias-and-activity-regulizers-and-when-t

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
                           batch_size=batch_size, callbacks=[_tensorboard, self.cp_callback, self.cp_callback_save],
                           verbose=verbose, shuffle=True)
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


def collect_accuracy_loss(events_path):
    epoch_loss = []
    epoch_accuracy = []

    for event in summary_iterator(events_path):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                if value.tag == 'epoch_loss':
                    epoch_loss.append(value.simple_value)
                elif value.tag == 'epoch_accuracy':
                    epoch_accuracy.append(value.simple_value)

    return epoch_accuracy, epoch_loss


def plot_loss(train_loss, valid_loss, name, color1='red', color2='green', linestyle='-') -> None:
    epoch_numbers = list(range(len(train_loss)))

    plt.plot(epoch_numbers, valid_loss, c=color1, label='{} valid loss: '.format(name) +
                                                         str(np.round(valid_loss[-1], 3)),
             linestyle=linestyle)
    plt.plot(epoch_numbers, train_loss, c=color2, label='{} train loss'.format(name),
             linestyle=linestyle)

    plt.scatter(epoch_numbers[-1], valid_loss[-1], c='black')
    return None


def plot_accuracy(train_acc, valid_acc, name, color1='red', color2='green', linestyle='-') -> None:
    epoch_numbers = list(range(len(train_acc)))

    plt.plot(epoch_numbers, valid_acc, c=color1, label='{} valid acc: '.format(name) +
                                                       str(np.round(valid_acc[-1], 3)), linestyle=linestyle)
    plt.plot(epoch_numbers, train_acc, c=color2, label='{} train acc'.format(name), linestyle=linestyle)

    plt.scatter(epoch_numbers[-1], valid_acc[-1], c='black')
    return None


def plot_valid_acc(valid_acc, name, color='red', linestyle='-') -> None:
    epoch_numbers = list(range(len(valid_acc)))

    plt.plot(epoch_numbers, valid_acc, c=color, label='{} valid acc: '.format(name) + str(np.round(valid_acc[-1], 3)), linestyle=linestyle)

    plt.scatter(epoch_numbers[-1], valid_acc[-1], c='black')

    return None


def plot_valid_loss(valid_loss, name, color='red', linestyle='-') -> None:
    epoch_numbers = list(range(len(valid_loss)))

    plt.plot(epoch_numbers, valid_loss, c=color, label='{} valid loss: '.format(name) + str(np.round(min(valid_loss), 3)),
             linestyle=linestyle)

    plt.scatter(epoch_numbers[int(np.argmin(valid_loss))], min(valid_loss), c='black')

    return None


if __name__ == "__main__":
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

    lstm = CustomModel(x_train, y_train, int_to_word, path="training_81/cp.ckpt".format(time_stamp))
    lstm.train(x_train, y_train, epochs=200, _tensorboard=tensorboard, batch_size=256, verbose=1)
    # lstm.load()
    #
    # generated_lyrics = generate_rap_lyrics(lstm.model, bars_no_start, word_to_int, int_to_word, padding_length,
    #                                        num_lines=10)
    # print(generated_lyrics)

    # lstm.test(x_test, y_test)
    # events_path_train_BEST = 'logs/1618419642.2406116/train/' \
    #                          'events.out.tfevents.1618419646.DESKTOP-SUN0R45.12996.7016.v2'
    # events_path_validation_BEST = 'logs/1618419642.2406116/validation/' \
    #                          'events.out.tfevents.1618419861.DESKTOP-SUN0R45.12996.4368738.v2'
    #
    # events_path_train_noRegular = 'logs/1617572113.074653/train/events.out.tfevents.1617572117.DESKTOP-SUN0R45.7272.6970.v2'
    # events_path_validation_noRegular = 'logs/1617572113.074653/validation/events.out.tfevents.1617572292.DESKTOP-SUN0R45.7272.4353426.v2'
    #
    # events_path_train_dropout = 'logs/1617625710.6619706/train/events.out.tfevents.1617625721.DESKTOP-SUN0R45.12512.6981.v2'
    # events_path_validation_dropout = 'logs/1617625710.6619706/validation/events.out.tfevents.1617625928.DESKTOP-SUN0R45.12512.4358519.v2'
    #
    # events_path_train_overRegularise = 'logs/1619092192.7243125/train/events.out.tfevents.1619092201.DESKTOP-SUN0R45.18820.7033.v2'
    # events_path_validation_overRegularise = 'logs/1619092192.7243125/validation/events.out.tfevents.1619092416.DESKTOP-SUN0R45.18820.4381233.v2'
    #
    # events_path_train_fastOverRegularise = 'logs/1619128082.2523046/train/events.out.tfevents.1619128088.DESKTOP-SUN0R45.4304.7033.v2'
    # events_path_valdiation_fastOverRegularise = 'logs/1619128082.2523046/validation/events.out.tfevents.1619128311.DESKTOP-SUN0R45.4304.4381233.v2'
    #
    # events_path_train_slowOverRegularise = 'logs/1619172796.5665648/train/events.out.tfevents.1619172813.DESKTOP-SUN0R45.16688.7033.v2'
    # events_path_validation_slowOverRegularise = 'logs/1619172796.5665648/validation/events.out.tfevents.1619173007.DESKTOP-SUN0R45.16688.4381233.v2'

    # events_path_train_onlyDense = 'logs/1619187646.984651/train/events.out.tfevents.1619187652.DESKTOP-SUN0R45.8372.6977.v2'
    # events_path_validation_onlyDense = 'logs/1619187646.984651/validation/events.out.tfevents.1619187845.DESKTOP-SUN0R45.8372.4359440.v2'

    # events_path_train_noNorm = 'logs/1619201590.754819/train/events.out.tfevents.1619201595.DESKTOP-SUN0R45.18280.6977.v2'
    # events_path_validation_noNorm = 'logs/1619201590.754819/validation/events.out.tfevents.1619201792.DESKTOP-SUN0R45.18280.4356821.v2'
    #
    # events_path_train_batchNorm = 'logs/1619209684.23159/train/events.out.tfevents.1619209688.DESKTOP-SUN0R45.16144.7016.v2'
    # events_path_validation_batchNorm = 'logs/1619209684.23159/validation/events.out.tfevents.1619209891.DESKTOP-SUN0R45.16144.4368738.v2'

    # train_acc1, train_loss1 = collect_accuracy_loss(events_path_train_BEST)
    # valid_acc1, valid_loss1 = collect_accuracy_loss(events_path_validation_BEST)
    #
    # train_acc2, train_loss2 = collect_accuracy_loss(events_path_train_noRegular)
    # valid_acc2, valid_loss2 = collect_accuracy_loss(events_path_validation_noRegular)
    #
    # train_acc3, train_loss3 = collect_accuracy_loss(events_path_train_dropout)
    # valid_acc3, valid_loss3 = collect_accuracy_loss(events_path_validation_dropout)
    #
    # train_acc4_0, train_loss4_0 = collect_accuracy_loss(events_path_train_overRegularise)
    # valid_acc4_0, valid_loss4_0 = collect_accuracy_loss(events_path_validation_overRegularise)
    #
    # train_acc4_1, train_loss4_1 = collect_accuracy_loss(events_path_train_fastOverRegularise)
    # valid_acc4_1, valid_loss4_1 = collect_accuracy_loss(events_path_valdiation_fastOverRegularise)
    #
    # train_acc4_2, train_loss4_2 = collect_accuracy_loss(events_path_train_slowOverRegularise)
    # valid_acc4_2, valid_loss4_2 = collect_accuracy_loss(events_path_validation_slowOverRegularise)

    # train_acc5_0, train_loss5_0 = collect_accuracy_loss(events_path_train_onlyDense)
    # valid_acc5_0, valid_loss5_0 = collect_accuracy_loss(events_path_validation_onlyDense)

    train_acc6, train_loss6 = collect_accuracy_loss(events_path_train_noNorm)
    valid_acc6, valid_loss6 = collect_accuracy_loss(events_path_validation_noNorm)

    # # =====================================================================================================

    # plt.title('Different accuracies without early stopping for initial models')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    #
    # plot_accuracy(train_acc1, valid_acc1, 'Final model')
    # plot_accuracy(train_acc2, valid_acc2, 'No regularisation', color1='orange', color2='blue', linestyle='--')
    # plot_accuracy(train_acc3, valid_acc3, 'Only dropout', color1="yellow", color2='purple', linestyle='--')
    #
    # plt.legend(loc='upper left')
    #
    # plt.show()

    # # =====================================================================================================

    # plt.title('Different losses (without early stopping)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    #
    # plot_loss(train_loss1, valid_loss1, 'Final')
    # plot_loss(train_loss2, valid_loss2, 'No regularisation', color1='orange', color2='blue', linestyle='--')
    # plot_loss(train_loss3, valid_loss3, 'Only dropout', color1='yellow', color2='purple', linestyle='--')
    #
    # plt.legend(loc='lower left')
    #
    # plt.show()

    # # =====================================================================================================

    # plt.title('The effects of ridge regularisation on LSTM layers')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    #
    # plot_accuracy(train_acc1, valid_acc1, 'Final model')
    # plot_valid_acc(valid_acc4_0, 'L2 - LR=0.001', color='orange', linestyle='--')
    # plot_valid_acc(valid_acc4_1, 'L2 - LR=0.01', color='yellow', linestyle='--')
    # plot_valid_acc(valid_acc4_2, 'L2 - LR=0.0001', color='pink', linestyle='--')
    #
    # plt.legend(loc='upper right')
    #
    # plt.show()

    # # =====================================================================================================

    # plt.title('The effects of ridge regularisation on LSTM layers')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    #
    # plot_accuracy(train_loss1, valid_loss1, 'Final model')
    # plot_valid_loss(valid_loss4_0, 'L2 - LR=0.001', color='orange', linestyle='--')
    # plot_valid_loss(valid_loss4_1, 'L2 - LR=0.01', color='yellow', linestyle='--')
    # plot_valid_loss(valid_loss4_2, 'L2 - LR=0.0001', color='pink', linestyle='--')
    #
    # plt.legend(loc='upper right')
    #
    # plt.show()

    # # =====================================================================================================

    # plt.title('Only regularising the dense layer')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    #
    # plot_accuracy(train_acc1, valid_acc1, 'Final model')
    # plot_accuracy(train_acc5_0, valid_acc5_0, 'L2 Dense LR=0.001', color1='orange', color2='blue', linestyle='--')
    # plot_valid_acc(valid_acc4_0, 'L2 LR=0.001', color='purple', linestyle='--')
    #
    # plt.legend(loc='upper right')
    #
    # plt.show()

    # # =====================================================================================================


