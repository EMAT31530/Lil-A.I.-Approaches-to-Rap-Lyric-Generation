import WordFunctions as WF
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
import sys


def plot_word_count(count, data_name='Training'):
    xticks = range(len(count))

    plt.title('Word frequency histogram for {} data'.format(data_name))
    plt.xlabel('Word index')
    plt.ylabel('Frequency')

    keys = list(xticks)
    values = list(count.values())
    sns.histplot(x=keys, weights=values, discrete=True)

    plt.show()


def mean_word_freq(count, name='train'):
    freqs = list(count.values())
    mean = np.mean(freqs)
    mean = np.round(mean, 4)
    print('Mean word frequency for {} data: {}'.format(name, mean))
    return mean


def std_word_freq(count, name='train'):
    freqs = list(count.values())
    std = np.std(freqs)
    std = np.round(std, 4)
    print('Std. word frequency for {} data: {}'.format(name, std))
    return std


def rhyme_score(count, path) -> float:
    train_phonemes = WF.g2p_dict(count)
    lines = WF.__generate_content(path, import_data=False)
    rhyme_metric = [WF.rhyme_metric(train_phonemes[lines[i].split()[-1]], train_phonemes[lines[i + 1].split()[-1]])
                    for i in range(len(lines) - 1)]
    rhyme_metric = np.mean(rhyme_metric)
    return np.round(rhyme_metric, 4)


def print_all_stats(path, name='Training') -> None:
    content_all = WF.all_words_list(path, import_data=False)
    word_freq = Counter(content_all)

    print('--------------')
    print('Total number of words:', len(content_all))
    # print('Training word frequency:', word_freq)
    print('Number of unique words:', len(word_freq))
    mean_word_freq(word_freq, name)
    std_word_freq(word_freq, name)
    print('Rhyme score for {}:'.format(name), rhyme_score(word_freq, path))

    return None


print_all_stats('AllLyrics_uncleanOLD.txt', name='Training')
print_all_stats('final_lyrics1.txt', name='RNN output')
print_all_stats('markovsylllyrics.txt', name='Syllable output')
print_all_stats('MarkovLyrics.txt', name='Markov output')
print_all_stats('eminemUTF8.txt', name='Eminem')

