# ~~~~~~ IMPORTS ~~~~~~
import numpy as np
import os
import sys
import string
import re  # For splitting strings with multiple delimiters

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

import PipelineV6 as pipeline
# ~~~~~~ References~~~~~~


# ~~~~~~ Functions ~~~~~~
def write_essay(essay):
    # Add a space to the file, to make sure it always exists
    with open('current.txt', 'w') as f:
        f.write(' ')
    f.close()

    # Clear the text file
    with open('current.txt', 'r+') as f:
        f.truncate(0)
    f.close()

    # Write all of the lines to the file
    with open('current.txt', 'w') as f:
        for line in essay:
            line_str = str(line)
            f.write(line_str)
            f.write("\n")  # Line ends!

    f.close()
    return essay


def essay_tolistandstr(essay):
    # Convert the essay to a np array then to a list then to a string, removing stopwords in the process
    # Convert the essay to a np array then to a list then to a string
    essay_arr = essay.to_numpy()

    print('shape before: ', np.shape(essay_arr))

    # np to list
    essay_lst = essay_arr.tolist()

    # Remove line breaks
    essay_str = ' '.join(str(v) for v in essay_lst if str(v) != '\n')

    # Split by punctuation and remove extra whitespace
    essay_lst = re.split(r'[;,<0123456789>/"=()!.\s]\s*', essay_str)

    # Trim the stopwords
    essay_str = ' '.join(str(v) for v in essay_lst if (str(v) not in stopwords2 and str(v) not in punctuation))

    # Update the list
    essay_lst = essay_str.split()

    print('shape before: ', np.shape(essay_lst))

    return zip(essay_str, essay_lst)


def create_wordcloud(text):

    mask = np.array(Image.open(os.path.join(currdir, "cloud.png")))

    wc = WordCloud(background_color="white",
                   mask=mask,
                   max_words=200)

    wc.generate(text)

    wc.to_file(os.path.join(currdir, "wc.png"))


# ~~ Globals ~~
# Stopwords to be removed for formatting
stopwords_format = {'br', '<br>', 'href', '\n', '<br />', "i'm", 'nan', '/>',
                    '<br', 'ilink', '"', 'r', 'n', 'i', 'of', 'for', 'in'
                    'to', 'as', 'an', 'or', 'www', 'com', 'sf', 'p', 'sx', 'az', 'im', 'se'}

numbers = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}

stopwords_uncommon = {'right', 'now', 'currently', 'will', 'often', 'though', 'even',
                      'amp', 'thing', 'stuff', 'know', 'best', 'mean', 'moment',
                      'still', 'point', 'want', 'bit', 'something', 'etc', 'got', 'possible',
                      'oh', 'interests', 'things', 'way'}

stopwords_specific = {'trying', 'day', 'home', 'getting', 'local', 'see', 'go', 'week', 'day', 'year', 'crazy',
                      'around', 'need', 'far', 'take', 'end', 'place', 'keep', 'last', 'days', 'weeks', 'say',
                      'time', 'pretty', 'open', 'find', 'side', 'well', 'come', 'sometime', 'let', 'guy', 'seem',
                      'least', 'free', 'mean', 'started', 'began', 'begun', 'back', 'month', 'away', 'wanna',
                      'guess', 'either', 'years', 'ago', 'less', 'yes', 'made', 'us', 'become', 'done', 'used',
                      'try', 'event', 'use', 'past', 'hard', 'sure', 'worked', 'finding', 'believe', 'said',
                      'involved', 'process'}

stopwords_connectives = {'and', 'but', 'then', 'because', 'recently', 'sometime', 'although', 'however', 'lately'}

stopwords_qualifiers = {'very', 'much', 'lot', 'little', 'big', 'small', 'quite', 'amazing', 'may',
                        'new', 'really', 'mostly', 'normally', 'super', 'show', 'start', 'interesting',
                        'bigger', 'great', 'probably', 'lots', 'alway', 'long', 'first', 'awesome',
                        'always', 'good', 'bad', 'many', 'few', 'usually', 'maybe', 'nice', 'kind',
                        'like', 'never', 'actually', 'commonly', 'old', 'might', 'enough', 'yet', 'moved', 'next',
                        'finally', 'lastly', 'clever', 'smart', 'every', 'cool', 'definitely', 'absolutely'}

pipeline.import_github(write_type='both')

# Wordcloud common stopwords
stopwords_wc = STOPWORDS
# Form the total stopwords list
stopwords2 = stopwords_format | numbers | stopwords_wc | stopwords_uncommon | stopwords_connectives \
             | stopwords_specific
punctuation = string.punctuation
letters = string.ascii_letters

# ~~ Main Code ~~
# Load the ratings data
print('Loading Profiles ...')
with open("AllLyrics_clean.txt", 'r') as temp_file:
    dataF = temp_file.readlines()
print('Profiles Loaded')


# Convert from a DataFrame to a numpy array
essay1_lst = dataF

# Keep track of the current directory
currdir = os.path.dirname(__file__)

print('Minimum size before: ', len(essay1_lst))

# Remove line breaks
essay1_str = ' '.join(str(v) for v in essay1_lst if str(v) != '\n')

# Split by punctuation and remove extra whitespace
essay1_lst = re.split(r'[:;,&%+_$^*<>0123456789/"=()?!.\s]\s*', essay1_str)

# Trim the stopwords
essay1_str = ' '.join(str(v) for v in essay1_lst if (str(v) not in stopwords2 and
                                                     str(v) not in punctuation and
                                                     str(v) not in letters))

# Update the list
essay1_lst = essay1_str.split()

print('Shape After: ', np.shape(essay1_lst))

create_wordcloud(essay1_str)

print('Wordcloud generated!')

write_essay(essay1_lst)

print('Text file generated!')
