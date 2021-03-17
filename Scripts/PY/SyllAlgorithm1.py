import pandas as pd
import numpy

validation = pd.read_csv('validation_data.csv')

validation_in = []
validation_out = []
estimated_count = []
success = 0
total = 0

for row in validation.itertuples():
    validation_in.append(row.Word)
    validation_out.append(row.Number_of_Syllables)


def syllable_count(word):
    word = word.lower()
    syllables = 0
    vowels = 'aeiouy'
    if word[0] in vowels:
        syllables += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            syllables += 1
    if word.endswith('e'):
        syllables -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        syllables += 1
    if syllables == 0:
        syllables = 1
    return syllables


for n, word in enumerate(validation_in):
    word_count = syllable_count(word)
    estimated_count.append(word_count)
    if word_count == validation_out[n]:
        total += 1
        success += 1
    else:
        total += 1

print('The accuracy is', success/total)
