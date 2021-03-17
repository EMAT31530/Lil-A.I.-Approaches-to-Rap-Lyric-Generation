import pandas as pd
import numpy
import pyphen

validation = pd.read_csv('validation_data.csv')

validation_in = []
validation_out = []
estimated_count = []
success = 0
total = 0

for row in validation.itertuples():
    validation_in.append(row.Word)
    validation_out.append(row.Number_of_Syllables)

dic = pyphen.Pyphen(lang='en_EN')  # set pyphen dictionary to English

for n, word in enumerate(validation_in):
    word_syllables = dic.inserted(word)
    word_count = len(word_syllables.split('-'))
    estimated_count.append(word_count)
    if word_count == validation_out[n]:
        total += 1
        success += 1
    else:
        total += 1

print('The accuracy is', success/total)
