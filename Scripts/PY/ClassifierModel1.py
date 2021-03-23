import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

train = pd.read_csv('training_data.csv')
test = pd.read_csv('testing_data.csv')
validation = pd.read_csv('validation_data.csv')

train_in = []
test_in = []
train_out = []
test_out = []
validation_in = []
validation_out = []

for row in train.itertuples():
    # train_in.append(row.Word)
    train_out.append(row.Number_of_Syllables)
    if row.Word == '                ':  # an empty word was getting in for some reason
        pass
    else:
        temp = list(row.Word)
        for i in range(len(temp)):
            temp[i] = ord(temp[i])
        train_in.append(temp)

for row in test.itertuples():
    # test_in.append(row.Word)
    test_out.append(row.Number_of_Syllables)
    if row.Word == '                ':
        pass
    else:
        temp = list(row.Word)
        for i in range(len(temp)):
            temp[i] = ord(temp[i])
        test_in.append(temp)

for row in validation.itertuples():
    # test_in.append(row.Word)
    validation_out.append(row.Number_of_Syllables)
    if row.Word == '                ':
        pass
    else:
        temp = list(row.Word)
        for i in range(len(temp)):
            temp[i] = ord(temp[i])
        validation_in.append(temp)

test_in = np.array(test_in)
test_out = np.array(test_out)
train_in = np.array(train_in)
train_out = np.array(train_out)
validation_in = np.array(validation_in)
validation_out = np.array(validation_out)

max_word = 143

model = keras.Sequential()
model.add(keras.layers.Embedding(max_word, 100))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(100, activation=tf.nn.relu))
model.add(keras.layers.Dense(50, activation=tf.nn.relu))
model.add(keras.layers.Dense(8, activation=tf.nn.softmax))
model.summary()

model.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.2, use_locking=False),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_in, train_out, epochs=80, batch_size=100, validation_data=(test_in, test_out), verbose=2)

results = model.evaluate(validation_in, validation_out)
print('Accuracy is', results[1])


def rap():
    num_lines = int(input('How many lines would you like the rap to be? '))
    num_generated_lines = int(input('How many lines should be generated to choose from? '))
    count = int(input("How many syllables per line? "))

    # Extract all of Mike's lyrics.
    text = open("AllLyrics_unclean.txt", "r").read()
    vocabulary = ''.join([i for i in text if not i.isdigit()]).replace("\n", " ").split(' ')

    # Generate text
    def line_generator(vocab):
        index = 1
        chain = {}
        # count = 16 # https://colemizestudios.com/rap-lyrics-syllables/, apparently rappers usually use semiquavers
        line_count = 0
        number_of_tries = 0

        for word in vocab[index:]:
            key = vocab[index - 1]
            if key in chain:
                chain[key].append(word)
            else:
                chain[key] = [word]
            index += 1

        word1 = random.choice(list(chain.keys()))
        line = word1.capitalize()
        word1_with_spaces = word1
        while len(word1_with_spaces) < 16:
            word1_with_spaces += ' '
        temp_word = list(word1_with_spaces)
        for i in range(len(temp_word)):
            temp_word[i] = ord(temp_word[i])
        temp_word = np.array(temp_word)
        word_syllables = np.argmax(model.predict(temp_word), axis=-1)
        word_count = word_syllables[0]
        line_count += word_count

        while line_count < count:
            number_of_tries += 1
            word2 = random.choice(chain[word1])
            word2_with_spaces = word2
            while len(word2_with_spaces) < 16:
                word2_with_spaces += ' '
            temp_word = list(word2)
            for i in range(len(temp_word)):
                temp_word[i] = ord(temp_word[i])
            temp_word = np.array(temp_word)
            word_syllables = np.argmax(model.predict(temp_word), axis=-1)
            word_count = word_syllables[0]
            line_count += word_count
            # print(n)
            if line_count > count:  # don't include word if it makes line go over syllable count
                line_count -= word_count
            else:
                word1 = word2
                line += ' ' + word2.lower()
            if number_of_tries > 99:  # if not finding a word with right number of syllables, stop trying
                line += ' ERROR FINDING CORRECT SYLLABLE WORD'
                line_count = count
        return line

    # Rhyme Functions
    def reverse_syllable_extract(text):
        sy_form = []
        characters = [char for char in text]
        sylls = ['a', 'e', 'i', 'o', 'u', 'y']
        for x in characters:
            if x in sylls:
                sy_form.append(x)
        sy_form.reverse()
        return sy_form

    def rev_syllable_stop_count(text1, text2):
        counter = True
        i = 0
        counter = 0
        syll1 = reverse_syllable_extract(text1)
        syll2 = reverse_syllable_extract(text2)
        while counter:
            if i < min(len(syll1), len(syll2)) and syll1[i] == syll2[i]:
                counter += 1
                i += 1
            else:
                counter = False
        return counter

    def next_line_stop_count(start_line, lines):
        sy_lines = []
        for i in lines:
            sy_lines.append(rev_syllable_stop_count(start_line, i))
        choice = sy_lines[0]
        count = 0
        for i in range(len(sy_lines)):
            if sy_lines[i] > choice:
                choice = sy_lines[i]
        return lines[sy_lines.index(choice)]

    start_line = line_generator(vocabulary)
    done = False
    while not done:
        if 'ERROR FINDING CORRECT SYLLABLE WORD' in start_line:
            start_line = line_generator(vocabulary)
        else:
            done = True

    all_other_lines = [line_generator(vocabulary) for i in range(num_generated_lines)]
    rap = [start_line]

    for n, line in enumerate(all_other_lines):
        done = False
        while not done:
            if 'ERROR FINDING CORRECT SYLLABLE WORD' in line:
                line = line_generator(vocabulary)
                all_other_lines[n] = line
            else:
                done = True

    for i in range(num_lines):
        if i % 2 == 1:
            next_line = next_line_stop_count(rap[len(rap) - 1], all_other_lines)
        else:
            next_line = random.choice(all_other_lines)
        all_other_lines.remove(next_line)
        rap.append(next_line)
    return rap


rap()
