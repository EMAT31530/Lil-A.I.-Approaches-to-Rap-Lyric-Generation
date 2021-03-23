import pandas as pd

with open('SyllableDictionary.txt') as file:
    content = file.readlines()

content = [x.strip() for x in content]
number_syllables = []
number_word = []

split_content = [word for line in content for word in line.split('=')]

words = split_content[::2]
syllables = split_content[1::2]
words_final = []

for word in words:
    while len(word) <= 16:  # because of numpy arrays, need to add empty spaces so all words have same num of characters
        word += ' '
    words_final.append(word)

for string in syllables:
    new_string = string.split('·')
    number_syllables.append(len(new_string))

df = pd.DataFrame({'Word': words_final, 'Number_of_Syllables': number_syllables})
df.to_csv('dic.csv')

#for word in words:
#    number_word.append(len(word))

#print(max(number_syllables))
#print(max(number_word))
