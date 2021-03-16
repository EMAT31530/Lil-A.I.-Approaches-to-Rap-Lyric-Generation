import pandas as pd

with open('SyllableDictionary.txt') as file:
    content = file.readlines()

content = [x.strip() for x in content]
number_syllables = []
number_word = []

split_content = [word for line in content for word in line.split('=')]

words = split_content[::2]
syllables = split_content[1::2]

for string in syllables:
    new_string = string.split('·')
    number_syllables.append(len(new_string))

df = pd.DataFrame({'Word': words, 'Number_of_Syllables': number_syllables})
df.to_csv('dic.csv')
