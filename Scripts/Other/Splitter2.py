import pandas as pd
from sklearn.model_selection import train_test_split

dictionary = pd.read_csv('dic.csv')

train, test = train_test_split(dictionary, test_size=0.2)  # 20% testing data
train, validation = train_test_split(train, test_size=0.25)  # 60% training data, 20% validation data
train.to_csv('training_data.csv')
test.to_csv('testing_data.csv')
validation.to_csv('validation_data.csv')
