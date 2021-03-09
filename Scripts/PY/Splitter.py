import pandas as pd
from sklearn.model_selection import train_test_split

dictionary = pd.read_csv('dic.csv')

train, test = train_test_split(dictionary, test_size=0.3)  # 70% training data, 30% testing
train.to_csv('training_data.csv')
test.to_csv('testing_data.csv')
