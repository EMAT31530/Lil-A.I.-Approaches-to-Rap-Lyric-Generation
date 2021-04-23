import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

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

history = model.fit(train_in, train_out, epochs=80, batch_size=500, validation_data=(test_in, test_out), verbose=2)

results = model.evaluate(validation_in, validation_out)
print('Accuracy is', results[1])
