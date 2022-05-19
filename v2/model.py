import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers

from utils import plot_train, plot_pred, derivative

# Load a dataset in a Pandas dataframe.
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
train_labels = train_df.pop('Payout')

# Normalization
norm_train = tf.keras.layers.Normalization(axis=-1)
norm_test = tf.keras.layers.Normalization(axis=-1)

payout = np.array(train_df['Spent'])
payout_normalizer = layers.Normalization(input_shape=[1,],axis=None)
payout_normalizer.adapt(payout)

model = tf.keras.Sequential([
    payout_normalizer,
    layers.Dense(units=1)
])

model.summary()

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=.1),
    loss = 'mean_absolute_error'
)

training = model.fit(
    train_df['Spent'],
    train_labels,
    epochs = 100,
    verbose = 0,
    validation_split = .2
)

hist = pd.DataFrame(training.history)
hist['epoch'] = training.epoch
hist.tail()

x = tf.linspace(0,30,20)
y = model.predict(x)

print(derivative(model, 3))
print(derivative(model, 4))
print(derivative(model, 5))

#plot_pred(x,y,train_df,train_labels)
print('Program finished with sucess')