import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import random

np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers

def data_generator(file_name, number_rows, line_scope):
    with open(file_name, mode='w') as csv_file:
        fieldnames = ['Spent', 'Payout']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(number_rows):
            spent = random.randint(0, 100)
            payout = round(spent * line_scope * random.uniform(0.95, 1.05), 2)
            row = {'Spent': spent, 'Payout': payout}
            writer.writerow(row)

def derivative(model, point):
    """
    Given a tensorflow model and a point, approximates the derivative.
    """
    pre = point - .01
    post = point + .01
    target = tf.linspace(pre,post,2)
    predictions = model.predict(target)
    m = (predictions[-1]-predictions[0])/(post-pre)
    return float(m)

def fit(data_path):
    """
    Given some campaign data, fit the model, train it, and return the model. 
    """
    data = pd.read_csv(data_path)
    payout = np.array(data['Spent'])
    payout_normalizer = layers.Normalization(input_shape=[1,],axis=None)
    payout_normalizer.adapt(payout)

    model = tf.keras.Sequential([
        payout_normalizer,
        layers.Dense(units=1)
    ])
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=.1),
        loss = 'mean_absolute_error'
    )
    training = model.fit(
        data['Spent'],
        data['Payout'],
        epochs = 100,
        verbose = 0,
        validation_split = .2
    )
    return model

"""
--- VISUALIZATION FUNCTIONS ---
"""

def plot(data):
    plt.scatter(data['Spent'],data['Payout'],label='Data')
    plt.xlabel('Spent')
    plt.ylabel('Payout')
    plt.legend()
    plt.show()

def plot_train(training):
    plt.plot(training.history['loss'],label='loss')
    plt.plot(training.history['val_loss'],label ='val_loss')
    plt.ylim([0,100])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pred(model,data_path):
    data = pd.read_csv(data_path)
    x = tf.linspace(0,30,20)
    y = model.predict(x)
    plt.scatter(data['Spent'],data['Payout'],label='Data')
    plt.plot(x,y,color='k',label='predictions')
    plt.xlabel('Spent')
    plt.ylabel('Payout')
    plt.legend()
    plt.show()

def recomendation_printer(model,data_path,budget):
    data = pd.read_csv(data_path)
    x = tf.linspace(0,30,20)
    y = model.predict(x)
    recom = model.predict(budget)
    plt.scatter(data['Spent'],data['Payout'],label='Data')
    plt.plot(x,y,color='k',label='predictions')
    plt.scatter(budget,recom,label='budget recommendation')
    plt.xlabel('Budget')
    plt.ylabel('Expected Payout')
    plt.legend()
    plt.show()