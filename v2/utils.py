import matplotlib.pyplot as plt
import tensorflow as tf

def plot_train(training):
    plt.plot(training.history['loss'],label='loss')
    plt.plot(training.history['val_loss'],label ='val_loss')
    plt.ylim([0,100])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pred(x,y,train_df,train_labels):
    plt.scatter(train_df['Spent'],train_labels,label='Data')
    plt.plot(x,y,color='k',label='predictions')
    plt.xlabel('Spent')
    plt.ylabel('Payout')
    plt.legend()
    plt.show()

def derivative(model, point):
    pre = point - .01
    post = point + .01
    target = tf.linspace(pre,post,2)
    predictions = model.predict(target)
    m = (predictions[-1]-predictions[0])/(post-pre)
    return float(m)