import struct
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import expit

def load_data():
    with open("train-labels-idx1-ubyte", "rb") as labels:
        magic, n = struct.unpack(">II", labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)

    with open("train-images-idx3-ubyte", "rb") as imgs:
        magic, num, nrows, ncols = struct.unpack(">IIII", imgs.read(16))

        train_images = np.fromfile(imgs, dtype= np.uint8).reshape(num,784)  
        #784 is becasue each picture of the digit is 28 pixel * 28pixel need to flatten it out

    with open("t10k-labels-idx1-ubyte", "rb") as labels:
        magic, n = struct.unpack(">II", labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)

    with open("t10k-images-idx3-ubyte", "rb") as imgs:
        magic, num, nrows, ncols = struct.unpack(">IIII", imgs.read(16))
        test_images = np.fromfile(imgs, dtype= np.uint8).reshape(num,784)  

    return train_images, train_labels, test_images, test_labels


def vistualize_data(img_array, label_array):
    fig, ax = plt.subplots(nrows=10,ncols=10,sharex=True, sharey=True)
    ax= ax.flatten()

    for i in range(100):
        img = img_array[label_array][i].reshape(28,28)
        ax[i].imshow(img,cmap="Greys", interpolation="nearest")
    plt.show()

# train_x,train_y, test_x,test_y =load_data()

# vistualize_data(train_x,train_y)

def onehot_encoding(y, num_labels =10):
    one_hot = np.zeros((num_labels, y.shape[0]))
    # print(one_hot)
    for i, val in enumerate(y):
        one_hot[val,i]= 1.0
    return one_hot

def sigmoid(z):
    # return (1/(1 + np.exp(-z)))
    return expit(z)

def visualize_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

visualize_sigmoid()
