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
    #expit is the same as the one above, just a built in from scipu
    return expit(z)

def sigmoid_gradiant(z):
    s = sigmoid()
    #plug and chug
    return s*(1-s)

def visualize_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

def calc_cost(y_enc, output):
    #look at the screenshot for cost function NN formula
    t1 = -y_enc *np.log(output)    #yk ^i is -y_enc     log is np.log     h theta x(i) is output
    t2 = (1-yenc) * np.log(1- output)  #t2 is everything thats after the plut sign in formula
    cost = np.sum(t1-t2) #the summation but since -1/m in teh front we take the differnece?
    return cost

visualize_sigmoid()
