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
    s = sigmoid(z)
    #plug and chug
    return s*(1-s)

def visualize_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

def calc_cost(y_enc, outpt):
    #look at the screenshot for cost function NN formula
    t1 = -y_enc * np.log(outpt)    #yk ^i is -y_enc     log is np.log     h theta x(i) is output
    t2 = (1 - y_enc)*np.log(1-outpt)  #t2 is everything thats after the plut sign in formula
    cost = np.sum(t1 - t2) #the summation but since -1/m in teh front we take the differnece?
    return cost

def bias_unit(X, where):
    #where is row or col
    #adding bias unit is like ax+b, need b to move ax up and down, left right to have a better fit
    #https://stackoverflow.com/questions/2480650/what-is-the-role-of-the-bias-in-neural-networks
    if where == 'column':
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
    elif where == 'row':
        X_new = np.ones((X.shape[0] + 1, X.shape[1]))
        X_new[1:, :] = X
    return X_new

def init_weights(n_features, n_hidden, n_output):
    w1 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_features+1))
    #uniform create np array with uniform distribution
    #can maybe be replaced with ReLU?
    w1 = w1.reshape(n_hidden, n_features+1)
    w2 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_hidden+1))
    w2 = w2.reshape(n_hidden, n_hidden+1)
    w3 = np.random.uniform(-1.0, 1.0, size=n_output*(n_hidden+1))
    w3 = w3.reshape(n_output, n_hidden+1)
    return w1, w2, w3

def feed_forward(x, w1,w2,w3):
    #everything here is of the screenshot "howdoes a NN compute"
    #col within rows is just bytes of data so we hadd col vector
    a1= bias_unit(x, where = "column")
    z2 = w1.dot(a1.T)
    a2 = sigmoid(z2)

    #after transpose we add bias units to row
    a2= bias_unit(a2, where = "row")
    z3 = w2.dot(a2)
    a3 = sigmoid(z3)

    a3 = bias_unit(a3, where ="row")
    z4 = w3.dot(a3)
    a4 = sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, a4

def predict(x, w1,w2,w3):
    a1, z2, a2, z3, a3, z4, a4 = feed_forward(x,w1,w2,w3)
    y_pred = np.argmax(a4, axis=0)
    return y_pred

def calc_grad(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
    #this is back propogation
    #see screnshot "gradient computation: back propogation algo"
    delta4 = a4-y_enc
    z3 = bias_unit(z3, where="row")
    delta3 = w3.T.dot(delta4)*sigmoid_gradiant(z3) #sigmoid gradiant is the g prime 
    delta3 = delta3[1:, :] #discard first row because it's the bias unit aka constant

    z2 = bias_unit(z2, where= "row")
    delta2 = w2.T.dot(delta3)*sigmoid_gradiant(z2)
    delta2 = delta2[1:, :]

    grad1 = delta2.dot(a1)
    grad2= delta3.dot(a2.T)
    grad3 = delta4.dot(a3.T)

    return grad1, grad2, grad3


def runModel(x, y, x_test, y_test):
    x_copy , y_copy = x.copy(), y.copy()  #copies for shuffling for better quality model
    y_enc = onehot_encoding(y) 
    epoch = 50 #epoch is how many times we wanna train 1 batch
    batch = 50 #batch is the amount of data we wanna train at once, images or etc
    

    w1,w2,w3 = init_weights(784, 75, 10)    #75 could be any #, 10 is 10 digits

    alpha = 0.001    #learing rate, how big of a step we take in parameter space, for gradients
    eta = 0.001     #global minimum
    dec = -0.00001   #decrease every epoch
    delta_w1_prev = np.zeros(w1.shape)   
    delta_w2_prev = np.zeros(w2.shape)
    delta_w3_prev = np.zeros(w3.shape)

    for i in range(epoch):
        total_cost = []
        shuffle = np.random.permutation(y_copy.shape[0])
        x_copy, y_enc = x_copy[shuffle], y_enc[:, shuffle]

        mini = np.array_split(range(y_copy.shape[0]), batch)

        for step in mini:
            #feed forward
            a1,z2,a2,z3,a3,z4,a4 = feed_forward(x_copy[step],w1,w2,w3)
            cost = calc_cost(y_enc[:,step],a4)

            total_cost.append(cost)

            #back propogate
            grad1,grad2,grad3 = calc_grad(a1,a2,a3,a4,z2,z3,z4,y_enc[:,step], w1,w2,w3)
            delta_w1,delta_w2,delta_w3 = eta*grad1, eta*grad2, eta*grad3

            w1 -= delta_w1 + alpha * delta_w1_prev
            w2 -= delta_w2 + alpha * delta_w2_prev
            w3 -= delta_w3 + alpha * delta_w3_prev

            delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2,delta_w3_prev

        print("epoch #", i)
    y_pred = predict(x_test, w1,w2,w3)
    acc = np.sum(y_test == y_pred, axis =0) / x_test.shape[0]
    print("training accuracey", acc*100)
    return 1

train_x, train_y, test_x, test_y = load_data()

y = runModel(train_x, train_y, test_x, test_y)