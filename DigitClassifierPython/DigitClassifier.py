import numpy as np
import pandas as pd
import os

data = pd.read_csv("./data.csv").to_numpy()
np.random.shuffle(data)
m, n = data.shape

data_dev = data[0:1000].T
Y_dev = data_dev[0].astype(int)
X_dev = data_dev[1:n].astype(float)

data_train = data[1000:m].T
Y_train = data_train[0].astype(int)
X_train = data_train[1:n].astype(float)

X_train /= 255.0
X_dev   /= 255.0

def ReLU(Z):
    return np.maximum(0, Z)

def Deriv_ReLU(Z):
    return (Z > 0).astype(float)

def SoftMax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shifted) 
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


def init_params():
    # W1: (128, 784),   b1: (128, 1)
    # W2: (64, 128),    b2: (64, 1)
    # W3: (10, 64),     b3: (10, 1)
    W1 = np.random.randn(128, 784) * np.sqrt(2/784)
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(64, 128) * np.sqrt(2/128)
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(10, 64) * np.sqrt(2/64)
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = SoftMax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def back_prop(Z1, A1, Z2, A2, Z3, A3, W3, W2, X, Y):
    """
    Calculate gradients for all params
    X: (784, m)
    Y: (m, 1) labels of 0-9
    A3: (10, m)
    """
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ3 = A3 - one_hot_Y

    dW3 = (1 / m) * dZ3.dot(A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * Deriv_ReLU(Z2)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * Deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def get_predictions(A3):
    return np.argmax(A3, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def save_parameters_to_csv(W1, b1, W2, b2, W3, b3, folder="./Weights"):
    os.makedirs(folder, exist_ok=True)
    np.savetxt(os.path.join(folder, "W1.csv"), W1, delimiter=",")
    np.savetxt(os.path.join(folder, "b1.csv"), b1, delimiter=",")
    np.savetxt(os.path.join(folder, "W2.csv"), W2, delimiter=",")
    np.savetxt(os.path.join(folder, "b2.csv"), b2, delimiter=",")
    np.savetxt(os.path.join(folder, "W3.csv"), W3, delimiter=",")
    np.savetxt(os.path.join(folder, "b3.csv"), b3, delimiter=",")

    with open(os.path.join(folder, "dev_accuracy.txt"), "w") as f:
        f.write(f"{best_dev_acc:.6f}")

def Gradient_Descent(X, Y, epochs, alpha):
    W1, b1, W2, b2, W3, b3 = init_params()

    global best_dev_acc
    best_dev_acc = 0.0

    for i in range(1, epochs+1):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W3, W2, X, Y)

        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        W3 = W3 - alpha * dW3
        b3 = b3 - alpha * db3

        if i % 10 == 0:
            train_preds = get_predictions(A3)
            train_acc = get_accuracy(train_preds, Y)

            _, _, _, _, _, A3_dev = forward_prop(W1, b1, W2, b2, W3, b3, X_dev)
            dev_preds = get_predictions(A3_dev)
            dev_acc = get_accuracy(dev_preds, Y_dev)
      
            print(f"Iteration {i}/{epochs} — train_acc: {train_acc:.4f} — dev_acc: {dev_acc:.4f}")

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_parameters_to_csv(W1, b1, W2, b2, W3, b3)
                print(f"  ➜ New best dev_acc={dev_acc:.4f}; weights saved")


    return W1, b1, W2, b2, W3, b3

epochs = 500
learning_rate = 0.05

W1, b1, W2, b2, W3, b3 = Gradient_Descent(X_train, Y_train, epochs, learning_rate)

    

