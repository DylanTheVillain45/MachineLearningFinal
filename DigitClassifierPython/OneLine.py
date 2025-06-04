import numpy as np

# 1) Load weights/biases exactly as you did before:
W1 = np.loadtxt("Weights/W1.csv", delimiter=",")   # shape (128, 784)
b1 = np.loadtxt("Weights/b1.csv", delimiter=",").reshape((128, 1))  # shape (128, 1)

W2 = np.loadtxt("Weights/W2.csv", delimiter=",")   # shape ( 64, 128)
b2 = np.loadtxt("Weights/b2.csv", delimiter=",") .reshape((64, 1))  # shape ( 64, 1)

W3 = np.loadtxt("Weights/W3.csv", delimiter=",")   # shape ( 10,  64)
b3 = np.loadtxt("Weights/b3.csv", delimiter=",").reshape((10, 1))   # shape ( 10, 1)

# 2) Grab exactly row 0 of mnist_test.csv (or pick any index you like)
mnist = np.loadtxt("mnist_test.csv", delimiter=",")  # shape (10000, 785)
idx = 53
row0 = mnist[idx]  # a length‐785 array: [label, pixel0, pixel1, …, pixel783]

# 3) Build X exactly as Unity is doing: 28×28 flatten, divide by 255
X = row0[1:] / 255.0           # shape (784,)
X = X.reshape((784, 1))         # column vector shape (784, 1)

# 4) Now run layer‐by‐layer forward‐prop in Python and print
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    exps = np.exp(Z_shift)
    return exps / np.sum(exps, axis=0, keepdims=True)

# Layer 1:
Z1_py = W1.dot(X) + b1        # shape (128, 1)
A1_py = relu(Z1_py)           # shape (128, 1)

# Layer 2:
Z2_py = W2.dot(A1_py) + b2     # shape (64, 1)
A2_py = relu(Z2_py)            # shape (64, 1)

# Layer 3 (output):
Z3_py = W3.dot(A2_py) + b3     # shape (10, 1)
A3_py = softmax(Z3_py)         # shape (10, 1)

# 5) Print out the **first 5 entries** of each Z and A so you can compare:
print("---- Python debug for index", idx, "----")
print("Z1_py[:5] =", Z1_py[:5].flatten())
print("A1_py[:5] =", A1_py[:5].flatten())
print("Z2_py[:5] =", Z2_py[:5].flatten())
print("A2_py[:5] =", A2_py[:5].flatten())
print("Z3_py[:5] =", Z3_py[:5].flatten())
print("A3_py[:5] =", A3_py[:5].flatten())
print(np.argmax(A3_py))
