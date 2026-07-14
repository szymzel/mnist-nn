import numpy as np

def relu(z):
    relu_output = np.maximum(0,z)
    return relu_output



def softmax(z):
    z = z - np.max(z)
    softmax_output = np.exp(z)/np.sum(np.exp(z))
    return softmax_output


def init_params():
    W1 = np.random.randn(128,784) * 0.01
    B1 = np.zeros((128,1))
    W2 = np.random.randn(10,128)*0.01
    B2 = np.zeros((10,1))
    return W1, B1, W2, B2

def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1 @ X + B1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def right_one(y_true):
    Y = np.zeros((10,1))
    Y[y_true] = 1
    return Y

def relu_derivative(z):
    return (z>0).astype(float)


def backward_prop(W1,B1,W2,B2,Z1,A1,Z2,A2,X,y_true):
    Y = right_one(y_true)

    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T
    dB2 = dZ2

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = dZ1 @ X.T
    dB1 = dZ1

    return dW1, dB1, dW2, dB2




def cross_entropy_loss(A2, y_true):
    L = -np.log(A2[y_true].item() + 1e-9)
    return L



def update_params(W1,B1,W2,B2,dW1,dB1,dW2,dB2, alpha):
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    return W1, B1, W2, B2


if __name__ == "__main__":
    example1 = np.array([-2, -1, 0, 1, 2])
    print(relu(example1))


    example2 = np.array([1,2,3])
    print(softmax(example2))
    print(np.sum(softmax(example2)))

    

