import neural_net as nn
import numpy as np

weights = np.load("trained_weight.npz")
W1 = weights['W1']
W2 = weights['W2']
B1 = weights['B1']
B2 = weights['B2']

test_data = np.loadtxt('mnist_test.csv', dtype=int, delimiter=',')
test_labels = test_data[:,0]
test_pixels = test_data[:,1:]


test_correct = 0
n_test = test_pixels.shape[0]

for i in range(n_test):
    X = test_pixels[i, :].reshape(784,1)
    y_true = test_labels[i]

    Z1, A1, Z2, A2 = nn.forward_prop(W1, B1, W2, B2, X)

    if A2.argmax() == y_true:
        test_correct +=1

test_accuracy = test_correct/n_test
print(f'=== Accuracy: {test_accuracy:.4f} ===')

