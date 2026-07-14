import neural_net as nn
import numpy as np
import matplotlib.pyplot as plt


weights = np.load("trained_weight.npz")
W1 = weights['W1']
W2 = weights['W2']
B1 = weights['B1']
B2 = weights['B2']

test_data = np.loadtxt('mnist_test.csv', dtype=int, delimiter=',')
test_labels = test_data[:,0]
test_pixels = test_data[:,1:]

errors = []
n_test = test_pixels.shape[0]

for i in range(n_test):
    X = test_pixels[i, :].reshape(784,1)
    y_true = test_labels[i]

    Z1, A1, Z2, A2 = nn.forward_prop(W1, B1, W2, B2, X)
    prediction = A2.argmax()

    if y_true != prediction:
        errors.append((test_pixels[i,:], prediction, y_true))
    
    if len(errors) >= 16:
        break

fig, axes = plt.subplots(4,4, figsize=(8,8))

for idx, ax in enumerate(axes.flat):
    pixels, prediction, y_true = errors[idx]
    image = pixels.reshape(28,28)
    ax.imshow(image, cmap='gray')
    ax.set_title(f'pred: {prediction}, true: {y_true}')
    ax.axis('off')

plt.tight_layout()
plt.show()

