import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('mnist_train.csv', dtype=int, delimiter=",")

label = data[:,0]
pixels = data[:, 1:]

random_int = np.random.randint(label.size)

number = pixels[random_int, :]
number = np.reshape(number, (28,28))

plt.imshow(number, cmap='gray')
plt.show()
print(label[random_int])