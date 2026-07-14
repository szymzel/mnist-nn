import numpy as np
import neural_net as nn


data = np.loadtxt('mnist_train.csv', dtype=int, delimiter=',')
labels = data[:,0]
pixels = data[:, 1:]

W1, B1, W2, B2 = nn.init_params()


alpha = 0.00001
epochs = 3
n_samples = pixels.shape[0]

for epoch in range(epochs):
    correct = 0
    total_loss = 0
    for i in range(n_samples):
        X = pixels[i,:].reshape(784,1)
        y_true = labels[i]

        Z1, A1, Z2, A2 = nn.forward_prop(W1, B1, W2, B2, X)
        loss = nn.cross_entropy_loss(A2, y_true)
        total_loss += loss


        if A2.argmax() == y_true:
            correct += 1
        

        dW1, dB1, dW2, dB2 = nn.backward_prop(W1,B1,W2,B2,Z1,A1,Z2,A2,X,y_true)
        W1, B1, W2, B2 = nn.update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

        if i % 10000 == 0:
            print(f"epoch {epoch+1}, sample {i}/{n_samples}, loss: {loss:.4f}")

    accuracy = correct/n_samples
    avg_loss = total_loss/n_samples   
    print(f'=== epoch: {epoch+1}, average loss: {avg_loss:.4f}, accuracy: {accuracy:.4f} ===')

np.savez('trained_weight.npz', W1=W1, B1=B1, W2=W2, B2=B2)
