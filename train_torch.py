import torch 
import torch.nn as nn
import numpy as np

data = np.loadtxt('mnist_train.csv', dtype=int, delimiter=',')
labels = data[:,0]
pixels = data[:, 1:] /255.0

X = torch.tensor(pixels, dtype=torch.float32)
Y = torch.tensor(labels, dtype=torch.long)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784,128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = NeuralNet()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.1)

epochs = 10000


for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model.forward(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()

    preds = pred.argmax(dim=1)
    if epoch % 1000 == 0:
        accuracy = (preds == Y).float().mean()
        
        print(f'---epoch: {epoch}, accuracy: {accuracy}, loss: {loss}---')

            
print(f'---total accuracy: {accuracy}, loss: {loss}---')

torch.save(model.state_dict(), 'model.pth')