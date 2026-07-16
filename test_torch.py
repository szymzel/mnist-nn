import torch
import torch.nn as nn
import numpy as np

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
model.load_state_dict(torch.load('model.pth'))
model.eval()

test_Data = np.loadtxt('mnist_test.csv', dtype=int, delimiter=',')
test_labels = test_Data[:,0]
test_pixels = test_Data[:, 1:]/255.0

X_test = torch.tensor(test_pixels, dtype=torch.float32)
Y_test = torch.tensor(test_labels, dtype=torch.long)

with torch.no_grad():
    test_pred = model(X_test)
    test_predictions = test_pred.argmax(dim=1)
    test_accuracy = (test_predictions == Y_test).float().mean()

print(f'test accuracy: {test_accuracy.item():.4f}')