import torch
import torch.nn as nn
import numpy as np
import json

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Train briefly on MNIST
import torchvision, torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

model = TinyMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training for 3 epochs...")
for epoch in range(3):
    for i, (imgs, labels) in enumerate(loader):
        optimizer.zero_grad()
        out = model(imgs.view(-1, 784))
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

# Export weights to JSON
weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().numpy().tolist()

with open('models/weights.json', 'w') as f:
    json.dump(weights, f)

print("Weights exported to models/weights.json!")
