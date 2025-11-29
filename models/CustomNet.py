import torch
from torch import nn

# Custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pooling1 = nn.AdaptiveAvgPool2d((1,1))
        self.flatten1 = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Forward pass
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.pooling1(x)
        x = self.flatten1(x)
        x = self.fc1(x)

        return x