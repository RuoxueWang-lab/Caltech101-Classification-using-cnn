import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3) # the input_channels of the first conv1 layer must match with the #channels of our input images
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2)
        self.linear1 = nn.Linear(32 * 55 * 55, 128)
        self.linear2 = nn.Linear(128, 101)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    
    
    