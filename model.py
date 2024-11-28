import torch
import torch.nn as nn
import torch.optim as optim

class MultiTaskCNN(nn.Module):
    def __init__(self):
        super(MultiTaskCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, padding=4)  
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 48, kernel_size=9, padding=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(48, 96, kernel_size=7, padding=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(96, 192, kernel_size=5, padding=2)
        
        self.conv5 = nn.Conv2d(192, 192, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=5, padding=2)

        # Global average pooling to reduce spatial dimensions to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # First classifier head - 10 outputs
        self.fc_head1 = nn.Linear(192, 10)

        # Second classifier head - 16 outputs
        self.fc_head2 = nn.Linear(192, 16)

    def forward(self, x):
        # Feature extraction
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        # Multi-task classification heads
        out1 = self.fc_head1(x)
        out2 = self.fc_head2(x)

        return out1, out2

def get_model_and_optimizer():
    # Create the model
    model = MultiTaskCNN()

    # Define the loss functions
    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    return model, loss_fn1, loss_fn2, optimizer

