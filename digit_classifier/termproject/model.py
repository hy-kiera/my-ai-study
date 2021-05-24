import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.p = 0.5 # prob of drop out
        self.channels = [24, 32, 64, 64, 96] # values of channels
        self.n_nodes = [128, 128] # number of nodes

        in_c = 3
        n_classes = 10

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, self.channels[0], kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[3], self.channels[4], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(7 * 7 * self.channels[4], self.n_nodes[0], bias=True)
        nn.init.kaiming_uniform_(self.fc1.weight) # He initialization
        
        self.layer3 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - self.p)
        )

        self.fc2 = nn.Linear(self.n_nodes[0], self.n_nodes[1], bias=True)
        nn.init.kaiming_uniform_(self.fc2.weight)

        self.layer4 = nn.Sequential(
            self.fc2,
            nn.ReLU(),
            nn.Dropout(p=1 - self.p)
        )

        self.fc3 = nn.Linear(self.n_nodes[1], n_classes, bias=True)
        nn.init.kaiming_uniform_(self.fc3.weight)

        self.layer5 = nn.Sequential(
            self.fc3
        )

        _layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        self.layers = nn.ModuleList(_layers)

    def forward(self, x):
        z = x
        for i, layer in enumerate(self.layers):
            if i == 2:
                z = z.view(z.size(0), -1) # flatten for FC
            z = layer(z)

        return z