from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # ((64 - 3 + 2*1) / 1) + 1 = 64
            nn.LeakyReLU(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # ((64 - 3 + 2*1) / 1) + 1 = 64
            nn.LeakyReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), # ((64 - 2) / 2) + 1 = 32

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # ((32 - 3 + 2*1) / 1) + 1 = 32
            nn.LeakyReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # ((32 - 3 + 2*1) / 1) + 1 = 32
            nn.LeakyReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), # ((32 - 2) / 2) + 1 = 16

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # ((16 - 3 + 2*1) / 1) + 1 = 16
            nn.LeakyReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2) # ((16 - 2) / 2) + 1 = 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), # Entrada de (8 x 8 x 128)
            nn.LeakyReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        cnn = self.cnn(x)
        classifier = self.classifier(cnn)
        return classifier