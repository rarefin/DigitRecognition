import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel, pool_kernel, pool_stride):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, padding=conv_kernel//2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_kernel//2),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.block(x)

        return x


class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()

        self.features = nn.Sequential(
            Block(in_channels=3, out_channels=48, conv_kernel=5, pool_kernel=2, pool_stride=2),
            Block(in_channels=48, out_channels=64, conv_kernel=5, pool_kernel=2, pool_stride=1),
            Block(in_channels=64, out_channels=128, conv_kernel=5, pool_kernel=2, pool_stride=2),
            Block(in_channels=128, out_channels=160, conv_kernel=5, pool_kernel=2, pool_stride=1),
            Block(in_channels=160, out_channels=192, conv_kernel=5, pool_kernel=2, pool_stride=2)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(192 * 9 * 9, 3072),
            nn.PReLU(),
            nn.Dropout(0.5),
        )

        self.digit_length = nn.Sequential(nn.Linear(3072, 7))
        self.digit1 = nn.Sequential(nn.Linear(3072, 11))
        self.digit2 = nn.Sequential(nn.Linear(3072, 11))
        self.digit3 = nn.Sequential(nn.Linear(3072, 11))
        self.digit4 = nn.Sequential(nn.Linear(3072, 11))
        self.digit5 = nn.Sequential(nn.Linear(3072, 11))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 192 * 9 * 9)
        x = self.fully_connected(x)

        length_logits = self.digit_length(x)
        digit1_logits = self.digit1(x)
        digit2_logits = self.digit2(x)
        digit3_logits = self.digit3(x)
        digit4_logits = self.digit4(x)
        digit5_logits = self.digit5(x)

        return length_logits, [digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits]
