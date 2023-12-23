import torch
import torch.nn as nn


# The idea of ResNet is to increase the number of convolutional layers without affecting the gradients.
# The skip-connection operation allows us to do so which does not let the gradients vanish.
class SubBlock(nn.Module):
    def __init__(self, inp_channels, mid_channels, expansion, identity_downsample=None, stride=(1, 1)):
        super(SubBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, mid_channels,
                               kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bnorm1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * expansion,
                               kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bnorm3 = nn.BatchNorm2d(mid_channels * expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        # Keep the original input which is not affected by the next operations
        input_x = x.clone()

        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bnorm3(x)

        # This is required when the data flows from the first sub_block to the rest
        # The size of the input should match the size of the output
        if self.identity_downsample is not None:
            input_x = self.identity_downsample(input_x)

        # Skip-connection operation
        x += input_x
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, sub_block, num_blocks, num_img_channels, num_classes):
        super(ResNet, self).__init__()
        self.inp_channels = 64
        self.conv1 = nn.Conv2d(num_img_channels, self.inp_channels, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bnorm1 = nn.BatchNorm2d(self.inp_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.expansion = 4
        self.block1 = self._make_block(sub_block, num_blocks[0], mid_channels=64, stride=(1, 1))
        self.block2 = self._make_block(sub_block, num_blocks[1], mid_channels=128, stride=(2, 2))
        self.block3 = self._make_block(sub_block, num_blocks[2], mid_channels=256, stride=(2, 2))
        self.block4 = self._make_block(sub_block, num_blocks[3], mid_channels=512, stride=(2, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_block(self, sub_block, num_residual_blocks, mid_channels, stride):
        identity_downsample = None
        layers = []

        # Either if the input space is halved for ex, 56x56 -> 28x28 (stride=2),
        # or number of filters changes between blocks,  we need to adapt the Identity (skip connection)
        # so it will be able to be added to the layer that's ahead
        if stride != 1 or self.inp_channels != mid_channels * self.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.inp_channels, mid_channels * self.expansion,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * self.expansion),
            )

        layers.append(sub_block(self.inp_channels, mid_channels, self.expansion, identity_downsample, stride))

        # The expansion size is always 4 for ResNet 50, 101, and 152
        self.inp_channels = mid_channels * self.expansion

        # For example, first resnet layer: 256 will be mapped to 64 as mid-layer,
        # Finally it gets back to 256. Then, 256 is repeated 'num_residual_blocks - 1' times as below.
        # As 64 filters are mapped to 256 at the end of the sub_block, no identity downsample is needed.
        for i in range(num_residual_blocks - 1):
            layers.append(sub_block(self.inp_channels, mid_channels, self.expansion))

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    batch_size = 5
    image_channels = 3
    image_size = [224, 224]
    num_classes = 1000
    ResNet_set = [[50, [3, 4, 6, 3]], [101, [3, 4, 23, 3]], [152, [3, 8, 36, 3]]]
    model_index = 1

    # Model
    model = ResNet(SubBlock, num_blocks=ResNet_set[model_index][1], num_img_channels=3, num_classes=1000).to(device)

    # Output
    x = torch.randn(batch_size, image_channels, image_size[0], image_size[1]).to(device)
    prediction = model(x).to(device)
    print(f"The output shape of each channel must be {batch_size} by {num_classes}")
    print(f"The output shape is {prediction.shape}")

