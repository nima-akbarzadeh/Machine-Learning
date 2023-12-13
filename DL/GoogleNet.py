import torch
from torch import nn


# A simple concolutional layer used in multiple steps
class Conv_block(nn.Module):
    def __init__(self, inp_channels, out_channels, **kwargs):
        super(Conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inp_channels, out_channels, **kwargs)
        self.bnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.bnorm(self.conv(x)))


# The idea is to ensemble all different types of image filters together
class Incp_block(nn.Module):
    def __init__(
            self, inp_channels, out_1by1, inp_3by3, out_3by3, inp_5by5, out_5by5, out_1by1pool
    ):
        super(Incp_block, self).__init__()
        self.sub_block1 = Conv_block(inp_channels, out_1by1, kernel_size=(1, 1))

        self.sub_block2 = nn.Sequential(
            Conv_block(inp_channels, inp_3by3, kernel_size=(1, 1)),
            Conv_block(inp_3by3, out_3by3, kernel_size=(3, 3), padding=1),
        )

        self.sub_block3 = nn.Sequential(
            Conv_block(inp_channels, inp_5by5, kernel_size=(1, 1)),
            Conv_block(inp_5by5, out_5by5, kernel_size=(5, 5), padding=2),
        )

        self.sub_block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            Conv_block(inp_channels, out_1by1pool, kernel_size=(1, 1)),
        )

    def forward(self, x):
        # The dimension of the input is: num_images * num_filters * image_shap0 * image_shape1
        # The output of Incp_block are filters which should be concatenated in the first dimension
        return torch.cat([self.sub_block1(x), self.sub_block2(x), self.sub_block3(x), self.sub_block4(x)], 1)


# According to the paper, this part helps discrimination in the lower stages in the classifier,
# increase the gradient signal that gets propagated back, and provide additional regularization.
class InceptionAux(nn.Module):
    def __init__(self, inp_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        self.conv = Conv_block(inp_channels, 128, kernel_size=(1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        # Keep the number of batches in dimension 0 and flatten the rest
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# The main architecture
class GoogLeNet(nn.Module):
    def __init__(self, num_img_channels=3, num_classes=1000, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = Conv_block(inp_channels=num_img_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv2 = Conv_block(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.incep3a = Incp_block(192, 64, 96, 128, 16, 32, 32)
        self.incep3b = Incp_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.incep4a = Incp_block(480, 192, 96, 208, 16, 48, 64)
        self.incep4b = Incp_block(512, 160, 112, 224, 24, 64, 64)
        self.incep4c = Incp_block(512, 128, 128, 256, 24, 64, 64)
        self.incep4d = Incp_block(512, 112, 144, 288, 32, 64, 64)
        self.incep4e = Incp_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.incep5a = Incp_block(832, 256, 160, 320, 32, 128, 128)
        self.incep5b = Incp_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

        self.training_phase = True

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.incep3a(x)
        x = self.incep3b(x)
        x = self.maxpool3(x)
        x = self.incep4a(x)

        # Auxiliary Softmax classifier 1
        aux_out1 = None
        if self.aux_logits and self.training_phase:
            aux_out1 = self.aux1(x)

        x = self.incep4b(x)
        x = self.incep4c(x)
        x = self.incep4d(x)

        # Auxiliary Softmax classifier 2
        aux_out2 = None
        if self.aux_logits and self.training_phase:
            aux_out2 = self.aux2(x)

        x = self.incep4e(x)
        x = self.maxpool4(x)
        x = self.incep5a(x)
        x = self.incep5b(x)
        x = self.avgpool(x)

        # Keep the number of batches in dimension 0 and flatten the rest
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux_out1, aux_out2, x
        else:
            return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    batch_size = 5
    image_channels = 3
    image_size = [224, 224]
    num_classes = 1000

    # Model
    model = GoogLeNet(num_img_channels=image_channels, num_classes=num_classes, aux_logits=True).to(device)

    # Output
    x = torch.randn(batch_size, image_channels, image_size[0], image_size[1]).to(device)
    prediction = model(x)
    print(f"The output shape of each channel must be {batch_size} by {num_classes}")
    [print(f"The output shape of every auxiliary and final outputs is {prediction[i].shape}") for i in range(len(prediction))]
