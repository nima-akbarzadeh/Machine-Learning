import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from ..classifier_FCN import NeuralNet


# 1. Load model
def model_preparation(path):
    input_size = 784  # 28x28
    hidden_size = 500
    num_classes = 10
    model = NeuralNet(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()


# 2. Image -> Tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


# 3. Predict
def get_prediction(image_tensor, model):
    images = image_tensor.reshape(-1, 28 * 28)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
