from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=False):
        super().__init__()

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 500),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(500, 20),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def make_features(cfg):
    layers = []
    in_channels = 3

    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    value,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU(True))
            in_channels = value

    return nn.Sequential(*layers)


cfgs = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg(model_name="vgg16", **kwargs):
    return VGG(make_features(cfgs[model_name]), **kwargs)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = base_dir / "models" / "vgg-catvsdog.pth"
    image_path = base_dir / "images" / "test_dog.jpg"

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    # weights_only=False 只用于加载你自己训练并信任的完整模型文件
    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    classes = ("cat", "dog")

    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()

    predicted_class = classes[predicted_index]

    print("cat probability: {:.4f}".format(probabilities[0, 0].item()))
    print("dog probability: {:.4f}".format(probabilities[0, 1].item()))
    print("predicted class:", predicted_class)