import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
import numpy as np

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


class DenseNetModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()

        base = densenet121(weights=DenseNet121_Weights.DEFAULT)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.features(x))
        x = self.pool(x).flatten(1)
        return self.classifier(x)

class_names = [
    "Normal",
    "Tuberculosis"
]

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

trained_model = None

def load_model():

    global trained_model

    if trained_model is None:

        trained_model = DenseNetModel()

        trained_model.load_state_dict(
            torch.load(
                "model/best_DenseNet121.pth",
                map_location="cpu"
            )
        )

        trained_model.eval()

    return trained_model

def predict(image):
    model = load_model()

    tensor = val_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)

        probs = torch.softmax(outputs, dim=1)[0]

        confidence, pred = torch.max(probs, 0)

    return (
        class_names[pred.item()],
        confidence.item() * 100,
        probs
    )

    # model = load_model()
    #
    # tensor = val_transform(
    #     image.convert("RGB")
    # ).unsqueeze(0)
    #
    # with torch.no_grad():
    #
    #     outputs = model(tensor)
    #
    #     probs = torch.softmax(
    #         outputs,
    #         dim=1
    #     )
    #
    #     confidence, pred = torch.max(
    #         probs,
    #         1
    #     )
    #
    # prediction = class_names[
    #     pred.item()
    # ]
    #
    # confidence = (
    #     confidence.item() * 100
    # )
    #
    # return prediction, confidence

def generate_heatmap(image):

    model = load_model()

    target_layers = [
        model.features.denseblock4
    ]

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers
    )

    resized = image.resize(
        (224,224)
    )

    img_np = (
        np.array(resized)
        .astype(np.float32)/255
    )

    tensor = val_transform(
        resized
    ).unsqueeze(0)

    grayscale_cam = cam(
        input_tensor=tensor
    )[0]

    heatmap = show_cam_on_image(
        img_np,
        grayscale_cam,
        use_rgb=True
    )

    return heatmap