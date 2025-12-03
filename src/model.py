import torch.nn as nn
from torchvision import models


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Replace the final fully-connected layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained)