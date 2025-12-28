from torchvision import models
import torch.nn as nn

def build_resnet18(num_classes: int = 100):
    """
    ResNet-18 with ImageNet weights; replaces final FC for `num_classes`.
    """
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # pretrained
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

if __name__ == "__main__":
    import torch
    model = build_resnet18(100)
    x = torch.randn(2, 3, 32, 32)  # CIFAR-100 size
    logits = model(x)
    print("logits shape:", logits.shape)  # expect [2, 100]