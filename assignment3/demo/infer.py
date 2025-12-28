import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from assignment3.src.model import build_resnet18

def main():
    model = build_resnet18(num_classes=100)
    model.load_state_dict(torch.load(
        "assignment3/results/runs/latest/global_last.pth",
        map_location="cpu"
    ))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])

    dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            out = model(x)
            pred = out.argmax(1).item()
        print(f"Sample {i}: true={y.item()} pred={pred}")
        if i == 4:
            break

if __name__ == "__main__":
    main()