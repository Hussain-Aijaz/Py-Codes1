import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Load CIFAR-10
dataset = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True
)

img, label = dataset[5]

# Define transforms
transformations = {
    "Resize": transforms.Resize((64, 64)),
    "Normalization": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "Random Rotation": transforms.RandomRotation(30),
    "Horizontal Flip": transforms.RandomHorizontalFlip(p=1.0),
    "Color Jitter": transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5
    )
}

# Plot results
fig, axes = plt.subplots(1, 6, figsize=(18, 4))
axes[0].imshow(img)
axes[0].set_title("Original")
axes[0].axis("off")

i = 1
for name, transform in transformations.items():
    transformed = transform(img)
    if torch.is_tensor(transformed):
        transformed = transformed.permute(1,2,0)
    axes[i].imshow(transformed)
    axes[i].set_title(name)
    axes[i].axis("off")
    i += 1

plt.tight_layout()
plt.show()
