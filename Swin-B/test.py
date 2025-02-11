import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import SwinForImageClassification


transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]  # ImageNet std
    ),
])
test_dir = "Dataset/test"

test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-base-patch4-window7-224-in22k",
    ignore_mismatched_sizes=True,
    num_labels=3
).to(device)

best_model_path = "best_swin_b_3class_5e_4.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

print(f"[INFO] Successfully loaded best model weights from '{best_model_path}'.")

@torch.no_grad()
def test_model(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        logits = outputs.logits

        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    test_loss = running_loss / total
    test_acc = 100.0 * correct / total
    return test_loss, test_acc

test_loss, test_acc = test_model(model, test_loader, device)
print(f"[INFO] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
