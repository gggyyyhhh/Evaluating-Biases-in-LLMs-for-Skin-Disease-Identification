import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import SwinForImageClassification
import numpy as np
from torch.utils.data import random_split

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='min', verbose=False, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path

        if self.mode == 'min':
            self.best = np.Inf
            self.monitor_op = lambda current, best: current < best - self.min_delta
        elif self.mode == 'max':
            self.best = -np.Inf
            self.monitor_op = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError("mode must be 'min' or 'max'")

        self.counter = 0
        self.early_stop = False

    def __call__(self, current, model):
        if self.monitor_op(current, self.best):
            self.best = current
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f'New best model saved with {current:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        logits = outputs.logits

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            logits = outputs.logits

            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]  # ImageNet std
        ),
    ])

    train_dir = "Dataset/train"

    whole_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    train_ratio = 0.8
    train_len = int(len(whole_dataset) * train_ratio)
    val_len = len(whole_dataset) - train_len

    train_data, val_data = random_split(
        whole_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-patch4-window7-224-in22k",
        ignore_mismatched_sizes=True,
        num_labels=3
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=5, min_delta=0.0001, mode='min', verbose=True,
                                   save_path='best_swin_b_3class_5e_5.pth')

    num_epochs = 50

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}   | Val   Acc: {val_acc:.2f}%")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    model.load_state_dict(torch.load('best_swin_b_3class_5e_5.pth'))
