import os
import shutil
import random
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def split_data(base_dir, output_dir, train_split=0.8, val_split=0.1, seed=42):
    random.seed(seed)
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for cls in classes:
        files = list(Path(os.path.join(base_dir, cls)).glob('*.jpg'))
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split) + n_train

        splits = {
            'train_data': files[:n_train],
            'val_data': files[n_train:n_val],
            'test_data': files[n_val:]
        }

        for split, file_list in splits.items():
            dest_dir = Path(output_dir, split, cls)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file in file_list:
                try:
                    img = Image.open(file).convert("RGB")
                except Exception as e:
                    print(f"Skipping corrupted image {file}: {e}")
                file_size = os.path.getsize(file)
                if file_size > 0:
                    shutil.copy(file, dest_dir)

        print("Dataset split completed.")


split_data('dataset/original', 'dataset/tmp')

learning_rate = 1e-3
batch_size = 256
epochs = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_path = "dataset/tmp/train_data"
val_path = "dataset/tmp/val_data"
test_path = "dataset/tmp/test_data"

train_dataset = ImageFolder(root=train_path, transform=transform)
val_dataset = ImageFolder(root=val_path, transform=transform)
test_dataset = ImageFolder(root=test_path, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6, pin_memory=True, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=6)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=6)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        logits = self.classifier(x)
        return logits


def train_loop(model, dataloader, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val_loop(model, dataloader, loss_fn, early_stopping):
    model.eval()
    total_loss = 0
    correct = 0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_loss /= num_batches
        correct /= size
        early_stopping(total_loss)
        print(f"Validation Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {total_loss:>8f}")


def test_loop(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {total_loss:>8f}")


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

early_stopping = EarlyStopping(patience=3, min_delta=0.01)
for i in range(epochs):
    print(f'Epoch {i}:')
    train_loop(model, train_dataloader, loss_fn, optimizer)
    val_loop(model, val_dataloader, loss_fn, early_stopping)
test_loop(model, test_dataloader, loss_fn)

torch.save(model.state_dict(), f'model_weights.pth')

