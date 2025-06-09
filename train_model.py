import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn.preprocessing import label_binarize
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score

# === CONFIGURAÇÕES ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = (224, 224)
NUM_CLASSES = 5
SEED = 42

torch.manual_seed(SEED)

project_root = Path(__file__).resolve().parent
image_dir = project_root / "resized_train"
train_csv = project_root / "train_split.csv"
test_csv = project_root / "test_split.csv"

# === TRANSFORMS ===
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === DATASET PERSONALIZADO ===
class RetinaDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df['filename'] = self.df['image'] + ".jpeg"
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.df.iloc[idx]['filename']
        image = Image.open(img_path).convert('RGB')
        label = int(self.df.iloc[idx]['level'])
        if self.transform:
            image = self.transform(image)
        return image, label

# === MODELO CNN DO ZERO ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))   # (B, 128, 28, 28)
        x = x.view(-1, 128 * 28 * 28)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# === FUNÇÃO DE TREINAMENTO ===
def train_model(model, train_loader, val_loader, optimizer, criterion):
    model.train()
    train_losses, val_losses, accs = [], [], []

    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        # Validação
        model.eval()
        correct, total = 0, 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        accs.append(correct / total)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {accs[-1]:.4f}")

    return train_losses, val_losses, accs

# === PREPARAÇÃO DOS DADOS ===
train_dataset = RetinaDataset(train_csv, image_dir, transform=train_transform)
test_dataset = RetinaDataset(test_csv, image_dir, transform=test_transform)

# Split treino/validação
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === TREINAMENTO ===
model = SimpleCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print("\n🔧 Iniciando treinamento...\n")
train_losses, val_losses, accs = train_model(model, train_loader, val_loader, optimizer, criterion)

# === AVALIAÇÃO ===
model.eval()
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_probs.extend(probs)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)
y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

# === MÉTRICAS ===
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("📈 F1-score (macro):", multiclass_f1_score(torch.tensor(y_pred), torch.tensor(y_true), num_classes=NUM_CLASSES, average='macro').item())
print("📈 MSE:", mean_squared_error(y_true, y_pred))
print("📈 ROC AUC:", roc_auc_score(y_true_bin, y_probs, multi_class='ovr'))

# === GRÁFICOS ===
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss por Época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(accs, label="Val Accuracy")
plt.title("Validação - Acurácia por Época")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend()
plt.show()
