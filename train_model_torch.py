# train_model_torch.py
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "rostros_dataset")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_FILENAME = "face_classifier.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS = 35
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crear carpeta de modelos si no existe
os.makedirs(MODEL_DIR, exist_ok=True)


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, translate=(0.1, 0.1), shear=10, scale=(0.9, 1.1)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(f"No se encontró la carpeta del dataset: {DATASET_PATH}\n"
                            "Asegúrate de que DATASET_PATH apunte a la carpeta que contiene subcarpetas por clase.")


train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
val_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_transform)

num_classes = len(train_dataset.classes)
print("Clases encontradas:", train_dataset.classes)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = FaceCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scaler = GradScaler(enabled=(device.type == "cuda"))

#Entrenamiento
def validate(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc="Validating", leave=True)
        for images, labels in val_pbar:
            images = images.to(device)
            labels = labels.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            val_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc

def train(num_epochs):
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print("\n=============================")
        print(f" Epoch {epoch}/{num_epochs}")
        print("=============================\n")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc="Training", leave=True)
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            avg_loss = running_loss / total
            avg_acc = 100.0 * correct / total
            train_pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%"})

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0
        val_loss, val_acc = validate(model, val_loader, device, criterion)
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Modelo mejorado guardado en: {MODEL_PATH}")

    print("\nEntrenamiento finalizado.")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "face_classifier_final.pth"))


if __name__ == "__main__":
    print("")  
    train(EPOCHS)