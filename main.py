import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy
import seaborn as sns

from imgdataset import ImageDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {DEVICE}")

IMG_SIZE = 256
CLASSES = ["chagas", "no_chagas"]
TRAINING_DATA_DIR   = "training_datasets/"
VALIDATION_DATA_DIR = "validation_datasets/"

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(TRAINING_DATA_DIR, CLASSES, transform_train)
val_dataset   = ImageDataset(VALIDATION_DATA_DIR, CLASSES, transform_val)

loader     = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset,   batch_size=32, shuffle=False)

print(f"Training:   {len(train_dataset)} imágenes")
print(f"Validación: {len(val_dataset)} imágenes")


class ImageNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4,4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = ImageNN().to(DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

historial_train = []
historial_val   = []

best_val_loss = float('inf')
patience = 5
epochs_sin_mejora = 0

for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            pred = model(batch_X)
            val_loss += loss_fn(pred, batch_y).item()
    val_loss /= len(val_loader)

    print(f"Época {epoch+1}, Pérdida: {total_loss:.4f}, Val Pérdida: {val_loss:.4f}")
    historial_train.append(total_loss)
    historial_val.append(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'mejor_modelo.pth')
        epochs_sin_mejora = 0
        print(f"  ✓ Nuevo mejor modelo guardado (val_loss: {val_loss:.4f})")
    else:
        epochs_sin_mejora += 1
        print(f"  Sin mejora {epochs_sin_mejora}/{patience}")
        if epochs_sin_mejora >= patience:
            print(f"Early stopping en época {epoch+1}")
            break

model.load_state_dict(torch.load('mejor_modelo.pth'))
print("Modelo restaurado al mejor punto.")

all_probs  = []
all_preds  = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(DEVICE)
        output  = model(batch_X)
        probs   = torch.softmax(output, dim=1)[:, 1]
        _, preds = torch.max(output, 1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

auc = roc_auc_score(all_labels, all_probs)
print(f"AUC: {auc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.savefig('confusion_matrix.png')
plt.show()

print(classification_report(all_labels, all_preds, target_names=CLASSES))

fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Azar')
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.savefig('curva_roc.png')
plt.show()

plt.figure()
plt.plot(historial_train, label='Pérdida (Entrenamiento)')
plt.plot(historial_val,   label='Pérdida (Validación)')
plt.legend()
plt.savefig('graficos_entrenamiento.png')
plt.show()