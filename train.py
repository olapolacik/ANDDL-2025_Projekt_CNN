import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

from model import CNN

# =========================
# 1. Dane
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Tworzymy folder na błędne predykcje
os.makedirs("results/wrong_preds", exist_ok=True)

classes = train_data.classes

# =========================
# 2. Konfiguracje eksperymentów
# =========================
experiments = [
    {'activation': 'relu', 'lr': 0.001, 'conv_layers': 2, 'name': 'ReLU_lr0.001_2Conv'},
    {'activation': 'relu', 'lr': 0.0001, 'conv_layers': 2, 'name': 'ReLU_lr0.0001_2Conv'},
    {'activation': 'leakyrelu', 'lr': 0.001, 'conv_layers': 2, 'name': 'LeakyReLU_lr0.001_2Conv'},
    {'activation': 'relu', 'lr': 0.001, 'conv_layers': 1, 'name': 'ReLU_lr0.001_1Conv'}
]

results = []

# =========================
# 3. Pętla po eksperymentach
# =========================
for exp in experiments:
    print(f"\n=== Eksperyment: {exp['name']} ===")

    # Model
    model = CNN(activation=exp['activation'], conv_layers=exp['conv_layers'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=exp['lr'])

    # TensorBoard
    writer = SummaryWriter(f"results/runs/{exp['name']}")

    # Trening
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = correct / total
        print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}, Accuracy: {acc:.4f}")
        writer.add_scalar('Loss/train', running_loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)

    # Test
    model.eval()
    correct = 0
    total = 0
    wrong_images = []
    wrong_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Zbieranie błędnych predykcji
            for i in range(len(labels)):
                if predicted[i] != labels[i] and len(wrong_images) < 5:
                    wrong_images.append(images[i])
                    wrong_labels.append(labels[i])
                    pred_labels.append(predicted[i])

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    results.append((exp['name'], test_acc))

    # =========================
    # 4. Wyświetlanie i zapis błędnych predykcji
    # =========================
    for i, img in enumerate(wrong_images):
        plt.imshow(img.squeeze().numpy(), cmap='gray')
        plt.title(f'Prawda: {classes[wrong_labels[i]]}, Pred: {classes[pred_labels[i]]}')
        plt.axis('off')
        plt.savefig(f'results/wrong_preds/{exp["name"]}_wrong_{i+1}.png')
        plt.close()

    writer.close()

# =========================
# 5. Zapis wyników do pliku
# =========================
with open("results/summary.txt", "w") as f:
    for name, acc in results:
        f.write(f"{name}: Test Accuracy = {acc:.4f}\n")

print("\nWszystkie eksperymenty zakończone. Wyniki zapisane w results/summary.txt")
print("Błędne predykcje zapisane w folderze results/wrong_preds/")
