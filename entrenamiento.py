import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from modelo import CNN
import pandas as pd

def cargar_datos(csv_file):
    data = pd.read_csv(csv_file, header=None).values
    X = data[:, :-1].astype('float32').reshape(-1, 1, 28, 28)
    y = data[:, -1].astype('int64')

    X = torch.tensor(X)
    y = torch.tensor(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def entrenar_rna(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    patience = 5
    trigger_times = 0

    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # Entrenamiento
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == y_batch).sum().item()
            total_predictions += y_batch.size(0)

        accuracy = 100 * correct_predictions / total_predictions
        print(f"Época {epoch+1}  Pérdida: {epoch_loss:.4f}, Precisión: {accuracy:.2f}%")

        # Validación
        val_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_val).sum().item()
                total += y_val.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Validación - Pérdida: {val_loss:.4f}, Precisión: {val_accuracy:.2f}%")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "modelo_mejorado.pth")
            print("Modelo mejorado guardado.")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping activado.")
                break

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = cargar_datos("figuras.csv")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    model = CNN().to(device)
    entrenar_rna(model, train_loader, val_loader, epochs=50, lr=0.001, device=device)
