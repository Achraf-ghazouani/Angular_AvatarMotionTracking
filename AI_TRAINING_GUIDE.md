# Guide d'Entraînement du Modèle IA

Ce guide explique comment entraîner et exporter un modèle LSTM PyTorch pour la correction des mouvements.

## 📋 Prérequis

```bash
pip install torch torchvision numpy pandas scikit-learn onnx
```

## 📊 1. Collecte des Données

### Structure des Données

Collecter des séquences de tracking de qualité :

```python
import numpy as np
import json

# Format des données
# Chaque frame contient 12 features
data_format = {
    "head_rotation": [x, y, z],      # Rotation de la tête (degrés)
    "hips_rotation": [x, y, z],      # Rotation des hanches
    "left_arm": [x, y, z],           # Position bras gauche
    "right_arm": [x, y, z]           # Position bras droit
}

# Exemple de collecte
tracking_sequences = []

# TODO: Intégrer avec l'application Angular pour collecter
# des sessions de tracking réelles
```

### Script de Collecte

```python
# collect_data.py
import json
from datetime import datetime

class DataCollector:
    def __init__(self, output_file="training_data.json"):
        self.sequences = []
        self.current_sequence = []
        self.output_file = output_file
    
    def add_frame(self, head_rot, hips_rot, left_arm, right_arm):
        frame = {
            "timestamp": datetime.now().isoformat(),
            "head": head_rot,
            "hips": hips_rot,
            "left_arm": left_arm,
            "right_arm": right_arm
        }
        self.current_sequence.append(frame)
    
    def save_sequence(self):
        if len(self.current_sequence) > 0:
            self.sequences.append(self.current_sequence)
            self.current_sequence = []
            self.save_to_file()
    
    def save_to_file(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.sequences, f, indent=2)
        print(f"Saved {len(self.sequences)} sequences")

# Usage
collector = DataCollector()
# Collecter pendant plusieurs sessions
# collector.add_frame(...)
# collector.save_sequence()
```

## 🧠 2. Modèle LSTM

### Architecture du Modèle

```python
# model.py
import torch
import torch.nn as nn

class MotionCorrectionLSTM(nn.Module):
    """
    Modèle LSTM pour corriger et prédire les mouvements
    
    Input: [batch_size, sequence_length, 12]
    Output: [batch_size, 12]
    """
    
    def __init__(self, 
                 input_size=12, 
                 hidden_size=64, 
                 num_layers=2,
                 dropout=0.2):
        super(MotionCorrectionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (optionnel)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, input_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch, sequence, features]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention (optionnel)
        # attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Prendre la dernière sortie
        last_output = lstm_out[:, -1, :]
        
        # Fully connected
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# Test du modèle
if __name__ == "__main__":
    model = MotionCorrectionLSTM()
    dummy_input = torch.randn(32, 10, 12)  # batch=32, seq=10, features=12
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

## 🏋️ 3. Entraînement

### Dataset Personnalisé

```python
# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

class MotionTrackingDataset(Dataset):
    def __init__(self, json_file, sequence_length=10):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.sequence_length = sequence_length
        self.sequences = self._prepare_sequences()
    
    def _prepare_sequences(self):
        sequences = []
        
        for session in self.data:
            if len(session) < self.sequence_length + 1:
                continue
            
            # Convertir en features
            features = []
            for frame in session:
                feature = (
                    frame['head'] + 
                    frame['hips'] + 
                    frame['left_arm'] + 
                    frame['right_arm']
                )
                features.append(feature)
            
            # Créer des séquences glissantes
            for i in range(len(features) - self.sequence_length):
                seq_input = features[i:i+self.sequence_length]
                seq_target = features[i+self.sequence_length]
                sequences.append((seq_input, seq_target))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_input, seq_target = self.sequences[idx]
        return (
            torch.FloatTensor(seq_input),
            torch.FloatTensor(seq_target)
        )

# Test
if __name__ == "__main__":
    dataset = MotionTrackingDataset('training_data.json')
    print(f"Dataset size: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
```

### Script d'Entraînement

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import MotionCorrectionLSTM
from dataset import MotionTrackingDataset
import matplotlib.pyplot as plt

def train_model(
    data_file='training_data.json',
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"Training on {device}")
    
    # Charger le dataset
    dataset = MotionTrackingDataset(data_file)
    
    # Split train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialiser le modèle
    model = MotionCorrectionLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Historique
    train_losses = []
    val_losses = []
    
    # Entraînement
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ Saved best model at epoch {epoch+1}")
        
        # Log
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
    
    # Plot des courbes
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig('training_history.png')
    print("📊 Training history saved")
    
    return model

if __name__ == "__main__":
    model = train_model()
    print("✅ Training completed!")
```

## 📤 4. Export en ONNX

```python
# export_onnx.py
import torch
from model import MotionCorrectionLSTM

def export_to_onnx(
    model_path='best_model.pth',
    output_path='motion_correction.onnx',
    sequence_length=10
):
    # Charger le modèle
    model = MotionCorrectionLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Input factice
    dummy_input = torch.randn(1, sequence_length, 12)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model exported to {output_path}")
    
    # Vérifier l'export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")
    
    # Afficher les infos
    print("\nModel Info:")
    print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")

if __name__ == "__main__":
    export_to_onnx()
```

## 🧪 5. Test du Modèle ONNX

```python
# test_onnx.py
import onnxruntime as ort
import numpy as np

def test_onnx_model(model_path='motion_correction.onnx'):
    # Charger le modèle
    session = ort.InferenceSession(model_path)
    
    # Input de test
    test_input = np.random.randn(1, 10, 12).astype(np.float32)
    
    # Inférence
    outputs = session.run(
        None,
        {'input': test_input}
    )
    
    print("✅ ONNX inference successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output: {outputs[0]}")
    
    return outputs

if __name__ == "__main__":
    test_onnx_model()
```

## 📋 6. Pipeline Complet

```bash
# 1. Collecter les données (via l'application Angular)
# 2. Entraîner le modèle
python train.py

# 3. Exporter en ONNX
python export_onnx.py

# 4. Tester l'export
python test_onnx.py

# 5. Copier le modèle dans l'application
cp motion_correction.onnx ../src/assets/models/

# 6. Activer l'IA dans la config Angular
```

## 🎯 Conseils d'Optimisation

### Augmentation des Données

```python
def augment_sequence(sequence):
    # Ajouter du bruit
    noise = np.random.normal(0, 0.01, sequence.shape)
    
    # Variation de vitesse
    if np.random.random() > 0.5:
        indices = np.sort(np.random.choice(len(sequence), len(sequence), replace=True))
        sequence = sequence[indices]
    
    return sequence + noise
```

### Régularisation

```python
# Dans le modèle
self.dropout = nn.Dropout(0.3)

# L2 regularization dans l'optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False
```

## 📊 Métriques d'Évaluation

```python
def evaluate_model(model, test_loader, device):
    model.eval()
    
    mse_total = 0
    mae_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_x)
            
            mse = nn.MSELoss()(output, batch_y)
            mae = nn.L1Loss()(output, batch_y)
            
            mse_total += mse.item()
            mae_total += mae.item()
    
    print(f"MSE: {mse_total / len(test_loader):.6f}")
    print(f"MAE: {mae_total / len(test_loader):.6f}")
```

## ✅ Checklist

- [ ] Collecter au moins 1000 séquences de tracking
- [ ] Séparer train/val/test (70/15/15)
- [ ] Entraîner le modèle (MSE < 0.01)
- [ ] Valider sur données de test
- [ ] Exporter en ONNX
- [ ] Tester l'inférence ONNX
- [ ] Intégrer dans l'application Angular
- [ ] Valider les performances temps réel

---

**Note**: Ce guide est un point de départ. Adaptez-le selon vos besoins spécifiques.
