import numpy as np
import pandas as pd
import random
import os
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
torchaudio.set_audio_backend("soundfile")
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

label_encoder = LabelEncoder()
base_path = '/scratch/rahulkmr/ald_bhashini/audio'

audio_data = []
sample_rates = []
labels = []
file_paths = []

for language in os.listdir(base_path):
    lang_folder = os.path.join(base_path, language)
    if not os.path.isdir(lang_folder):
        continue
    for filename in os.listdir(lang_folder):
        file_path = os.path.join(lang_folder, filename)
        file_paths.append(file_path)
        waveform, sr = torchaudio.load(file_path)
        audio_data.append(waveform)
        sample_rates.append(sr)
        labels.append(language)

print(f"Total files loaded: {len(audio_data)}")
print(f"Unique sample rates: {set(sample_rates)}")

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, target_sr=16000, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.target_sr = target_sr
        self.mfcc_transform = T.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sr:
            resampler = T.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
        mfcc = self.mfcc_transform(waveform)
        delta = F.compute_deltas(mfcc)
        delta_delta = F.compute_deltas(delta)
        features = torch.cat((mfcc, delta, delta_delta), dim=1)
        label = self.labels[idx]
        features = pad_or_truncate(features, max_len=500)
        features = normalize(features)
        return features, label

def pad_or_truncate(tensor, max_len):
    time_dim = tensor.shape[2]
    if time_dim < max_len:
        pad_amt = max_len - time_dim
        return f.pad(tensor, (0, pad_amt), mode='constant', value=0)
    elif time_dim > max_len:
        return tensor[:, :, :max_len]
    else:
        return tensor

def normalize(features):
    mean = features.mean()
    std = features.std()
    return (features - mean) / (std + 1e-9)

class LanguageDetector(nn.Module):
    def __init__(self, input_features=120, num_classes=4, lstm_hidden_size=128, max_time_steps=300):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        dummy_input = torch.zeros(1, input_features, max_time_steps)
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
        lstm_input_size = cnn_out.shape[1]
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = x.squeeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)

labels = label_encoder.fit_transform(labels)
dataset = AudioDataset(file_paths, labels)

file_paths_temp, file_paths_test, labels_temp, labels_test = train_test_split(
    file_paths, labels, test_size=0.1, stratify=labels, random_state=42
)
file_paths_train, file_paths_val, labels_train, labels_val = train_test_split(
    file_paths_temp, labels_temp, test_size=0.1111, stratify=labels_temp, random_state=42
)
# train:val:test = 80:10:10

train_dataset = AudioDataset(file_paths_train, labels_train)
val_dataset = AudioDataset(file_paths_val, labels_val)
test_dataset = AudioDataset(file_paths_test, labels_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = LanguageDetector(input_features=120, num_classes=4, max_time_steps=300).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            predicted = torch.argmax(outputs, dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    return acc, f1, cm, report

# Training loop
num_epochs = 60
best_val_loss = float('inf')
patience = 5
wait = 0
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
    val_loss = running_loss / len(train_loader)
    train_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss= {val_loss:.4f}, Val Acc= {val_acc:.4f}, Val F1= {val_f1:.4f}, Time= {time.time()-start_time:.2f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break

# Plot training history
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training Loss and Validation Accuracy')
plt.legend()
plt.savefig("training_history.png")

# Load best model and evaluate
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)

# Measure latency
start_eval = time.time()
test_acc, test_f1, test_cm, report = evaluate(model, test_loader, device)
end_eval = time.time()
latency = end_eval - start_eval

print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")
print("Evaluation Latency (seconds):", latency)
print("Confusion Matrix:\n", test_cm)
print("Classification Report:\n", report)

# Save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# Save classification report to file
with open("classification_report.txt", "w") as f:
    f.write(report)
