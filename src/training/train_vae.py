import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.vae import VAE
from src.config import DEVICE, SEQ_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/generated_midis", exist_ok=True)

# Load dataset
data_path = "data/processed/synthetic_music_data.npy"
if not os.path.exists(data_path):
    synthetic = np.random.randint(0, 128, size=(800, SEQ_LENGTH), dtype=np.int64)
    np.save(data_path, synthetic)

data = np.load(data_path)
data = torch.tensor(data, dtype=torch.long).to(DEVICE)

def one_hot(x, num_classes=128):
    return torch.nn.functional.one_hot(x, num_classes).float()

dataset = one_hot(data)

# Model
model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.01 * kl_loss  # β = 0.01

losses_recon = []
losses_kl = []
losses_total = []

print("Training Task 2: VAE Multi-Genre Generator...")
for epoch in range(EPOCHS):
    epoch_recon = 0
    epoch_kl = 0
    epoch_total = 0
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i:i+BATCH_SIZE]
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        
        epoch_total += loss.item()
        epoch_recon += nn.MSELoss()(recon, batch).item()
        epoch_kl += (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())).item()

    batches = max(1, (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE)
    avg_total = epoch_total / batches
    avg_recon = epoch_recon / batches
    avg_kl = epoch_kl / batches
    
    losses_recon.append(avg_recon)
    losses_kl.append(avg_kl)
    losses_total.append(avg_total)
    print(f"Epoch {epoch+1}/{EPOCHS} - Total Loss: {avg_total:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

# Save loss curves
plt.figure(figsize=(10,5))
plt.plot(losses_recon, label="Reconstruction Loss")
plt.plot(losses_kl, label="KL Loss")
plt.plot(losses_total, label="Total VAE Loss")
plt.title("Task 2: VAE Multi-Genre Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("outputs/plots/task2_vae_losses.png")
plt.close()

# Save model
torch.save(model.state_dict(), "models/vae.pth")
print("Task 2 training completed! Loss curves saved.")

# Generate 8 multi-genre MIDI samples (Task 2 deliverable)
model.eval()
with torch.no_grad():
    for i in range(8):
        z = torch.randn(1, 32).to(DEVICE)  # sample from latent space
        fake = model.decode(z)  # using decode method
        pitches = torch.argmax(fake.squeeze(0), dim=1).cpu().numpy()
        from midiutil import MIDIFile
        mf = MIDIFile(1)
        mf.addTempo(0, 0, 120)
        for t, p in enumerate(pitches[:32]):
            mf.addNote(0, 0, int(p), t*0.25, 0.5, 100)
        with open(f"outputs/generated_midis/task2_sample_{i}.mid", "wb") as f:
            mf.writeFile(f)

print("✅ 8 MIDI samples generated for Task 2 (Multi-Genre)")
