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

from src.models.transformer import MusicTransformer
from src.config import DEVICE, SEQ_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/generated_midis", exist_ok=True)

# Load dataset (raw token IDs - perfect for Transformer)
data_path = "data/processed/synthetic_music_data.npy"
if not os.path.exists(data_path):
    synthetic = np.random.randint(0, 128, size=(800, SEQ_LENGTH), dtype=np.int64)
    np.save(data_path, synthetic)

data = np.load(data_path)  # shape: (800, 64)
data = torch.tensor(data, dtype=torch.long).to(DEVICE)     # token IDs (0-127)

# Model
model = MusicTransformer(vocab_size=128).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

losses = []
perplexities = []

print("Training Task 3: Transformer for Long Coherent Sequences...")
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]                    # (batch, seq)
        optimizer.zero_grad()
        
        # Input = all except last token, Target = shifted by 1
        input_seq = batch[:, :-1]
        target = batch[:, 1:]
        
        output = model(input_seq)                       # (batch, seq-1, vocab)
        loss = criterion(output.reshape(-1, 128), target.reshape(-1))
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    batches = max(1, (len(data) + BATCH_SIZE - 1) // BATCH_SIZE)
    avg_loss = epoch_loss / batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    losses.append(avg_loss)
    perplexities.append(perplexity)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

# Save loss & perplexity plot
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(losses)
plt.title("Task 3: Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(1,2,2)
plt.plot(perplexities)
plt.title("Task 3: Perplexity")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.savefig("outputs/plots/task3_loss_perplexity.png")
plt.close()

# Save model
torch.save(model.state_dict(), "models/transformer.pth")
print("Task 3 training completed! Perplexity report saved.")

# Generate 10 long-sequence MIDI compositions (Task 3 deliverable)
model.eval()
with torch.no_grad():
    for i in range(10):
        # Start with a seed note
        generated = [60]  # middle C
        for _ in range(SEQ_LENGTH - 1):
            input_tensor = torch.tensor([generated], dtype=torch.long).to(DEVICE)
            output = model(input_tensor)
            next_token = torch.argmax(output[0, -1]).item()
            generated.append(next_token)
        
        # Export to MIDI
        from midiutil import MIDIFile
        mf = MIDIFile(1)
        mf.addTempo(0, 0, 120)
        for t, pitch in enumerate(generated):
            mf.addNote(0, 0, pitch, t * 0.25, 0.5, 100)
        with open(f"outputs/generated_midis/task3_long_{i}.mid", "wb") as f:
            mf.writeFile(f)

print("✅ 10 long coherent MIDI compositions generated for Task 3")
print("Perplexity (final):", perplexities[-1])
