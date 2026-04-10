import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH = 64
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
