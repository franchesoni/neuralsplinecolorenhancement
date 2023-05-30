import torch

DATASET_DIR = 'dataset/C/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'