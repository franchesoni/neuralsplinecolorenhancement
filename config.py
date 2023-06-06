import torch

DATASET_DIR = '/home/maxdunitz/Desktop/mlbriefs2/workdir/neural_spline_enhancement/C'#dataset/C/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'