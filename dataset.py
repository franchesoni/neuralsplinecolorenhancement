from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

def norm_img(x):
    return (x - x.min()) / (x.max() - x.min())

class TrainMIT5KDataset(Dataset):
    def __init__(self, datadir):
        self.datadir = Path(datadir)
        self.image_paths = self.datadir / 'train_processed' / 'raw'
        self.target_paths = self.datadir / 'train_processed' / 'target'
        self.image_paths = sorted(self.image_paths.glob('*.jpg'))
        self.target_paths = sorted(self.target_paths.glob('*.jpg'))
        assert [ipath.name for ipath in self.image_paths] == [tpath.name for tpath in self.target_paths]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target_path = self.target_paths[index]
        image = norm_img(to_tensor(Image.open(image_path)))
        target = to_tensor(Image.open(target_path)) / 255.0
        return image, target

    def __len__(self):
        return len(self.image_paths)
