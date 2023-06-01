from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def norm_img(x):
    return (x - x.min()) / (x.max() - x.min())


class TrainMIT5KDataset(Dataset):
    def __init__(self, datadir):
        self.datadir = Path(datadir)
        image_paths = self.datadir / "train_processed" / "raw"
        target_paths = self.datadir / "train_processed" / "target"
        self.image_paths = sorted(image_paths.glob("*.jpg"))
        self.target_paths = sorted(target_paths.glob("*.jpg"))
        assert [ipath.name for ipath in self.image_paths] == [
            tpath.name for tpath in self.target_paths
        ]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target_path = self.target_paths[index]
        image = norm_img(to_tensor(Image.open(image_path)))
        target = to_tensor(Image.open(target_path))
        return image, target

    def __len__(self):
        return len(self.image_paths)


class ValMIT5KDataset(Dataset):
    def __init__(self, datadir):
        self.datadir = Path(datadir)
        with open(self.datadir / "val_images.txt") as f:
            val_images = f.read().splitlines()
        self.image_paths = sorted([self.datadir / "train" / "raw" / valimgname for valimgname in val_images])
        self.target_paths = sorted([self.datadir / "train" / "target" / valimgname for valimgname in val_images])
        assert [ipath.name for ipath in self.image_paths] == [
            tpath.name for tpath in self.target_paths
        ]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target_path = self.target_paths[index]
        image = Image.open(image_path)
        target = Image.open(target_path)
        return image, image_path, target, target_path

    def __len__(self):
        return len(self.image_paths)
