import tqdm
import os, sys
from pathlib import Path
import torch

print("importing local...")
from config import DATASET_DIR, DEVICE
from dataset import TestMIT5KDataset, ValMIT5KDataset, norm_img
from ptcolor import rgb2lab, squared_deltaE94

from competitors.inference import (
   # adaint,
    clutnet, curl, ia3dlut, #ltmnet,
    maxim, neural_spline_enhancement)

competitors = {
    # 'adaint': adaint,
    'clutnet': clutnet,
    'curl': curl,
    'ia3dlut': ia3dlut,
    # 'ltmnet': ltmnet,
    'maxim': maxim,
    'nse': neural_spline_enhancement
}


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def generate_predictions(
        demo_fn,
        dataset,
        dstdir=Path("predictions"),
    ):
    dstdir.mkdir(exist_ok=True)  # create dest dir
    for val_img, val_img_path, _, _ in tqdm.tqdm(dataset):
        dst_img_path = dstdir / val_img_path.name
        if dst_img_path.is_file():
            continue
        # print(val_img_path)
        # print(dst_img_path)
        with HiddenPrints():
            demo_fn(val_img_path, dst_img_path)
        


if __name__ == '__main__':
    # dataset = ValMIT5KDataset(datadir=DATASET_DIR)
    dataset = TestMIT5KDataset(datadir=DATASET_DIR)
    assert len(dataset) > 0, "dataset is empty"

    def de76(rgb1, rgb2):
        return torch.norm(rgb2lab(rgb1) - rgb2lab(rgb2), dim=1).mean()

    def de94(rgb1, rgb2):
        return squared_deltaE94(rgb2lab(rgb1), rgb2lab(rgb2)).mean()


    for name, method in competitors.items():
        rundir = Path(f'predictions/{name}')
        generate_predictions(
            method.demo,
            dataset,
            dstdir=rundir,
        )
    # loss_fns = {"de76": de76, "de94": de94, "mse": torch.nn.MSELoss()}
    # preds_dir = Path('aaa')
    # targets_dir = Path(DATASET_DIR) / "train" / "target"
    # evaluate_predictions(preds_dir, targets_dir, loss_fns=loss_fns)
    # # train_dir = Path(DATASET_DIR) / "train"
    # # validate_identity(train_dir, loss_fns=loss_fns)
