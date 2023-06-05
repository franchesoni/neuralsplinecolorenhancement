from pathlib import Path
import os

import tqdm
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
import numpy as np
from skimage.metrics import structural_similarity

from config import DATASET_DIR
from ptcolor import squared_deltaE94, rgb2lab
from dataset import norm_img

def de76(rgb1, rgb2):
    return torch.norm(rgb2lab(rgb1) - rgb2lab(rgb2), dim=1).mean()

def de94(rgb1, rgb2):
    return squared_deltaE94(rgb2lab(rgb1), rgb2lab(rgb2)).mean()

def psnr(rgb1, rgb2):
    mse = torch.nn.MSELoss()(rgb1, rgb2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def ssim(rgb1, rgb2):
    """Structural similarity index metric"""
    rgb1 = rgb1[0].permute(1,2,0).cpu().numpy()
    rgb2 = rgb2[0].permute(1,2,0).cpu().numpy()
    return structural_similarity(rgb2, rgb1,  channel_axis=2, data_range=rgb1.max() - rgb1.min())

def evaluate_predictions(preds_dir: Path, targets_dir: Path, loss_fns: dict, pred_filenames=None):
    if pred_filenames is None:
        pred_filenames = [f for f in os.listdir(preds_dir) if f.startswith('00') and f.endswith('jpg')]
    results = {}
    for pred_filename in tqdm.tqdm(pred_filenames):
        pred_img = Image.open(preds_dir / pred_filename)
        if (targets_dir / pred_filename).is_file():
            target_img = Image.open(targets_dir / pred_filename)
        else:
            print(f"target for {pred_filename} not found")
            continue
        pred_img, target_img = norm_img(to_tensor(pred_img))[None], to_tensor(target_img)[None]
        results[pred_filename] = {}
        for loss_name, loss_fn in loss_fns.items():
            results[pred_filename][loss_name] = loss_fn(pred_img, target_img)
        # print(pred_filename, results[pred_filename])
    print('-'*20)
    for loss_name in loss_fns.keys():
        print(loss_name, np.mean([results[fn][loss_name] for fn in results]))




def main():
    loss_fns = {"de76": de76, "de94": de94, "mae": torch.nn.L1Loss(), "psnr": psnr, "mse": torch.nn.MSELoss(), "ssim": ssim}
    for mode in ['val', 'test']:
        filenames_txt_path = Path(DATASET_DIR) / {'val':'val_images.txt', 'test':'test_images.txt'}[mode]
        with open(filenames_txt_path, 'r') as f:
            filenames =  f.read().splitlines()

        for method_name in ['clutnet', 'curl', 'ia3dlut', 'maxim', 'nse', 'oracle3dlut']: 
            print('='*20)
            print(mode, method_name)
            preds_dir = Path('predictions') / method_name
            evaluate_predictions(preds_dir,
                                targets_dir=Path(DATASET_DIR) / {'val':'train', 'test':'test'}[mode] / "target",
                                loss_fns=loss_fns, pred_filenames=filenames)

if __name__ == '__main__':
    main()
