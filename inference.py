print("importing packages...")
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image

print("importing torch...")
import torch
import torch._dynamo
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor


print("importing local...")
from config import DATASET_DIR, DEVICE
from dataset import ValMIT5KDataset, norm_img
from ptcolor import rgb2lab, squared_deltaE94
from splines import SimplestSpline

print("ended imports, starting...")


def generate_predictions(
        backbone,
        spline,
        dataset,
        dstdir=Path("predictions"),
    ):
    dstdir.mkdir(exist_ok=True)
    with torch.no_grad():
        backbone.eval()
        for val_img, val_img_path, _, _ in tqdm.tqdm(dataset):
            input_val_img = norm_img(to_tensor(val_img.resize((448,448)))).unsqueeze(0).to(DEVICE)
            params_tensor_batch = backbone(input_val_img)
            splineinput = norm_img(to_tensor(val_img)).unsqueeze(0)
            out_batch = spline(splineinput, params_tensor_batch)
            params = spline.get_params(params_tensor_batch)['ys']
            plt.figure()
            for axind, axes in enumerate(spline.A.T):
                plt.plot(
                    torch.linspace(spline.mins[axind], spline.maxs[axind], spline.n_knots+2),
                    torch.cat((spline.mins[axind][None], params[0, axind], spline.maxs[axind][None])),
                    label=axes
                    )
            plt.legend()
            plt.savefig(dstdir / f'params_{val_img_path.name}')
            plt.close()

            outimg = np.clip(out_batch[0].permute(1,2,0).cpu().numpy(), 0,1)
            out_img = Image.fromarray((outimg*255).astype(np.uint8))
            out_img.save(dstdir / val_img_path.name)
            
        

def evaluate_predictions(preds_dir: Path, targets_dir: Path, loss_fns: dict):
    preds_filenames = [f for f in os.listdir(preds_dir) if f.startswith('00') and f.endswith('jpg')]
    results = {}
    for pred_filename in tqdm.tqdm(preds_filenames):
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
    print('='*20)
    for loss_name in loss_fns.keys():
        print(loss_name, np.mean([results[fn][loss_name] for fn in results]))


def validate_identity(train_dir: Path, loss_fns: dict):
    with open(train_dir.parent / "val_images.txt") as f:
        val_images = f.read().splitlines()
    raw_dir = train_dir / "raw"
    targets_dir = train_dir / "target"
    preds_dir = raw_dir

    preds_filenames = val_images
    results = {}
    for pred_filename in tqdm.tqdm(preds_filenames):
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
    print('='*20)
    for loss_name in loss_fns.keys():
        print(loss_name, np.mean([results[fn][loss_name] for fn in results]))


    


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    SEED = 0
    n_knots = 8
    ckptpath = Path("runs/Jun01_19-44-48_weird-power/backbone_23.pth")
    rundir = Path('aaa')
    DEVICE = 'cpu'

    seed_everything(SEED)

    A = torch.tensor([[1,0,0], [0,1,0], [0,0,1]
                      , [1,1,1]
                      , [-1,-1,1], [-1,1,-1], [1,-1,-1]
                      ]).float()
    A = (A / torch.norm(A, dim=1, keepdim=True)).T
    A = A.to(DEVICE)
    spline = SimplestSpline(n_knots=n_knots, A=A).to(DEVICE)
    n_params = spline.get_n_params()

    backbone = torchvision.models.mobilenet_v3_small(num_classes=n_params).to(DEVICE)
    state_dict = torch.load(ckptpath, map_location=DEVICE)
    backbone.load_state_dict(state_dict)

    # net.fc = torch.nn.Linear(512, n_params)

    dataset = ValMIT5KDataset(datadir=DATASET_DIR)
    assert len(dataset) > 0, "dataset is empty"

    def de76(rgb1, rgb2):
        return torch.norm(rgb2lab(rgb1) - rgb2lab(rgb2), dim=1).mean()

    def de94(rgb1, rgb2):
        return squared_deltaE94(rgb2lab(rgb1), rgb2lab(rgb2)).mean()

    backbone = torch.compile(
        backbone, mode="reduce-overhead", disable=True
    )  # doesn't work, see https://github.com/pytorch/pytorch/issues/102539

    # generate_predictions(
    #     backbone,
    #     spline,
    #     dataset,
    #     dstdir=rundir,
    # )
    loss_fns = {"de76": de76, "de94": de94, "mse": torch.nn.MSELoss()}
    preds_dir = Path('aaa')
    targets_dir = Path(DATASET_DIR) / "train" / "target"
    evaluate_predictions(preds_dir, targets_dir, loss_fns=loss_fns)
    # train_dir = Path(DATASET_DIR) / "train"
    # validate_identity(train_dir, loss_fns=loss_fns)
