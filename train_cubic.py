print('importing packages...')
import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from PIL import Image
print('importing torch...')
import torch
from torch.utils.tensorboard import SummaryWriter
import torch._dynamo
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.functional import to_tensor
print('importing local...')
from dataset import TrainMIT5KDataset
from splines import TPS2RGBSpline, TPS2RGBSplineXY, SimplestSpline
from config import DATASET_DIR, DEVICE
from ptcolor import squared_deltaE94, rgb2lab
print('ended imports, starting...')

def validate_image(backbone, spline, val_img, logdir):
    if not os.path.exists(logdir):
        Path(logdir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        params_tensor_batch = backbone(val_img)
        ys = np.array(spline.get_params(params_tensor_batch)['ys'][0].cpu())
        xs = np.linspace(0, 1, ys.shape[1]+2)
        plt.figure()
        plt.plot(xs, [0]+list(ys[0])+[1], c='r')
        plt.plot(xs, [0]+list(ys[1])+[1], c='g')
        plt.plot(xs, [0]+list(ys[2])+[1], c='b')
        plt.savefig(logdir / 'params.png')
        plt.close()
        out_batch = spline(val_img, params_tensor_batch)  # (1, 3, H, W)
        out = np.clip(out_batch[0].permute(1, 2, 0).cpu().numpy(), 0, 1)  # (H, W, 3)
        outimg = Image.fromarray((out * 255).astype("uint8"))
        outimg = outimg.resize((W, H))
        outimg.save(logdir / 'out.jpg')

def fit(
    backbone,
    spline,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    n_epochs=144,
    verbose=True,
    profiler=None,
):
    logger = SummaryWriter()
    logdir = Path(logger.get_logdir())
    val_img = Image.open(Path(DATASET_DIR) / "train" / "raw" / "004999.jpg")
    H, W = val_img.height, val_img.width
    val_img = val_img.resize((448,448))
    val_img = to_tensor(val_img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    running_loss = 0
    for epoch_idx in range(n_epochs):
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch_idx}") as pbar:
            for i, (raw_batch, target_batch) in enumerate(dataloader):
                raw_batch, target_batch = raw_batch.to(DEVICE), target_batch.to(DEVICE)
                params_tensor_batch = backbone(raw_batch)
                out_batch = spline(raw_batch, params_tensor_batch)
                loss = loss_fn(out_batch, target_batch)
                # loss += (-params_tensor_batch > 0).sum() * 10
                # loss += (params_tensor_batch-1 > 0).sum() * 1
                loss.backward()
                optimizer.step()
                # scheduler.step()
                scheduler.step(loss)

                logger.add_scalar("train/loss", loss, epoch_idx * len(dataloader) + i)
                # exponential moving average
                if running_loss == 0:
                    running_loss = loss.item()
                else:
                    running_loss = 0.9 * running_loss + 0.1 * loss.item()
                pbar.update(1)
                pbar.set_postfix({"loss": running_loss, "batch": i})



                if i % 60 == 0:  # dev
                    validate_image(backbone, spline, val_img, logdir)


                if profiler:
                    # save profiler stats
                    profiler.disable()
                    profiler.dump_stats(logdir / f"profiler.prof")
                    profiler.enable()

            torch.save(backbone.state_dict(), logdir / f"backbone_{epoch_idx}.pth")
    return backbone

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    SEED = 0
    lr = 1e-6
    n_knots = 4
    batch_size = 46
    n_epochs = 24
    reset = False
    import cProfile

    seed_everything(SEED)
    # initialize profiler
    pr = cProfile.Profile()
    pr.enable()

    A = torch.tensor([[1,0,0], [0,1,0], [0,0,1]
                      , [1,1,1]
                      , [-1,-1,1], [-1,1,-1], [1,-1,-1]
                      ]).float()
    A = (A / torch.norm(A, dim=1, keepdim=True)).T
    A = A.to(DEVICE)
    spline = SimplestSpline(n_knots=n_knots, A=A).to(DEVICE)
    n_params = spline.get_n_params()
    torch._dynamo.config.verbose = True
    spline = torch.compile(spline, mode="reduce-overhead", disable=True)

    backbone = torchvision.models.mobilenet_v3_small(num_classes=n_params).to(DEVICE)
    # current_model_dict = backbone.state_dict()
    # loaded_state_dict = torch.load("backbone_23.pth", map_location=DEVICE)
    # new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
    # backbone.load_state_dict(new_state_dict, strict=False)
    # net.fc = torch.nn.Linear(512, n_params)




    dataset = TrainMIT5KDataset(datadir=DATASET_DIR)
    assert len(dataset) > 0, "dataset is empty"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True
    )
    
    initckpt = "backbone_23.pth"

    if os.path.isfile(initckpt) and not reset:
        state_dict = torch.load(initckpt, map_location=DEVICE)
        backbone.load_state_dict(state_dict)
    else:
        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        initial_params_ys = spline.init_params()['xs'].to(DEVICE)
        for i, (raw, enh) in enumerate(dataloader):
            raw = raw.to(DEVICE)
            params_tensor = backbone(raw)
            est_params = spline.get_params(params_tensor)
            loss = loss_fn(est_params['xs'], initial_params_ys) + loss_fn(est_params['ys'], initial_params_ys) + loss_fn(est_params['lambdas'], torch.tensor(0.1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("iter", i, loss.item())
            # break  # dev
            if loss < 0.0005:
                break

            torch.save(backbone.state_dict(), "backbone_init.pth")

    # class IdentityBackbone(torch.nn.Module):
    #     def forward(self, x):
    #         return initial_params_ys.reshape(params_tensor.shape[1:])[None]
    
    # idbk = IdentityBackbone().to(DEVICE)

    # validate over one image
    val_img = Image.open(Path(DATASET_DIR) / "train" / "raw" / "004999.jpg")
    H, W = val_img.height, val_img.width
    val_img = val_img.resize((448,448))
    val_img = to_tensor(val_img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
    # validate_image(idbk, spline, val_img, Path('identity'))
    validate_image(backbone, spline, val_img, Path('initial'))

    def loss_fn(rgb1, rgb2):
        return torch.norm(rgb2lab(rgb1) - rgb2lab(rgb2), dim=1).mean()
        # return squared_deltaE94(rgb2lab(rgb1), rgb2lab(rgb2)).mean()

    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=lr,
    #     pct_start=0.05,
    #     steps_per_epoch=len(dataloader) // batch_size,
    #     epochs=n_epochs,
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=500, verbose=True
    )
    backbone = torch.compile(
        backbone, mode="reduce-overhead", disable=True
    )  # doesn't work, see https://github.com/pytorch/pytorch/issues/102539

    enhancer = fit(
        backbone,
        spline,
        dataloader,
        optimizer,
        scheduler,
        loss_fn,
        n_epochs=n_epochs,
        verbose=True,
        profiler=pr,
    )

    pr.disable()
    pr.dump_stats(f"profiler.prof")
    # breakpoint()
