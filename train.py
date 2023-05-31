import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import torch._dynamo
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
from torchvision.transforms.functional import to_tensor
from abc import ABC, abstractmethod
from dataset import TrainMIT5KDataset
from splines import TPS2RGBSpline, TPS2RGBSplineXY, SimplestSpline
from config import DATASET_DIR
from ptcolor import squared_deltaE94, rgb2lab

def fit(
    backbone,
    spline,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    n_epochs=240,
    verbose=True,
    profiler=None,
):
    logger = SummaryWriter()
    val_img = Image.open(Path(DATASET_DIR) / "train" / "raw" / "004999.jpg")
    H, W = val_img.height, val_img.width
    val_img = val_img.resize((448,448))
    val_img = to_tensor(val_img).unsqueeze(0)  # (1, 3, H, W)

    for epoch_idx in range(n_epochs):
        pbar = tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch_idx}")
        for i, (raw_batch, target_batch) in enumerate(dataloader):
            params_tensor_batch = backbone(raw_batch)
            out_batch = spline(raw_batch, params_tensor_batch)
            loss = loss_fn(out_batch, target_batch)
            loss += (-params_tensor_batch > 0).sum() * 1e-1
            loss += (params_tensor_batch-1 > 0).sum() * 1e-1
            loss.backward()
            optimizer.step()
            # scheduler.step()
            scheduler.step(loss)

            logger.add_scalar("train/loss", loss, epoch_idx * len(dataloader) + i)
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item(), "batch": i})



            with torch.no_grad():
                if i % 40 == 0:
                    params_tensor_batch = backbone(val_img)
                    ys = np.array(spline.get_params(params_tensor_batch)['ys'][0])
                    xs = np.linspace(0, 1, ys.shape[1]+2)
                    plt.figure()
                    plt.plot(xs, [0]+list(ys[0])+[1], c='r')
                    plt.plot(xs, [0]+list(ys[1])+[1], c='g')
                    plt.plot(xs, [0]+list(ys[2])+[1], c='b')
                    plt.savefig('params.png')
                    plt.close()
                    out_batch = spline(val_img, params_tensor_batch)  # (1, 3, H, W)
                    out = out_batch[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                    outimg = Image.fromarray((out * 255).astype("uint8"))
                    outimg = outimg.resize((W, H))
                    outimg.save('out.jpg')

            if profiler:
                # save profiler stats
                profiler.disable()
                profiler.dump_stats(f"profiler.prof")
                profiler.enable()

        torch.save(backbone.state_dict(), f"backbone_{epoch_idx}.pth")
    return enhancer

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    SEED = 0
    lr = 1e-5
    n_knots = 10
    batch_size = 16
    n_epochs = 24
    import cProfile

    seed_everything(SEED)
    # initialize profiler
    pr = cProfile.Profile()
    pr.enable()

    spline = SimplestSpline(n_knots=n_knots)
    n_params = spline.get_n_params()
    torch._dynamo.config.verbose = True
    spline = torch.compile(spline, mode="reduce-overhead")

    backbone = torchvision.models.mobilenet_v3_small(num_classes=n_params)
    # net.fc = torch.nn.Linear(512, n_params)




    dataset = TrainMIT5KDataset(datadir=DATASET_DIR)
    assert len(dataset) > 0, "dataset is empty"
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )

    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    initial_params = spline.init_params()
    for i, (raw, enh) in enumerate(dataloader):
        params_tensor = backbone(raw)
        est_params = spline.get_params(params_tensor)
        loss = loss_fn(est_params['ys'], initial_params['ys'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("iter", i, loss.item())
        if loss < 0.005:
            break


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
        optimizer, "min", factor=0.5, patience=50, verbose=True
    )
    backbone = torch.compile(
        backbone, mode="reduce-overhead", disable=False
    )  # doesn't work, see https://github.com/pytorch/pytorch/issues/102539

    enhancer = fit(
        backbone,
        spline,
        dataloader,
        optimizer,
        scheduler,
        loss_fn,
        n_epochs=24,
        verbose=True,
        profiler=pr,
    )

    pr.disable()
    pr.dump_stats(f"profiler.prof")
    # breakpoint()
