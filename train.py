import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import torch._dynamo
from torch.utils.data import DataLoader
import torchvision
from abc import ABC, abstractmethod
from dataset import TrainMIT5KDataset
from splines import TPS2RGBSpline


def fit(
    backbone,
    spline,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    n_epochs=24,
    verbose=True,
    profiler=None,
):
    logger = SummaryWriter()

    for epoch_idx in range(n_epochs):
        pbar = tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch_idx}")
        for i, (raw_batch, target_batch) in enumerate(dataloader):
            params_tensor_batch = backbone(raw_batch)
            out_batch = spline(raw_batch, params_tensor_batch)
            loss = loss_fn(out_batch, target_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # scheduler.step(loss)

            logger.add_scalar("train/loss", loss, epoch_idx * len(dataloader) + i)
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item(), "batch": i})

            if profiler:
                # save profiler stats
                profiler.disable()
                profiler.dump_stats(f"tests/profiler.prof")
                profiler.enable()
    return enhancer


if __name__ == "__main__":
    lr = 5e-5
    n_knots = 10
    batch_size = 8
    n_epochs = 24
    import cProfile

    # initialize profiler
    pr = cProfile.Profile()
    pr.enable()

    spline = TPS2RGBSpline(n_knots=n_knots)
    n_params = spline.get_n_params()
    torch._dynamo.config.verbose = True
    spline = torch.compile(spline, mode="reduce-overhead")

    resnet = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    resnet.fc = torch.nn.Linear(512, n_params)
    backbone = resnet
    backbone = torch.compile(
        backbone, mode="reduce-overhead", disable=True
    )  # doesn't work, see https://github.com/pytorch/pytorch/issues/102539

    dataset = TrainMIT5KDataset(datadir="dataset/C")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )

    optimizer = torch.optim.SGD(backbone.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        pct_start=0.05,
        steps_per_epoch=len(dataloader) // batch_size,
        epochs=n_epochs,
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "min", factor=0.5, patience=500, verbose=True
    # )

    loss_fn = torch.nn.MSELoss()
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
    pr.dump_stats(f"tests/profiler.prof")
    # breakpoint()
