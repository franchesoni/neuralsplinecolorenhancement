from torch.nn import MSELoss
import pytorch_lightning as pl


from model import (
    LightningLUTNet,
    AverageGammaLUTNet,
    AdaptiveGammaLUTNet,
    SimplestSpline,
    ThinnestPlateSpline,
)
from data import FiveKDataModule

path_to_C = '/home/maxdunitz/Desktop/mlbriefs2/workdir/neural_spline_enhancement/C'


def train_average_gamma():
    lutnet = AverageGammaLUTNet()
    PLlutnet = LightningLUTNet(lutnet, loss_fn=MSELoss())
    trainer = pl.Trainer(
        fast_dev_run=False,
        overfit_batches=2,
        max_time="0:0:0:30",
        log_every_n_steps=1,
    )
    trainer.fit(
        PLlutnet, datamodule=FiveKDataModule(batch_size=8, transform="resize")
    )


def train_adaptive_gamma():
    lutnet = AdaptiveGammaLUTNet()
    PLlutnet = LightningLUTNet(lutnet, loss_fn=MSELoss())
    trainer = pl.Trainer(
        fast_dev_run=True,
        overfit_batches=1,
        max_time="0:0:0:30",
        log_every_n_steps=1,
    )
    trainer.fit(
        PLlutnet, datamodule=FiveKDataModule(batch_size=8, transform="resize")
    )


def train_simplest_spline():
    lutnet = SimplestSpline()
    PLlutnet = LightningLUTNet(lutnet, loss_fn=MSELoss())
    trainer = pl.Trainer(
        fast_dev_run=False,
        overfit_batches=2,
        max_time="0:0:0:30",
        log_every_n_steps=1,
    )
    trainer.fit(
        PLlutnet,
        datamodule=FiveKDataModule(
            "/home/maxdunitz/Desktop/mlbriefs2/workdir/neural_spline_enhancement/C",
            batch_size=8,
            transform="resize",
        ),
    )


def train_thinnest_plate_spline():
    lutnet = ThinnestPlateSpline()
    PLlutnet = LightningLUTNet(lutnet, loss_fn=MSELoss())
    trainer = pl.Trainer(
        fast_dev_run=False,
        overfit_batches=8,
        max_time="0:0:30:30",
        log_every_n_steps=1,
    )
    trainer.fit(
        PLlutnet,
        datamodule=FiveKDataModule(
            path_to_C, batch_size=8, transform="resize"
        ),
    )


if __name__ == "__main__":
    # train_average_gamma()
    # train_adaptive_gamma()
    # train_simplest_spline()
    train_thinnest_plate_spline()
