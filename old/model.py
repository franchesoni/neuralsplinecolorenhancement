from numpy import index_exp
import pytorch_lightning as pl
from torch import nn
import torch
from abc import ABC, abstractmethod
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid

from backbones import GammaBackbone, SpliNetBackbone


class AdaptiveGammaLUTNet(nn.Module):
    # LUT methods
    def __init__(self):
        super().__init__()
        self.backbone = SpliNetBackbone(n=1, nc=8, n_input_channels=3, n_output_channels=1)

    def get_params(self, x):
        gamma = self.backbone(x)
        return {"gamma": gamma}

    def enhance(self, x, params):
        return x ** params["gamma"]


class SimplestSpline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SpliNetBackbone(n=5, nc=21, n_input_channels=3, n_output_channels=3)

    def get_params(self, x):
        return {"ys": self.backbone(x)}

    def enhance(self, x, params):
        # x is (B, 3, H, W)  params['ys'] is (B, 1, 1, N)
        # something sophisticated
        print("SHAPE!", params['ys'].shape, x.shape)
        out = x.clone()
        for channel_ind in range(x.shape[1]):
            out[:, channel_ind] = self.apply_to_one_channel(out[:, channel_ind], params)
        return out
    
    def apply_to_one_channel(self, x, params):
        # x is (B, H, W)
        # params is {'ys': ys} and ys is (B, 1, 1, N=5)
        # something sophisticated
        ys = params['ys'].reshape(params['ys'].shape[0], params['ys'].shape[-1])  # (B, N)
        N = ys.shape[-1]
        xs = torch.linspace(0, 255, N+2)[None]  # (1, N)
        slopes = torch.diff(ys)/(xs[:, 1]-xs[:, 0])
        out = torch.zeros_like(x)
        for i in range(1, N):
            locations = (x < xs[:, i]) * (xs[:, i-1] <= x)
            res = ys[:, i, None, None] - (xs[:, i]-x)*slopes[:, i-1, None, None]
            out[locations] = res[locations]
        return out
class TPS_Alpha(nn.Module):
    def __init__(self, nknots=50):
        super().__init__()
        self.backbone = SpliNetBackbone(
            n=(2 * nknots + 4), nc=8, n_input_channels=3, n_output_channels=3
        )
        self.nknots = nknots

    def get_params(self, x, lambda_param=0.1):
        nout = self.backbone(x)
        xs = nout[:, :, :, : self.nknots]
        alphas = nout[:, :, :, self.nknots:]
        return {"xs": xs, "alphas": alphas}

    @staticmethod
    def build_k(xs_eval, xs_control):
        # "classic" TPS energy (m=2), null space is just the affine
        # functions span{1, r, g, b} if for instance the dimension of the
        # null space is 3
        # xs_control : (Bx)Nx3
        # xs_eval : (Bx)Mx3
        # returns (Bx)Mx(N+4) matrix
        xs_control = xs_control[:, 0]  # (B, n_channels, n_knots)
        B, n_channels, n_knots = xs_control.shape
        B, n_channels, M = xs_eval.shape
        d = torch.linalg.norm(
            xs_eval.reshape(B, n_channels, M, 1)
            - xs_control.reshape(B, n_channels, 1, n_knots),
            axis=1,
        )  # (B, M, n_knots)
        return torch.concat(
            (d, torch.ones((B, M, 1)), xs_eval.permute(0, 2, 1)), axis=2
        )

    def enhance(self, x, params, lscale=10000):
        # x is (B, 3, H, W)  params['ys'] is (B, n_experts, n_channels,
        # n_knots); params['xs'] is (B, n_experts, n_channels, n_knots);
        # params['lambdas'] is (B, n_experts, n_channels, n_knots))
        # we have n_knots total control points -- the same ones in each
        # channel -- and 3 lambdas
        B, n_channels, H, W = x.shape
        fimg = x.clone().reshape(B, 3, H * W)
        out = torch.empty_like(fimg)
        for i in range(n_channels):
            K_pred_i = self.build_k(fimg, params["xs"])
            B, _, n_channels, n_knots = params["xs"].shape
            alphas = params["alphas"][:, 0, i].reshape((B, n_knots+n_channels+1, 1))
            out[:, i] = (K_pred_i @ alphas)[..., -1]
        return out.reshape(x.shape)  # HxWx3

class ThinnestPlateSpline(nn.Module):
    def __init__(self, nknots=10):
        super().__init__()
        self.backbone = SpliNetBackbone(
            n=(2 * nknots + 1), nc=8, n_input_channels=3, n_output_channels=3
        )
        self.nknots = nknots

    def get_params(self, x, lambdas_scale=1000):
        nout = self.backbone(x)
        xs = nout[:, :, :, : self.nknots]
        ys = nout[:, :, :, self.nknots : -1]
        ls = nout[:, :, :, -1:]
        return {"ys": ys, "xs": xs, "lambdas": ls / lambdas_scale}

    @staticmethod
    def build_k(xs_eval, xs_control):
        # "classic" TPS energy (m=2), null space is just the affine
        # functions span{1, r, g, b} if for instance the dimension of the
        # null space is 3
        # xs_control : (Bx)Nx3
        # xs_eval : (Bx)Mx3
        # returns (Bx)Mx(N+4) matrix
        xs_control = xs_control[:, 0]  # (B, n_channels, n_knots)
        B, n_channels, n_knots = xs_control.shape
        B, n_channels, M = xs_eval.shape
        d = torch.linalg.norm(
            xs_eval.reshape(B, n_channels, M, 1)
            - xs_control.reshape(B, n_channels, 1, n_knots),
            axis=1,
        )  # (B, M, n_knots)
        return torch.concat(
            (d, torch.ones((B, M, 1)), xs_eval.permute(0, 2, 1)), axis=2
        )

    @staticmethod
    def build_k_train(xs_control, lambda_param):
        # "classic" TPS energy (m=2), null space is just the affine
        # functions span{1, r, g, b}
        # xs_control : B x n_expert x n_channel x n_knots
        # l : B
        # returns Bx(number_knots+dim_null)x(number_knots+dim_null) matrix
        xs_control = xs_control[:, 0]  # (B, 3, n_knots)
        B, n_channels, n_knots = xs_control.shape
        assert len(lambda_param.shape) == 1
        dim_null = n_channels + 1
        identity = torch.zeros(B, n_knots, n_knots)
        identity = identity + torch.eye(n_knots)[None]
        d = (
            torch.linalg.norm(
                xs_control.reshape(B, n_channels, n_knots, 1)
                - xs_control.reshape(B, n_channels, 1, n_knots),
                axis=1,
            )
            + lambda_param.reshape(len(lambda_param), 1, 1) * identity
        )
        exs_control = torch.hstack(
            (torch.ones((B, 1, n_knots)), xs_control)
        )  # concat axis=1
        left = torch.concat((d, exs_control), axis=1)  # hstack
        right = torch.concat(
            (
                exs_control.permute(0, 2, 1),
                torch.zeros((B, dim_null, dim_null)),
            ),
            axis=1,
        )  # hstack
        K = torch.concat((left, right), axis=2)  # vstack
        return K

    def enhance(self, x, params, lscale=10000):
        # x is (B, 3, H, W)  params['ys'] is (B, n_experts, n_channels,
        # n_knots); params['xs'] is (B, n_experts, n_channels, n_knots);
        # params['lambdas'] is (B, n_experts, n_channels, n_knots))
        # we have n_knots total control points -- the same ones in each
        # channel -- and 3 lambdas
        B, n_channels, H, W = x.shape
        fimg = x.clone().reshape(B, 3, H * W)
        out = torch.empty_like(fimg)
        for i in range(n_channels):
            lambda_param = params["lambdas"][:, :, i, :]  # (B, 1, 1)
            K_ch_i = self.build_k_train(
                params["xs"], lambda_param=lambda_param.reshape(len(lambda_param))
            )
            K_pred_i = self.build_k(fimg, params["xs"])
            B, _, n_channels, n_knots = params["xs"].shape
            zs = torch.zeros((B, n_channels + 1, 1)).requires_grad_()
            ys = params["ys"][:, 0, i].reshape((B, n_knots, 1))
            out[:, i] = (
                K_pred_i
                @ torch.linalg.pinv(K_ch_i)
                @ (torch.cat((ys, zs), axis=1))
            )[..., -1]
        return out.reshape(x.shape)  # HxWx3


class AverageGammaLUTNet(nn.Module):
    # LUT methods
    def __init__(self):
        super().__init__()
        self.backbone = GammaBackbone()

    def get_params(self, x):
        return {"gamma": self.backbone.gamma}

    def enhance(self, x, params):
        return x ** params["gamma"]


################333333


class LightningLUTNet(pl.LightningModule):
    def __init__(self, lutnet, loss_fn):
        super().__init__()
        self.lutnet = lutnet
        self.loss_fn = loss_fn

    def predict(self, x):
        assert len(x) == 1  # assume x is a tensor of size (1, 3, H, W)
        if x.size(2) == 256 and x.size(3) == 256:
            out = self(x)  # call forward
        else:
            params = self.lutnet.get_params(
                resize(x, (256, 256))
            )  # obtain params with small image
            out = self.lutnet.enhance(x, params)  # enhance large image
        return out

    def forward(self, x):
        params = self.lutnet.get_params(x)
        out = self.lutnet.enhance(x, params)
        return out

    def training_step(self, batch, batch_idx):
        raw, target = batch
        out = self(raw)
        loss = self.loss_fn(out, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        raw, target = batch  # this one has batch size 1
        out = self.predict(
            raw
        )  # we use predict and not forward because of image size
        loss = self.loss_fn(out, target)
        self.log("val_loss", loss)
        input_grid = make_grid(raw)
        out_grid = make_grid(target)
        self.logger.experiment.add_image(f"input_{batch_idx}", input_grid)
        self.logger.experiment.add_image(f"output_{batch_idx}", out_grid)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
