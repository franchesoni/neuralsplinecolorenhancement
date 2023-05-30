from abc import ABC, abstractmethod
from PIL import Image
import torch
import numpy as np

from dataset import TrainMIT5KDataset
from config import DATASET_DIR, DEVICE


class AbstractSpline(ABC):
    @abstractmethod
    def init_params(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def enhance(self, raw, params):
        raise NotImplementedError


class AdaptiveGamma(AbstractSpline, torch.nn.Module):
    def init_params(self, initial_value=1, **kwargs):
        gamma = torch.ones(1) * initial_value
        gamma.requires_grad = True
        return {"gamma": gamma}

    def enhance(self, x, params):
        return x ** params["gamma"]


class TPS2RGBSpline(AbstractSpline, torch.nn.Module):
    def init_params(self, raw, enh, n_knots=10, lparam=1., d_null=4, **kwargs):
        self.n_knots = n_knots
        self.d_null = d_null
        assert isinstance(raw, torch.Tensor) and isinstance(enh, torch.Tensor)
        assert raw.shape == (3, 448, 448)
        with torch.no_grad():
            raw = raw.permute(1, 2, 0)  # HxWx3
            enh = enh.permute(1, 2, 0)  # HxWx3

            r_img = raw.reshape(-1, 3)
            e_img = enh.reshape(-1, 3)
            M = len(r_img)

            # choose n_knots random knots
            print("number knots", self.n_knots, type(self.n_knots))

            idxs = np.arange(M)
            idxs = idxs[: self.n_knots]
            K = self.build_k_train(r_img[idxs, :], lparam=lparam)
            print("K", K.shape)
            print("e_img", e_img[idxs, :].shape)
            y = torch.vstack((e_img[idxs, :], torch.zeros((self.d_null, 3))))
            alphas = torch.linalg.pinv(K) @ y
        alphas.requires_grad = True
        xs = r_img[idxs, :]
        xs.requires_grad = True
        d = {"alphas": alphas, "xs": xs}
        return d

    def build_k_train(self, xs_control, lparam):
        # "classic" TPS energy (m=2), null space is just the affine functions span{1, r, g, b}
        # xs_control : Nx3
        # xs_eval : Mx3
        # returns Mx(N+4) matrix
        M = xs_control.shape[0]
        dim_null = xs_control.shape[1]
        d = torch.linalg.norm(
            xs_control[:, None] - xs_control[None], axis=2
        ) + lparam * torch.eye(M)
        top = torch.hstack((d, torch.ones((M, 1)), xs_control))
        bottom = torch.hstack(
            (
                torch.vstack((torch.ones((1, M)), xs_control.T)),
                torch.zeros((dim_null + 1, dim_null + 1)),
            )
        )
        return torch.vstack((top, bottom))

    def enhance(self, raw, params):
        assert raw.shape[1:] == (3, 448, 448)
        assert params['xs'].shape[1:] == (self.n_knots, 3)
        assert params['xs'].shape[0] == raw.shape[0], 'batch size mismatch'
        raw = raw.permute(0, 2, 3, 1)  # HxWx3
        # MAYBE THIS RESHAPING IS WRONG
        fimg = raw.reshape(raw.shape[0], -1, raw.shape[3])  # BxMx3, flattened image
        K = self.build_k(fimg, params["xs"])
        out = K @ params["alphas"]
        return (out.reshape(raw.shape)+raw).permute(0, 3, 1, 2)  # Bx3xHxW

    def build_k(self, xs_eval: torch.Tensor, xs_control: torch.Tensor):
        # "classic" TPS energy (m=2), null space is just the affine functions span{1, r, g, b} if for instance the dimension of the null space is 3
        # xs_control : BxNx3
        # xs_eval : BxMx3
        # returns BxMx(N+4) matrix
        B = xs_eval.shape[0]
        M = xs_eval.shape[1]
        d = torch.linalg.norm(
            xs_eval[:, :, None] - xs_control[:, None], axis=3  # B x M x 1 x 3  # B x 1 x N x 3
        )  # B x M x N
        assert d.shape == (B, M, self.n_knots)
        return torch.cat((d, torch.ones((B, M, 1)), xs_eval), dim=2)

    def forward(self, raw, params):
        return self.enhance(raw, params)


def find_best_knots(raw, target, spline, loss_fn, n_iter=1000, lr=1e-4, verbose=False):
    params = spline.init_params(raw=raw, enh=enh)
    params = {k: v[None].detach() for k, v in params.items()}  # add batch size
    for k in params:
        params[k].requires_grad = True
    optimizer = torch.optim.SGD([param for param in params.values()], lr=lr, weight_decay=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=50, verbose=verbose
    )
    spline = torch.compile(spline, mode='reduce-overhead')

    best_loss = 1e9
    for i in range(n_iter):
        out = spline(raw[None], params)
        loss = loss_fn(out, target[None])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if verbose:
            print(f"iter {i+1}/{n_iter}, loss:", loss)
            outimg = (
                np.clip(out.detach()[0].permute(1, 2, 0).numpy(), 0, 1) * 255
            ).astype(np.uint8)
            Image.fromarray(outimg).save(f"tests/oracle_current.png")
            if loss < best_loss:
                best_loss = loss
                Image.fromarray(outimg).save(f"tests/oracle_best.png")

    return params


if __name__ == "__main__":
    # get data
    ds = TrainMIT5KDataset(datadir=DATASET_DIR)
    raw, enh = ds[19]
    raw, enh = raw.to(DEVICE), enh.to(DEVICE)
    # get spline

    # spline = AdaptiveGamma()
    # spline = SimplestSpline()
    spline = TPS2RGBSpline()
    spline = spline.to(DEVICE)
    # get best knots
    params = find_best_knots(
        raw, enh, spline, torch.nn.MSELoss(), n_iter=1000, lr=1e-2, verbose=True
    )
