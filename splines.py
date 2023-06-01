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

    @abstractmethod
    def get_params(self, params_tensor):
        raise NotImplementedError

    @abstractmethod
    def get_n_params(self, n_knots):
        # returns the number of params given the number of knots
        raise NotImplementedError

    def forward(self, raw, params):
        if not type(params) == dict:
            params = self.get_params(params)
        return self.enhance(raw, params)

class AdaptiveGamma(AbstractSpline, torch.nn.Module):
    def init_params(self, initial_value=1, **kwargs):
        gamma = torch.ones(1) * initial_value
        gamma.requires_grad = True
        return {"gamma": gamma}

    def enhance(self, x, params):
        return x ** params["gamma"]

    def get_params(self, params_tensor):
        # returns the dict of params from params tensor
        return {"gamma": params_tensor}

    def get_n_params(self):
        return 1


class SimplestSpline(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots, A=None):
        """
        n_knots: number of knots, without 0,0 and 1,1, int
        axis: None or 3xA, if None, then RGB, else, each column is an axes"""
        super().__init__()
        self.n_knots = n_knots
        with torch.no_grad():
            self.A = A  # (3, n_axis)
            self.n_axis = 3 if A is None else A.shape[1]
            if A is not None:
                assert torch.norm(A, dim=0).allclose(torch.ones(self.n_axis)), "axis must be normalized"
                self.pinv_axis = torch.linalg.pinv(A)  # (n_axis, 3)

    def get_n_params(self):
        return self.n_axis * self.n_knots

    def get_params(self, params_tensor):
        # params_tensor is (B, n_channels*n_knots)
        assert params_tensor.shape[-1] == self.get_n_params()
        B = params_tensor.shape[0]
        params_tensor = params_tensor.reshape(B, self.n_axis, self.n_knots)
        params = {"ys": params_tensor}
        return params

    def init_params(self):
        if self.A is None:
            ys = torch.linspace(0, 1, self.n_knots+2)[1:-1][None, None]  # (B, n_knots)
        else:
            mins = (self.A * (self.A < 0)).sum(dim=0)
            maxs = (self.A * (self.A > 0)).sum(dim=0)
            ys = torch.empty(1, self.n_axis, self.n_knots)
            for i in range(self.n_axis):
                ys[0,i,:] = torch.linspace(mins[i], maxs[i], self.n_knots+2)[1:-1]
        return {"ys": ys}

    def enhance(self, raw, params):
        # x is (B, 3, H, W)  params['ys'] is (B, n_ch, n_knots)
        # something sophisticated
        if self.A is None:
            return self.enhance_RGB(raw, params)
        else:
            return self.enhance_arbitrary(raw, params)

    def enhance_RGB(self, raw, params):
        # x is (B, 3, H, W)  params['ys'] is (B, n_ch, n_knots)
        ys = params['ys']
        out = raw.clone()
        for channel_ind in range(self.n_axis):
            out[:, channel_ind] = self.apply_to_one_channel(out[:, channel_ind], ys[:, channel_ind])
        return out 

    def enhance_arbitrary(self, raw, params):
        # x is (B, 3, H, W)  params['ys'] is (B, n_ch, n_knots)
        B, C, H, W = raw.shape
        assert C == 3
        finput = raw.permute(0, 2, 3, 1)  # (B,H,W,C)
        finput = finput @ self.A  # (B,H,W,n_axis)
        finput = finput.permute(0, 3, 1, 2)  # (B,n_axis,H,W)
        ys = params['ys']
        estimates = torch.empty((B, self.n_axis, H, W))
        for axes_ind in range(self.n_axis):
            estimates[:, axes_ind] = self.apply_to_one_channel(finput[:, axes_ind], ys[:, axes_ind])
        estimates = estimates.permute(0, 2, 3, 1)  # (B,H,W,n_axis)
        out = estimates @ self.pinv_axis  # (B,H,W,3)
        out = out.permute(0, 3, 1, 2)  # (B,3,H,W)
        return out

    
    def apply_to_one_channel(self, raw, ys):
        # raw is (B, H, W)
        # ys is (B, knots)
        # add the two extra knots 0 and 1
        ys = torch.cat([torch.zeros_like(ys[:, :1]), ys, torch.ones_like(ys[:, :1])], dim=1)  # (B, N+2)
        xs = torch.linspace(0, 1, self.n_knots+2)[None].to(ys.device)  # (1, N+2)
        slopes = torch.diff(ys) / (xs[:, 1] - xs[:, 0])  # (B, N+1)
        out = torch.zeros_like(raw)
        for i in range(1, self.n_knots+2):
            locations =  (xs[:, i-1, None, None] <= raw) * (raw < xs[:, i, None, None]) 
            height_to_go = ((xs[:, i, None, None] - raw)  # (B, 1, 1) - (B, H, W) = (B, H, W)
                            * slopes[:, i-1, None, None]  # (B, 1, 1)
                            )
            res = ys[:, i, None, None] - height_to_go
            out[locations] = res[locations]
        return out




class TPS2RGBSplineXY(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots):
        super().__init__()
        self.n_knots = n_knots
        self.n_ch = 3

    def init_params(self, raw, enh, d_null=4, **kwargs):
        self.n_ch = d_null-1
        assert isinstance(raw, torch.Tensor) and isinstance(enh, torch.Tensor)
        assert raw.shape == (self.n_ch, 448, 448)
        with torch.no_grad():
            raw = raw.permute(1, 2, 0)  # HxWx3
            enh = enh.permute(1, 2, 0)  # HxWx3

            r_img = raw.reshape(-1, 3)
            e_img = enh.reshape(-1, 3)
            M = len(r_img)

            # choose n_knots random knots

            idxs = np.arange(M)
            np.random.shuffle(idxs)
            idxs = idxs[: self.n_knots]
            xs = r_img[idxs, :]
            ys = e_img[idxs, :]
        ys.requires_grad = True
        xs.requires_grad = True
        print("XS", xs.shape)
        lparam = torch.rand(self.n_ch)/10
        lparam.requires_grad = True
        d = {"ys": ys, "xs": xs, "l":lparam}
        return d

    def build_k_train(self, xs_control, lparam):
        # "classic" TPS energy (m=2), null space is just the affine functions span{1, r, g, b}
        # xs_control : Nx3
        # xs_eval : Mx3
        # returns Mx(N+4) matrix
        B = xs_control.shape[0]
        M = xs_control.shape[1]
        dim_null = xs_control.shape[2]+1
        d = torch.linalg.norm(
            xs_control[:, :, None] - xs_control[:, None], axis=3
        )
        d = d + lparam.reshape(B, 1, 1) * torch.eye(M).reshape(1, M, M)
        top = torch.cat((d, torch.ones((B,M,1)), xs_control), dim=2)
        bottom = torch.cat(
            (
                torch.cat((torch.ones(B,1,M), xs_control.permute(0,2,1)), dim=1),
                torch.zeros((B,dim_null, dim_null))), dim=2
        
            )
        return torch.cat((top, bottom), dim=1)

    def build_k(self, xs_eval: torch.Tensor, xs_control: torch.Tensor):
        # "classic" TPS energy (m=2), null space is just the affine functions span{1, r, g, b} if for instance the number of channels is 3
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

    def enhance(self, raw, params):
        # raw is (B, 3, H, W)  params['ys'] is (B, n_experts, n_channels,
        # n_knots); params['xs'] is (B, n_experts, n_channels, n_knots);
        # params['lambdas'] is (B, n_experts, n_channels, 1);
        # we have n_knots total control points -- the same ones in each
        # channel -- and 3 lambdas
        assert raw.shape[1:] == (3, 448, 448)
        assert params['xs'].shape[1:] == (self.n_knots, self.n_ch)
        assert params['xs'].shape[0] == raw.shape[0], 'batch size mismatch'
        B, n_channels, H, W = raw.shape
        assert n_channels == self.n_ch
        fimg = raw.clone().reshape(B, H * W, n_channels)
        out = torch.empty_like(fimg)
        K_pred = self.build_k(fimg, params["xs"])
        for i in range(n_channels):
            lambda_param = params["lambdas"][:, i]  # (B, 3) to (B,)
            K_ch_i = self.build_k_train(
                params["xs"], lparam=lambda_param
            )
            B, n_knots, n_channels = params["xs"].shape
            zs = torch.zeros((B, n_channels + 1, 1)).requires_grad_()
            ys = params["ys"][:, :, i].reshape((B, n_knots, 1))
            out1 = K_pred @ torch.linalg.pinv(K_ch_i) @ (torch.cat((ys, zs), axis=1))
            out[:, :, i] = out1[..., -1]
        return out.reshape(raw.shape)  # HxWx3


    def get_params(self, params_tensor):
        # returns the dict of params from params tensor
        n_knots = self.n_knots
        xs = params_tensor[:, :3*n_knots].reshape(-1, n_knots, 3)
        ys = params_tensor[:, 3*n_knots:6*n_knots].reshape(-1, n_knots, 3)
        lambdas = params_tensor[:, 6*n_knots:].reshape(-1, 3)
        return {"xs":xs, "ys":ys, "lambdas":lambdas}

    def get_n_params(self):
        # returns the number of params given the number of knots
        return (3*(2*self.n_knots)) + 3

















class TPS2RGBSpline(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots=10):
        super().__init__()
        self.n_knots = n_knots

    def init_params(self, raw, enh, lparam=1., d_null=4, **kwargs):
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
            y = torch.vstack((e_img[idxs, :], torch.zeros((d_null, 3))))
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
                torch.vstack((torch.ones((1, M)), xs_control.transpose(-1,-2))),
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
        out = K @ params["alphas"]  # B x M x (n_knots+4), B x (n_knots+4) x 3
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

    def get_params(self, params_tensor):
        # returns the dict of params from params tensor
        n_knots = self.n_knots
        alphas = params_tensor[:, :3*(n_knots+4)].reshape(-1, n_knots+4, 3)
        xs = params_tensor[:, 3*(n_knots+4):].reshape(-1, n_knots, 3)
        return {"alphas": alphas, "xs": xs}

    def get_n_params(self):
        # returns the number of params given the number of knots
        return (3*(self.n_knots+4)) + (3*self.n_knots)





def find_best_knots(raw, target, spline, loss_fn, n_iter=1000, lr=1e-4, verbose=False):
    params = spline.init_params(raw=raw, enh=enh)
    params = {k: v[None].detach() for k, v in params.items()}  # add batch size
    for k in params:
        params[k].requires_grad = True
    optimizer = torch.optim.SGD([param for param in params.values()], lr=lr, weight_decay=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=50, verbose=verbose
    )
    #spline = torch.compile(spline, mode='reduce-overhead')

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
    spline = TPS2RGBSplineXY()
    spline = spline.to(DEVICE)
    # get best knots
    params = find_best_knots(
        raw, enh, spline, torch.nn.MSELoss(), n_iter=1000, lr=1e-2, verbose=True
    )
