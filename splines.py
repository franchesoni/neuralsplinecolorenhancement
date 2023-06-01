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



class NaturalCubicXY(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots, nch=3):
        super().__init__()
        self.n_knots = n_knots
        self.n_ch = nch

    def init_params(self, raw, enh, **kwargs):
        assert isinstance(raw, torch.Tensor) and isinstance(enh, torch.Tensor)
        assert raw.shape == (self.n_ch, 448, 448)
        with torch.no_grad():
            raw = raw.permute(1, 2, 0)  # HxWx3
            enh = enh.permute(1, 2, 0)  # HxWx3

            r_img = raw.reshape(-1, self.n_ch)
            e_img = enh.reshape(-1, self.n_ch)
            M = len(r_img)

            # choose n_knots random knots
            idxs = np.arange(M)
            np.random.shuffle(idxs)
            idxs = idxs[: self.n_knots]
            xs = r_img[idxs, :]
            ys = e_img[idxs, :]
        ys.requires_grad = True
        xs.requires_grad = True
        lparam = torch.rand(self.n_ch)/10
        lparam.requires_grad = True
        d = {"ys": ys, "xs": xs, "l":lparam}
        return d

    def build_k_train(self, xs_control, lparam):
        '''sets up the linear system in equation (37) for the natural cubic spline on [-1,1]
        as in algorithm 4, but for index set X=[-1,1]
        '''
        B = xs_control.shape[0]
        M = xs_control.shape[1]
        ms = torch.minimum(xs_control[:,:,None], xs_control[:,None,:])
        #x1s, x2s = torch.meshgrid(xs_control, xs_control, indexing='ij')

        # compute the kernel k^1 of H_1 (defined in section 2.6.1) elementwise (order m=2) on X=[-1,1]
        # the kernel is defined in equation 43 and the explicit form is given in remark 2.56
        #ms =  d = torch.linalg.norm(
        #    xs_control[:, :, None] - xs_control[:, None], axis=3
        #)  #torch.minimum(x1s, x2s)
        K1 = xs_control[:,:,None]*xs_control[:,None,:]*ms - 0.5*(xs_control[:,:,None]+xs_control[:,None,:])*ms**2 + 1/3.0*ms**3
        M_ = K1 + lparam[:,None, None]*(torch.eye(xs_control.shape[1])[None,:,:])
    
        # complement the Gram matrix of representers of evaluation with nullspace functions ("T" in (37))
        # the nullspace has orthonormal basis {1, 1+x} (see remark 2.56)
        top = torch.cat((M_, torch.ones((B,M,1)), xs_control[:,:,None]), dim=2)
        bottom = torch.cat(
            (
                torch.cat((torch.ones(B,1,M), xs_control[:,:,None].permute(0,2,1)), dim=1),
                torch.zeros((B,2,2))), dim=2
            )
        return torch.cat((top, bottom), dim=1)

    def build_k(self, xs_eval: torch.Tensor, xs_control: torch.Tensor):
        # "classic" TPS energy (m=2), null space is just the affine functions span{1, r, g, b} if for instance the number of channels is 3
        # xs_control : BxNx3
        # xs_eval : BxMx3
        # returns BxMxN matrix
        B = xs_eval.shape[0]
        M = xs_eval.shape[1]
        ms = torch.minimum(xs_control[:,None,:], xs_eval[:,:,None])

        # compute the kernel k^1 of H_1 (defined in section 2.6.1) elementwise (order m=2) on X=[-1,1]
        # the kernel is defined in equation 43 and the explicit form is given in remark 2.56
        K1 = xs_control[:,None,:]*xs_eval[:,:,None]*ms - 0.5*(xs_control[:,None,:]+xs_eval[:,:,None])*ms**2 + 1/3.0*ms**3

        assert K1.shape == (B, M, self.n_knots)
        return torch.cat((K1, torch.ones((B, M, 1)), xs_eval[:,:,None]), dim=2)

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
        for i in range(n_channels):
            lambda_param = params["lambdas"][:, i]  # (B, 3) to (B,)
            K_pred = self.build_k(fimg[:,:,i], params["xs"][:,:,i])
            K_ch_i = self.build_k_train(
                params["xs"][:,:,i], lparam=lambda_param
            )
            B, n_knots, n_channels = params["xs"].shape
            zs = torch.zeros((B, 2, 1)).requires_grad_()
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


class TPS2RGBSplineXY(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots, nch=3):
        super().__init__()
        self.n_knots = n_knots
        self.n_ch = nch

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
