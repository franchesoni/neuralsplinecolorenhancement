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
                assert torch.norm(A, dim=0).allclose(torch.ones(self.n_axis, device=A.device)), "axis must be normalized"
                self.pinv_axis = torch.linalg.pinv(A)  # (n_axis, 3)
                self.mins = (self.A * (self.A < 0)).sum(dim=0)  # (n_axis,)
                self.maxs = (self.A * (self.A > 0)).sum(dim=0)

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
            mins, maxs = self.mins, self.maxs
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
        estimates = torch.empty((B, self.n_axis, H, W), device=finput.device)
        for axes_ind in range(self.n_axis):
            estimates[:, axes_ind] = self.apply_to_one_channel(finput[:, axes_ind], ys[:, axes_ind], xsmin=self.mins[axes_ind], xsmax=self.maxs[axes_ind])
        estimates = estimates.permute(0, 2, 3, 1)  # (B,H,W,n_axis)
        out = estimates @ self.pinv_axis  # (B,H,W,3)
        out = out.permute(0, 3, 1, 2)  # (B,3,H,W)
        return out

    
    def apply_to_one_channel(self, raw, ys, xsmin=0, xsmax=1):
        # raw is (B, H, W)
        # ys is (B, knots)
        # add the two extra knots 0 and 1
        ys = torch.cat([torch.ones_like(ys[:, :1])*xsmin, ys, torch.ones_like(ys[:, :1])*xsmax], dim=1)  # (B, N+2)
        xs = torch.linspace(xsmin, xsmax, self.n_knots+2)[None].to(ys.device)  # (1, N+2)
        slopes = torch.diff(ys) / (xs[:, 1] - xs[:, 0])  # (B, N+1)
        out = torch.ones_like(raw) * 99  # placeholder
        for i in range(1, self.n_knots+2):
            locations =  (xs[:, i-1, None, None] <= raw) * (raw <= xs[:, i, None, None]) 
            height_to_go = ((xs[:, i, None, None] - raw)  # (B, 1, 1) - (B, H, W) = (B, H, W)
                            * slopes[:, i-1, None, None]  # (B, 1, 1)
                            )
            res = ys[:, i, None, None] - height_to_go
            out[locations] = res[locations]
        assert not (out == 99).any()
        return out



class AxisSimplestSpline(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots, n_axis):
        """
        n_knots: number of knots, without 0,0 and 1,1, int
        axis: None or 3xA, if None, then RGB, else, each column is an axes"""
        super().__init__()
        self.n_knots = n_knots
        self.n_axis = n_axis

    def get_n_params(self):
        return (self.get_n_params_ys()  # knots
                + 3 * self.n_axis)  # A

    def get_n_params_ys(self):
        return self.n_axis * self.n_knots

    def get_params(self, params_tensor):
        return self.get_params_ys(params_tensor)  # we only return ys, A is given from outside

    def get_params_ys(self, params_tensor):
        # params_tensor is (B, n_axis*n_knots)
        assert params_tensor.shape[-1] == self.get_n_params_ys()
        B = params_tensor.shape[0]
        params_tensor = params_tensor.reshape(B, self.n_axis, self.n_knots)
        params = {"ys": params_tensor}
        return params

    def init_params(self, A=None):
        with torch.no_grad():
            if A is None:
                ys = torch.linspace(0, 1, self.n_knots+2)[1:-1][None, None]  # (B, n_knots)
            else:
                mins = (A * (A < 0)).sum(dim=0)  # (n_axis,)
                maxs = (A * (A > 0)).sum(dim=0)
                ys = torch.empty(1, self.n_axis, self.n_knots)
                for i in range(self.n_axis):
                    ys[0,i,:] = torch.linspace(mins[i], maxs[i], self.n_knots+2)[1:-1]
        return {"ys": ys, "A":A}

    def enhance(self, raw, params):
        # x is (B, 3, H, W)  params['ys'] is (B, n_ch, n_knots), params['A'] is (3, n_axis)
        assert params['A'].shape[2] == self.n_axis
        return self.enhance_arbitrary(raw, params)

    def enhance_arbitrary(self, raw, params):
        # x is (B, 3, H, W)  params['ys'] is (B, n_ch, n_knots)
        A = params['A']
        pinv_axis = torch.linalg.pinv(A)  # (B, n_axis, 3)
        mins = torch.sum(A * (A < 0), dim=1)  # (B, n_axis,)
        maxs = torch.sum(A * (A > 0), dim=1)

        B, C, H, W = raw.shape
        assert C == 3
        finput = raw.permute(0, 2, 3, 1)  # (B,H,W,C)
        finput = finput.reshape(B, H*W, C)  # (B,H*W,C)
        finput = torch.bmm(finput, A)  # (B,H*W,n_axis)
        finput = finput.reshape(B, H, W, self.n_axis)  # (B,H,W,n_axis)
        finput = finput.permute(0, 3, 1, 2)  # (B,n_axis,H,W)
        ys = params['ys']
        estimates = torch.empty((B, self.n_axis, H, W), device=finput.device)
        for axes_ind in range(self.n_axis):
            estimates[:, axes_ind] = self.apply_to_one_channel(finput[:, axes_ind], ys[:, axes_ind],
                                                            xsmin=mins[:, axes_ind], xsmax=maxs[:, axes_ind])
        estimates = estimates.permute(0, 2, 3, 1)  # (B,H,W,n_axis)
        estimates = estimates.reshape(B, H*W, self.n_axis)  # (B,H*W,n_axis)
        out = torch.bmm(estimates, pinv_axis)  # (B,H*W,3)
        out = out.reshape(B, H, W, 3)  # (B,H,W,3)
        # Image.fromarray((np.clip(out[0].detach().numpy(),0,1)*255).astype(np.uint8)).show()
        out = out.permute(0, 3, 1, 2)  # (B,3,H,W)
        return out

    
    def apply_to_one_channel(self, raw, ys, xsmin=0, xsmax=1):
        # raw is (B, H, W)
        # ys is (B, knots)
        # xsmin, xsmax are (B,)
        # add the two extra knots 0 and 1
        eps = 1e-4
        ys = torch.cat([torch.ones_like(ys[:, :1])*xsmin[..., None], ys, torch.ones_like(ys[:, :1])*xsmax[...,None]], dim=1)  # (B, N+2)
        xs = (torch.linspace(0, 1, self.n_knots+2, device=ys.device)[None] *
        ((xsmax[...,None] + eps) - xsmin[...,None]) + xsmin[...,None])  # (B, N+2)
        slopes = (torch.diff(ys, dim=1) / (xs[:, 1] - xs[:, 0])[...,None])
        out = torch.ones_like(raw) * 99  # placeholder
        for i in range(1, self.n_knots+2):
            locations =  (xs[:, i-1, None, None] <= raw) * (raw <= xs[:, i, None, None]) 
            height_to_go = ((xs[:, i, None, None] - raw)  # (B, 1, 1) - (B, H, W) = (B, H, W)
                            * slopes[:, i-1, None, None]  # (B, 1, 1)
                            )
            res = ys[:, i, None, None] - height_to_go
            out[locations] = res[locations]
        assert not (out == 99).any()
        return out


class AxisPerImageSimplestSpline(AxisSimplestSpline):
    def get_params(self, params_tensor):
        nys = self.get_n_params_ys()
        params_tensor_ys = params_tensor[:, :nys]
        paramsys = self.get_params_ys(params_tensor_ys)
        params_tensor_A = params_tensor[:, nys:]
        paramsA = self.get_params_A(params_tensor_A)
        params = {**paramsys, **paramsA}
        return params

    def get_params_A(self, params_tensor):
        # params_tensor is (B, n_axis*n_knots)
        B = params_tensor.shape[0]
        params_tensor = params_tensor.reshape(B, 3, self.n_axis)
        params = {"A": params_tensor}
        return params

## NATURAL CUBIC SPLINE WITH ARBITRARY AXIS, LEARN CONTROL POINTS XS, YS AND LAMBDAS



class NaturalCubicArbitraryXY(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots, A=None):
        super().__init__()
        self.n_knots = n_knots
        if A is None:
            self.n_ch = 3
            self.A = np.eye(3)
        else:
            with torch.no_grad():
                self.A = A
                _, self.n_ch = A.shape
                assert torch.norm(A, dim=0).allclose(torch.ones(self.n_ch, device=A.device)), "axis must be normalized"
                self.pinv_axis = torch.linalg.pinv(A)  # (n_axis, 3)
                self.mins = (self.A * (self.A < 0)).sum(dim=0)  # (n_axis,)
                self.maxs = (self.A * (self.A > 0)).sum(dim=0)

    def init_params(self):
        if self.A is None:
            xs = torch.linspace(0, 1, self.n_knots+2)[1:-1][None, None]  # (B, n_knots)
            ys = torch.linspace(0, 1, self.n_knots+2)[1:-1][None, None]  # (B, n_knots)
        else:
            mins, maxs = self.mins, self.maxs
            ys = torch.empty(1, self.n_ch, self.n_knots)
            xs = torch.empty(1, self.n_ch, self.n_knots)
            for i in range(self.n_ch):
                ys[0,i,:] = torch.linspace(mins[i], maxs[i], self.n_knots+2)[1:-1]
                xs[0,i,:] = torch.linspace(mins[i], maxs[i], self.n_knots+2)[1:-1]
        lparam = torch.rand(self.n_ch)/10

        return {"ys": ys, "xs": xs, "lambdas": lparam}

    def get_n_params(self):
        return self.n_ch * (2*self.n_knots + 1)

    def get_params(self, params_tensor):
        # params_tensor is (B, n_channels*n_knots)
        assert params_tensor.shape[-1] == self.get_n_params()
        B = params_tensor.shape[0]
        params_tensor_x = params_tensor[:, :self.n_ch*self.n_knots].reshape(B, self.n_ch, self.n_knots)
        params_tensor_y = params_tensor[:, self.n_ch*self.n_knots:2*self.n_ch*self.n_knots].reshape(B, self.n_ch, self.n_knots)
        params_tensor_l = params_tensor[:, 2*self.n_ch*self.n_knots:].reshape(B, self.n_ch, 1)
        params = {"ys": params_tensor_y, "xs": params_tensor_x, "lambdas": params_tensor_l}
        return params

    def enhance(self, raw, params):
        # x is (B, 3, H, W)  params['ys'] is (B, n_ch, n_knots)
        # something sophisticated
        if self.A is None:
            return self.enhance_RGB(raw, params)
        else:
            return self.enhance_arbitrary(raw, params)

    def enhance_RGB(self, raw, params):
        # x is (B, 3, H, W)  params['ys'] is (B, n_ch, n_knots)
        xs = params['xs']
        ys = params['ys']
        ls = params['lambdas']
        out = raw.clone()
        for channel_ind in range(self.n_ch):
            out[:, channel_ind] = self.apply_to_one_channel(out[:, channel_ind], xs[:, channel_ind], ys[:, channel_ind], ls[:, channel_ind])
        return out 

    def enhance_arbitrary(self, raw, params):
        # x is (B, 3, H, W)  params['ys'] is (B, n_ch, n_knots)
        B, C, H, W = raw.shape
        assert C == 3
        finput = raw.permute(0, 2, 3, 1)  # (B,H,W,C)
        finput = finput @ self.A  # (B,H,W,n_axis)
        finput = finput.permute(0, 3, 1, 2)  # (B,n_axis,H,W)
        xs = params['xs']
        ys = params['ys']
        ls = params['lambdas']
        estimates = torch.empty((B, self.n_ch, H, W), device=finput.device)
        for axes_ind in range(self.n_ch):
            estimates[:, axes_ind, :, :] = self.apply_to_one_channel(finput[:, axes_ind], xs[:, axes_ind], ys[:, axes_ind], ls[:, axes_ind], xsmin=self.mins[axes_ind], xsmax=self.maxs[axes_ind]).reshape(B,H,W)
        estimates = estimates.permute(0, 2, 3, 1)  # (B,H,W,n_axis)
        out = estimates @ self.pinv_axis  # (B,H,W,3)
        out = out.permute(0, 3, 1, 2)  # (B,3,H,W)
        return out

    def build_k_train(self, xs_control, lparam, min_val=0):
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
        K1 = xs_control[:,:,None]*xs_control[:,None,:]*(ms-min_val) - 0.5*(xs_control[:,:,None]+xs_control[:,None,:])*(ms+min_val)**2 + 1/3.0*(ms-min_val)**3
        M_ = K1 + lparam[:,None]*(torch.eye(xs_control.shape[1], device=K1.device)[None,:,:])
        # complement the Gram matrix of representers of evaluation with nullspace functions ("T" in (37))
        # the nullspace has orthonormal basis {1, 1+x} (see remark 2.56)
        top = torch.cat((M_, torch.ones((B,M,1), device=M_.device), xs_control[:,:,None]), dim=2)
        bottom = torch.cat(
            (
                torch.cat((torch.ones((B,1,M), device=top.device), xs_control[:,:,None].permute(0,2,1)-min_val), dim=1),
                torch.zeros((B,2,2), device=top.device)), dim=2
            )
        return torch.cat((top, bottom), dim=1)

    def build_k(self, xs_eval: torch.Tensor, xs_control: torch.Tensor, min_val=0):
        _, M = xs_control.shape
        B = xs_eval.shape[0]
        H, W = xs_eval.shape[1], xs_eval.shape[2]
        xe2 = xs_eval.clone().reshape(B, H * W)
        ms = torch.minimum(xs_control[:,None,:], xe2[:,:,None])

        # compute the kernel k^1 of H_1 (defined in section 2.6.1) elementwise (order m=2) on X=[-1,1]
        # the kernel is defined in equation 43 and the explicit form is given in remark 2.56
        K1 = xs_control[:,None,:]*xe2[:,:,None]*(ms-min_val) - 0.5*(xs_control[:,None,:]+xe2[:,:,None])*(ms+min_val)**2 + 1/3.0*(ms-min_val)**3
        assert K1.shape == (B, H*W, self.n_knots)
        return torch.cat((K1, torch.ones((B, H*W, 1), device=K1.device), xe2[:,:,None]), dim=2)

    def apply_to_one_channel(self, raw, xs, ys, ls, xsmin=0, xsmax=1):
        # raw is (B, H, W)
        # ys is (B, knots)
        # add the two extra knots 0 and 1
        B = raw.shape[0]
        kt = self.build_k_train(xs, ls, xsmin)
        kp = self.build_k(raw, xs, xsmin)        
        B = raw.shape[0]
        zs = torch.zeros((B, 2, 1), device=raw.device)
        ys = ys.reshape((B, self.n_knots, 1))
        return  kp @ torch.linalg.pinv(kt) @ (torch.cat((ys, zs), axis=1))













class NaturalCubic(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots, nch=3):
        super().__init__()
        self.n_knots = n_knots
        self.n_ch = nch

    def init_params(self, raw, enh, **kwargs):
        assert isinstance(raw, torch.Tensor) and isinstance(enh, torch.Tensor)
        assert raw.shape == (self.n_ch, 448, 448)
        with torch.no_grad():
            ts = np.linspace(0, 1, self.n_knots)
            print("RAW", raw.shape)
            xs = torch.zeros(0)

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
        d = {"alphas": ys}
        return d

    def build_k(self, xs_eval: torch.Tensor, xs_control: torch.Tensor):
        B = xs_eval.shape[0]
        M = xs_eval.shape[1]
        ms = torch.minimum(xs_control[:,None,:], xs_eval[:,:,None])
        K1 = xs_control[:,None,:]*xs_eval[:,:,None]*ms - 0.5*(xs_control[:,None,:]+xs_eval[:,:,None])*ms**2 + 1/3.0*ms**3
        assert K1.shape == (B, M, self.n_knots)
        return torch.cat((K1, torch.ones((B, M, 1)), xs_eval[:,:,None]), dim=2)

    def enhance(self, raw, params):
        assert raw.shape[1:] == (self.n_ch, 448, 448)
        assert params['alphas'].shape[1:] == (self.n_knots+2, self.n_ch)
        assert params['alphas'].shape[0] == raw.shape[0], 'batch size mismatch'
        B, n_channels, H, W = raw.shape
        assert n_channels == self.n_ch
        fimg = raw.clone().reshape(B, H * W, n_channels)
        out = torch.empty_like(fimg)
        for i in range(n_channels):
            K_pred = self.build_k(fimg[:,:,i], params["xs"][:,:,i])
            alphas = params["alphas"][:, :, i].reshape((B, self.n_knots+2, 1))
            out1 = K_pred @ alphas
            out[:, :, i] = out1[..., -1]
        return out.reshape(raw.shape)  # HxWx3

    def get_params(self, params_tensor):
        # returns the dict of params from params tensor
        xs = params_tensor[:, :self.n_ch*self.n_knots].reshape(-1, self.n_knots, self.n_ch)
        alphas = params_tensor[:, self.n_ch*self.n_knots:].reshape(-1, self.n_knots+2, self.n_ch)
        return {"xs":xs, "alphas":alphas}

    def get_n_params(self):
        # returns the number of params given the number of knots
        return 2*self.n_ch*self.n_knots +2*self.n_ch

class NaturalCubicUniform(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots, nch=3):
        super().__init__()
        self.n_knots = n_knots
        self.n_ch = nch



class NaturalCubicXY(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots, nch=3):
        super().__init__()
        self.n_knots = n_knots
        self.n_ch = nch
    
    def init_params(self, raw, enh, **kwargs):
        assert isinstance(raw, torch.Tensor) and isinstance(enh, torch.Tensor)
        assert raw.shape == (self.n_ch, 448, 448)
        with torch.no_grad():
            ts = np.linspace(0, 1, self.n_knots)
            print("RAW", raw.shape)
            xs = torch.zeros(0)

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
        assert raw.shape[1:] == (self.n_ch, 448, 448)
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
        xs = params_tensor[:, :self.n_ch*self.n_knots].reshape(-1, self.n_knots, self.n_ch)
        ys = params_tensor[:, self.n_ch*self.n_knots:2*self.n_ch*self.n_knots].reshape(-1, self.n_knots, self.n_ch)
        lambdas = params_tensor[:, 2*self.n_ch*self.n_knots:].reshape(-1, self.n_ch)
        return {"xs":xs, "ys":ys, "lambdas":lambdas}

    def get_n_params(self):
        # returns the number of params given the number of knots
        return self.n_ch*(2*self.n_knots + 1)


class TPS2RGBSplineXY(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots, n_ch=3):
        super().__init__()
        self.n_knots = n_knots
        self.n_ch = n_ch

    def init_params(self, raw, enh, d_null=4, **kwargs):
        self.n_ch = d_null-1
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
        assert raw.shape[1:] == (self.n_ch, 448, 448)
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
        xs = params_tensor[:, :self.n_ch*self.n_knots].reshape(-1, self.n_knots, self.n_ch)
        ys = params_tensor[:, self.n_ch*self.n_knots:2*self.n_ch*self.n_knots].reshape(-1, self.n_knots, self.n_ch)
        lambdas = params_tensor[:, 2*self.n_ch*n_knots:].reshape(-1, self.n_ch)
        return {"xs":xs, "ys":ys, "lambdas":lambdas}

    def get_n_params(self):
        # returns the number of params given the number of knots
        return self.n_ch*(2*self.n_knots + 1)

class TPS2RGBSpline(AbstractSpline, torch.nn.Module):
    def __init__(self, n_knots=10, n_ch=3):
        super().__init__()
        self.n_knots = n_knots
        self.n_ch = n_ch

    def init_params(self, raw, enh, lparam=1., d_null=4, **kwargs):
        assert isinstance(raw, torch.Tensor) and isinstance(enh, torch.Tensor)
        assert raw.shape == (self.n_ch, 448, 448)
        with torch.no_grad():
            raw = raw.permute(1, 2, 0)  # HxWx3
            enh = enh.permute(1, 2, 0)  # HxWx3

            r_img = raw.reshape(-1, self.n_ch)
            e_img = enh.reshape(-1, 3)
            M = len(r_img)

            # choose n_knots random knots
            print("number knots", self.n_knots, type(self.n_knots))

            idxs = np.arange(M)
            idxs = idxs[: self.n_knots]
            K = self.build_k_train(r_img[idxs, :], lparam=lparam)
            print("K", K.shape)
            print("e_img", e_img[idxs, :].shape)
            y = torch.vstack((e_img[idxs, :], torch.zeros((d_null, self.n_ch))))
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
        assert raw.shape[1:] == (self.n_ch, 448, 448)
        assert params['xs'].shape[1:] == (self.n_knots, self.n_ch)
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
        alphas = params_tensor[:, :self.n_ch*(n_knots+self.n_ch+1)].reshape(-1, self.n_knots+self.n_ch+1, self.n_ch)
        xs = params_tensor[:, self.n_ch*(self.n_knots+self.n_ch+1):].reshape(-1, self.n_knots, self.n_ch)
        return {"alphas": alphas, "xs": xs}

    def get_n_params(self):
        # returns the number of params given the number of knots
        return self.n_ch*(self.n_knots+self.n_ch+1) + self.n_ch*self.n_knots





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
