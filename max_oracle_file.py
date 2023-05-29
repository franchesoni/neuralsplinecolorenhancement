import time
from abc import ABC, abstractmethod
import numpy as np
from paper.config import DEVICE

import torch
from PIL import Image

from splines import BiancoSpline, TPS_RGB_ORDER_2, GaussianSpline, TPS_RGB_ORDER_2_slow_train
from ptcolor import deltaE94, rgb2lab



# given two images, find the parameters of the optimal transformation between them


class AbstractOracle(ABC):
    @abstractmethod
    def fit(self, raw, enh):
        raise NotImplementedError

    @abstractmethod
    def predict(self, raw, params):
        raise NotImplementedError


####### COMMON LOSSES ########
def compute_rgb_mse(out, enh):
    return torch.linalg.norm(out - enh, axis=2).mean()

def compute_bianco_loss(out, enh):
    out_lab = rgb2lab(
        out.permute(2, 0, 1)[None],
        white_point="d65",
        gamma_correction="srgb",
        clip_rgb=False,
        space="srgb",
    )
    enh_lab = rgb2lab(
        enh.permute(2, 0, 1)[None],
        white_point="d65",
        gamma_correction="srgb",
        clip_rgb=False,
        space="srgb",
    )
    return deltaE94(enh_lab, out_lab).mean()

class TPS_order2_oracle_slow_train(AbstractOracle):
    def __init__(self, n_knots=[200], n_iter=1000, d_null = 4):
        self.n_knots = n_knots
        self.n_iter = n_iter
        self.d_null = d_null

    def fit(self, raw, enh, verbose=True):
        params = self.init_params(raw, enh)
        optim = torch.optim.AdamW([{'params':params['ys'], 'lr':0.005}, {'params':params['xs'], 'lr':0.02}, {'params':params['lambdas'], 'lr':0.0004}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.9, patience=10, verbose=True)

        traw, tenh = torch.from_numpy(raw).double(), torch.from_numpy(enh).double()
        best_loss = 1e9
        print("TRANSFORMATION LOSS", compute_bianco_loss(traw, tenh))
        for i in range(self.n_iter):
            out = TPS_RGB_ORDER_2_slow_train.predict(traw, params)
            loss = compute_bianco_loss(out, tenh)
            if verbose:
                print(f"iter {i+1}/{self.n_iter}, loss:", loss)
                if loss < best_loss:
                    best_loss = loss
                    outimg = np.clip(out.detach().numpy(), 0, 255).astype(np.uint8)
                    Image.fromarray(outimg).save(f'tests/oracle_TPS_best.png')
                    self.params = params
                outimg = np.clip(out.detach().numpy(), 0, 255).astype(np.uint8)
                Image.fromarray(outimg).save(f'tests/oracle_TPS_current.png')
            scheduler.step(loss)
            loss.backward()
            optim.step()
            time.sleep(1)

        return self.params

    def predict(self, raw, params=None):
      if type(raw) != torch.Tensor:
        raw = torch.from_numpy(raw).double()
      with torch.no_grad():
        out = TPS_RGB_ORDER_2_slow_train.predict(raw, params).numpy()
      return out

    def init_params(self, raw, enh, l=0.01):
        print("type", type(raw), "enh", type(enh))
        if type(raw) != torch.Tensor:
            raw = torch.from_numpy(raw).double()
        if type(enh) != torch.Tensor:
            enh = torch.from_numpy(enh).double()
        r_img = raw.reshape(-1, raw.shape[2])
        e_img = enh.reshape(-1, enh.shape[2])

        # choose n_knots random knots
        M = r_img.shape[0]
        idxs = np.arange(M)
        np.random.shuffle(idxs)
        idxs = idxs[:self.n_knots[0]]
        K = TPS_RGB_ORDER_2.build_k_train(r_img[idxs,:], l=l)
        y = torch.vstack((e_img[idxs,:], torch.zeros((self.d_null,3))))
        lambdas = np.ones((1,raw.shape[2]))*l
        d = {'xs': torch.tensor(r_img[idxs,:].detach().numpy(), dtype=torch.float64).requires_grad_(), 
			 'ys': torch.tensor(e_img[idxs,:].detach().numpy(), dtype=torch.float64).requires_grad_(),
			 'lambdas': torch.tensor(lambdas, dtype=torch.float64).requires_grad_()}
        return d


class TPS_order2_RGB_oracle(AbstractOracle):
    def __init__(self, n_knots=[30], n_iter=1000, d_null = 4):
        self.n_knots = n_knots
        self.n_iter = n_iter
        self.d_null = d_null
    
    def fit(self, raw, enh, verbose=True):
        params = self.init_params(raw, enh)
        optim = torch.optim.AdamW([{'params':params['alphas'], 'lr':0.001}, {'params':params['xs'], 'lr':0.001}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.5, patience=5, verbose=True)

        traw, tenh = torch.from_numpy(raw).double(), torch.from_numpy(enh).double()
        best_loss = 1e9
        print("TRANSFORMATION LOSS", compute_bianco_loss(traw, tenh))
        for i in range(self.n_iter):
            out = TPS_RGB_ORDER_2.predict(traw, params) 
            loss = compute_bianco_loss(out, tenh)
            if verbose:
                print(f"iter {i+1}/{self.n_iter}, loss:", loss)
                if loss < best_loss:
                    best_loss = loss
                    outimg = np.clip(out.detach().numpy(), 0, 255).astype(np.uint8)
                    Image.fromarray(outimg).save(f'tests/oracle_TPS_best.png')
                    self.params = params
                outimg = np.clip(out.detach().numpy(), 0, 255).astype(np.uint8)
                Image.fromarray(outimg).save(f'tests/oracle_TPS_current.png')
            scheduler.step(loss)
            loss.backward()
            optim.step()
            time.sleep(1)

        return self.params

    def predict(self, raw, params=None):
      if type(raw) != torch.Tensor:
        raw = torch.from_numpy(raw).double()
      with torch.no_grad():
        out = TPS_RGB_ORDER_2.predict(raw, params).numpy()
      return out

    def init_params(self, raw, enh, l=0.01):
        print("type", type(raw), "enh", type(enh))
        if type(raw) != torch.Tensor:
            raw = torch.from_numpy(raw).double()
        if type(enh) != torch.Tensor:
            enh = torch.from_numpy(enh).double()
        r_img = raw.reshape(-1, 3) 
        e_img = enh.reshape(-1, 3)
        M = len(r_img)

        # choose n_knots random knots
        print("number knots", self.n_knots, type(self.n_knots))
        if len(self.n_knots) == 1:
            idxs = np.arange(M)
            np.random.shuffle(idxs)
            idxs = idxs[:self.n_knots[0]]
            K = TPS_RGB_ORDER_2.build_k_train(r_img[idxs,:], l=l)
            print("K", K.shape)
            print("e_img", e_img[idxs,:].shape)
            y = torch.vstack((e_img[idxs,:], torch.zeros((self.d_null,3))))
            alphas = torch.linalg.pinv(K)@y
            d = {'alphas': torch.tensor(alphas.detach().numpy(), dtype=torch.float64).requires_grad_(), 'xs': torch.tensor(r_img[idxs,:].detach().numpy(), dtype=torch.float64).requires_grad_()}
            return d

        # n_knots is given as an array (number_of_knots_red, number_of_knots_green, number_of_knots_blue) 
        # TODO: allow ragged arrays for different numbers of knots in each channel
        idxs0 = np.arange(M)
        np.random.shuffle(idxs0)
        idxs0 = idxs0[:self.n_knots[0]]
        K0 = TPS_RGB_ORDER_2.build_k_train(r_img[idxs0,:], l=l)
        alphas0 = torch.linalg.pinv(K0)@e_img[idxs,0]
        idxs1 = np.arange(M)
        np.random.shuffle(idxs1)
        idxs1 = idxs1[:self.n_knots[1]]
        K1 = TPS_RGB_ORDER_2.build_k_train(r_img[idxs1,:], l=l)
        alphas1 = torch.linalg.pinv(K1)@e_img[idxs,1]
        idxs2 = np.arange(M)
        np.random.shuffle(idxs2)
        idxs2 = idxs2[:self.n_knots[2]]
        K2 = TPS_RGB_ORDER_2.build_k_train(r_img[idxs2,:], l=l)
        alphas2 = torch.linalg.pinv(K1)@e_img[idxs,2]
        alphas_all = torch.hstack((alphas0, alphas1, alphas2))
        xs_all = torch.hstack((r_img[idxs0,:], r_img[idxs1,:], r_img[idxs2,:]))
        d = {'alphas': torch.tensor(alphas_all.detach().numpy(), dtype=torch.float64).requires_grad_(), 'xs': torch.tensor(xs_all.detach().numpy(), dtype=torch.float64).requires_grad_()}
        return d

         

                      
class GaussianOracle(AbstractOracle):
    def __init__(self, n_knots=30, n_iter=1000, per_channel=False, verbose=False):
        self.n_knots = n_knots
        self.per_channel = per_channel  # the opposite of 3D
        self.n_iter = n_iter
        self.verbose = verbose

    def fit(self, raw, enh):
        params = self.init_params()  # init params
        optim = torch.optim.AdamW([{'params': params['sigmas'], 'lr':1.25}, {'params':params['alphas'], 'lr':1.25e2}, {'params':params['xs'], 'lr':1.25e1}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.5, patience=5, verbose=True)

        traw, tenh = torch.from_numpy(raw).double(), torch.from_numpy(enh).double()

        best_loss = 1e9
        for i in range(self.n_iter):
            out = traw + GaussianSpline.predict(traw, params)  # predict residual


            loss = compute_bianco_loss(out, tenh)
            if self.verbose:
                print(f"iter {i+1}/{self.n_iter}, loss:", loss)
                if loss < best_loss:
                    best_loss = loss
                    outimg = np.clip(out.detach().numpy(), 0, 255).astype(np.uint8)
                    Image.fromarray(outimg).save(f'tests/oracle_Gaussian_best.png')
                    self.params = params
                outimg = np.clip(out.detach().numpy(), 0, 255).astype(np.uint8)
                Image.fromarray(outimg).save(f'tests/oracle_Gaussian_current.png')
            scheduler.step(loss)
            loss.backward()
            optim.step()

        return self.params

    def predict(self, raw, params=None):
      if type(raw) != torch.Tensor:
        raw = torch.from_numpy(raw).double()
      params = params or self.params
      with torch.no_grad():
        out = GaussianSpline.predict(raw, params).numpy()
      return out 

    def init_params(self):
        # activate gradient!!!
        if self.per_channel:
          params = dict(
            alphas = torch.randn(self.n_knots, 3, requires_grad=True),
            xs = torch.randint(low=0, high=256, size=(self.n_knots, 1, 3), requires_grad=True),
            sigmas = torch.ones(1, 3, requires_grad=True)
          )
        else:
          params = dict(
            alphas = torch.randn(self.n_knots, 3, requires_grad=True, dtype=torch.double),
            xs = torch.randint(low=0, high=256, size=(self.n_knots, 3, 3), dtype=torch.double, requires_grad=True),
            sigmas = torch.ones(1, 3, requires_grad=True, dtype=torch.double)
          )
        return params


raw = np.asarray( Image.open("raw.jpg") )
enh = np.asarray( Image.open("target.jpg") )


x = TPS_order2_oracle_slow_train().fit(raw, enh)

print(x)
