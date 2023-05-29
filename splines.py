from abc import ABC, abstractmethod
from PIL import Image
import torch
import numpy as np

class AbstractSpline(ABC):
    @abstractmethod
    def init_params(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, raw, params):
        raise NotImplementedError




def find_best_knots(raw, target, spline, loss_fn, n_iter=1000, lr=1e-2, device='cpu', verbose=False):
    params = spline.init_params()
    # configure params to be optimized
    params.requires_grad = True
    optimizer = torch.optim.Adam([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=25, verbose=verbose)

    best_loss = 1e9
    for i in range(n_iter):
        out = raw + spline.predict(raw, params)  # predict residual
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if verbose:
            print(f"iter {i+1}/{n_iter}, loss:", loss)
            outimg = np.clip(out.detach().numpy(), 0, 255).astype(np.uint8)
            Image.fromarray(outimg).save(f'tests/oracle_Gaussian_current.png')
            if loss < best_loss:
                best_loss = loss
                Image.fromarray(outimg).save(f'tests/oracle_best.png')

    return params

