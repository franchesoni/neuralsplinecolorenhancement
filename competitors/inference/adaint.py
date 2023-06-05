# package error
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

thisfilepath = Path(__file__)
packagepath = thisfilepath.parent.parent / "methods" / "AdaInt"
sys.path.append(str(packagepath / "adaint"))
sys.path.append(str(packagepath))

from demo import enhancement_inference as enhance
from mmagic.apis import init_model
from mmagic.core import tensor2img


def _minmaxnorm(x):
    return (x - x.min()) / (x.max() - x.min())


def out_to_rgbnpuint8(out):
    return tensor2img(out)


def load_img_fn(img_path):
    img = _minmaxnorm(np.array(Image.open(img_path)))
    img = torch.Tensor(img).permute(2, 0, 1)[None]
    return img


def get_model():
    config_file = packagepath / "adaint/configs/fivekrgb.py"
    ckpt_path = packagepath / "pretrained/AiLUT-FiveK-sRGB.pth"
    model = init_model(config_file, ckpt_path)
    return model


def demo(img_path, out_path):
    model = get_model()
    out = enhance(model, img_path)
    out = out_to_rgbnpuint8(out)
    Image.fromarray(out).save(out_path)


if __name__ == "__main__":
    paper_name = "adaint"
    demo(
        img_path=thisfilepath.parent / "demo_images/a0094_480p.jpg",
        out_path=thisfilepath.parent / f"results/{paper_name}_480p.jpg",
    )
    demo(
        img_path=thisfilepath.parent / "demo_images/a0094_1080p.jpg",
        out_path=thisfilepath.parent / f"results/{paper_name}_1080p.jpg",
    )
