import os
import torch
import numpy as np
from PIL import Image
import sys
import tensorflow as tf

from pathlib import Path
thisfilepath = Path(__file__)
packagepath = thisfilepath.parent.parent / "methods" / "CLUT-Net"
sys.path.append(str(packagepath))

from parameter import parser
from models import CLUTNet


def _minmaxnorm(x):
  return (x - x.min()) / (x.max() - x.min())

def out_to_rgbnpuint8(out):
  return (_minmaxnorm(out[0].permute(1, 2, 0).cpu().numpy())*255).astype(np.uint8)

def enhance(model, img_tuple):
  img_ds, img = img_tuple
  with torch.no_grad():
    out, _ = model(img_ds, img)
  return out

def load_img_fn(img_path):
  img_ds = _minmaxnorm(np.array(Image.open(img_path).resize((256, 256))))
  img_ds = torch.Tensor(img_ds).permute(2, 0, 1)[None]
  img = _minmaxnorm(np.array(Image.open(img_path)))
  img = torch.Tensor(img).permute(2, 0, 1)[None]
  return img_ds, img

def get_model():
  opt = parser.parse_args()
  opt.epoch = 361
  opt.model = "20+05+20"
  model = CLUTNet(opt.model, dim=opt.dim)
  load = torch.load(packagepath / "FiveK/20+05+20_models/model0361.pth", map_location='cpu')
  model.load_state_dict(load, strict=True)
  print("model loaded from epoch "+str(opt.epoch))
  return model


def demo(img_path, out_path):
  img = load_img_fn(img_path)
  model = get_model()
  out = enhance(model, img)
  out = out_to_rgbnpuint8(out)
  Image.fromarray(out).save(out_path)



if __name__ == "__main__":
    paper_name="clutnet"
    demo(
        img_path=thisfilepath.parent / "demo_images/a0094_480p.jpg",
        out_path=thisfilepath.parent / f"results/{paper_name}_480p.jpg",
    )
    demo(
        img_path=thisfilepath.parent / "demo_images/a0094_1080p.jpg",
        out_path=thisfilepath.parent / f"results/{paper_name}_1080p.jpg",
    )

