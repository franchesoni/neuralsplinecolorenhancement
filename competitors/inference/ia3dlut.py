# package error
import torch
import numpy as np
from PIL import Image
import sys
import tensorflow as tf

from pathlib import Path
thisfilepath = Path(__file__)
packagepath = thisfilepath.parent.parent / "methods" / "Image-Adaptive-3DLUT"
sys.path.append(str(packagepath))

from models_x import *

def _minmaxnorm(x):
  return (x - x.min()) / (x.max() - x.min())

def out_to_rgbnpuint8(out):
  return (_minmaxnorm(out[0].permute(1, 2, 0).cpu().numpy())*255).astype(np.uint8)

def enhance(model, img):
  with torch.no_grad():
    out = model(img)
  return out

def load_img_fn(img_path):
  img = _minmaxnorm(np.array(Image.open(img_path)))
  img = torch.Tensor(img).permute(2, 0, 1)[None]
  return img

def get_model():
  LUT0 = Generator3DLUT_identity()
  LUT1 = Generator3DLUT_zero()
  LUT2 = Generator3DLUT_zero()
  classifier = Classifier()
  trilinear_ = TrilinearInterpolation()
  # Load pretrained models
  LUTs = torch.load(packagepath / "pretrained_models/sRGB/LUTs.pth", map_location='cpu')
  LUT0.load_state_dict(LUTs["0"])
  LUT1.load_state_dict(LUTs["1"])
  LUT2.load_state_dict(LUTs["2"])
  LUT0.eval()
  LUT1.eval()
  LUT2.eval()
  classifier.load_state_dict(torch.load(packagepath / "pretrained_models/sRGB/classifier.pth", map_location='cpu'))
  classifier.eval()
  def generate_LUT(img):

      pred = classifier(img).squeeze()

      LUT = (
          pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT
      )  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

      return LUT
  def model(img):
    LUT = generate_LUT(img)
    _, result = trilinear_(LUT, img)
    return result

  return model


def demo(img_path, out_path):
  img = load_img_fn(img_path)
  model = get_model()
  out = enhance(model, img)
  out = out_to_rgbnpuint8(out)
  Image.fromarray(out).save(out_path)



if __name__=='__main__':
  paper_name="ia3dlut"
  demo(
      img_path=thisfilepath.parent / "demo_images/a0094_480p.jpg",
      out_path=thisfilepath.parent / f"results/{paper_name}_480p.jpg",
  )
  demo(
      img_path=thisfilepath.parent / "demo_images/a0094_1080p.jpg",
      out_path=thisfilepath.parent / f"results/{paper_name}_1080p.jpg",
  )