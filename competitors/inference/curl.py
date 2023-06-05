# package error
import torch
import numpy as np
from PIL import Image
import sys
import tensorflow as tf

from pathlib import Path
thisfilepath = Path(__file__)
packagepath = thisfilepath.parent.parent / "methods" / "CURL"
sys.path.append(str(packagepath))

import model

def _minmaxnorm(x):
  return (x - x.min()) / (x.max() - x.min())

def out_to_rgbnpuint8(out):
  return (_minmaxnorm(out[0].permute(1, 2, 0).cpu().numpy())*255).astype(np.uint8)

def enhance(net, img):
  with torch.no_grad():
    out, _ = net(img)
  return out

def load_img_fn(img_path):
  img = _minmaxnorm(np.array(Image.open(img_path)))
  img = torch.Tensor(img).permute(2, 0, 1)[None]#.cuda()
  return img

def get_model():
  checkpoint_filepath = packagepath / "pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt"
  # Build Model
  net = model.CURLNet()
  checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
  net.load_state_dict(checkpoint['model_state_dict'])
  net = net#.cuda()
  net.eval()
  return net


def demo(img_path, out_path):
  img = load_img_fn(img_path)
  model = get_model()
  out = enhance(model, img)
  out = out_to_rgbnpuint8(out)
  Image.fromarray(out).save(out_path)



if __name__=='__main__':
  paper_name="curl"
  demo(
      img_path=thisfilepath.parent / "demo_images/a0094_480p.jpg",
      out_path=thisfilepath.parent / f"results/{paper_name}_480p.jpg",
  )
  demo(
      img_path=thisfilepath.parent / "demo_images/a0094_1080p.jpg",
      out_path=thisfilepath.parent / f"results/{paper_name}_1080p.jpg",
  )