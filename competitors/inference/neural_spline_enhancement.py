import torch
import numpy as np
from PIL import Image
import sys
sys.path.append("/home/franchesoni/projects/current/color/neural_spline_enhancement")

from pathlib import Path
thisfilepath = Path(__file__)
packagepath = thisfilepath.parent.parent / "methods" / "neural_spline_enhancement"
sys.path.append(str(packagepath))



def _minmaxnorm(x):
  return (x - x.min()) / (x.max() - x.min())

def out_to_rgbnpuint8(out):
  return (_minmaxnorm(out[0][0].permute(1, 2, 0).cpu().numpy())*255).astype(np.uint8)

def enhance(model, img):
  model.eval()
  with torch.no_grad():
    out, splines = model(img)
  return out

def load_img_fn(img_path):
  img = _minmaxnorm(np.array(Image.open(img_path)))
  img = torch.Tensor(img).permute(2, 0, 1)[None]#.cuda()
  return img

def get_model():
  from NeuralSpline import NeuralSpline
  model = NeuralSpline(10, 8, 1, colorspace='srgb', 
apply_to='rgb', abs=True, 
downsample_strategy='avgpool', n_input_channels=3)#.cuda()
  # load weights from net
  state = torch.load(packagepath / "models/expC.pth", map_location='cpu')
  model.load_state_dict(state['state_dict'])
  return model

def demo(img_path, out_path):
  img = load_img_fn(img_path)
  model = get_model()
  out = enhance(model, img)
  out = out_to_rgbnpuint8(out)
  Image.fromarray(out).save(out_path)



if __name__=='__main__':
    paper_name="nse"
    demo(
        img_path=thisfilepath.parent / "demo_images/a0094_480p.jpg",
        out_path=thisfilepath.parent / f"results/{paper_name}_480p.jpg",
    )
    demo(
        img_path=thisfilepath.parent / "demo_images/a0094_1080p.jpg",
        out_path=thisfilepath.parent / f"results/{paper_name}_1080p.jpg",
    )
