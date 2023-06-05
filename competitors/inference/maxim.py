from PIL import Image
import sys

from pathlib import Path
thisfilepath = Path(__file__)
packagepath = thisfilepath.parent.parent / "methods" / "maxim"
sys.path.append(str(packagepath))

from inferencedemo import *

def _minmaxnorm(x):
  return (x - x.min()) / (x.max() - x.min())

def out_to_rgbnpuint8(out):
  return out[0] 

def enhance(model, img):
  return predict(img)

def load_img_fn(img_path):
  return pre_process(img_path)[0]

def demo(img_path, out_path):
  input_img, height, width, height_even, width_even = pre_process(img_path)
  preds = predict(input_img)
  out = post_process(preds, height, width, height_even, width_even)
  Image.fromarray(out).save(out_path)


if __name__=='__main__':
    paper_name="maxim"
    demo(
        img_path=thisfilepath.parent / "demo_images/a0094_480p.jpg",
        out_path=thisfilepath.parent / f"results/{paper_name}_480p.jpg",
    )
    demo(
        img_path=thisfilepath.parent / "demo_images/a0094_1080p.jpg",
        out_path=thisfilepath.parent / f"results/{paper_name}_1080p.jpg",
    )

# download models
# wget -O ../maxim/checkpoint.npz https://storage.googleapis.com/gresearch/maxim/ckpt/Enhancement/FiveK/checkpoint.npz

# # RUN THIS BEFORE # # 
# cd ../maxim
# pip install -r requirements.txt
# pip install --upgrade jax
# python setup.py build
# python setup.py install