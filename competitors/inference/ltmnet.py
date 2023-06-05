# package error
import torch
import numpy as np
from PIL import Image
import sys
import tensorflow as tf

from pathlib import Path
thisfilepath = Path(__file__)
packagepath = thisfilepath.parent.parent / "methods" / "ltmnet"
sys.path.append(str(packagepath))

from models.ltmnet_helpers import post_process_residual, post_process

def _minmaxnorm(x):
  return (x - x.min()) / (x.max() - x.min())

def out_to_rgbnpuint8(out):
  return (_minmaxnorm(out[0].numpy())*255).astype(np.uint8)

def enhance(model, img_tuple, res=False):
  img512p, full_img = img_tuple
  img512p = img512p[tf.newaxis, ...]
  full_img = full_img[tf.newaxis, ...]
  if res:
    results = post_process_residual(img512p, full_img, [8,8], 3, model, model.residual_net,
                                    training=False, clip=True)
  else:
    tone_curves = model(img512p, training=False)
    results = post_process(tone_curves, full_img, [8,8], 3)
  results = [x[0] for x in results]  # drop the batch dimension
  return results



def load_img_fn(img_path):
  # img = _minmaxnorm(np.array(Image.open(img_path)))
  # img512p = _minmaxnorm(np.array(Image.open(img_path).resize((512, 512))))
  img = (np.array(Image.open(img_path))).astype(float)
  img512p = (np.array(Image.open(img_path).resize((512, 512)))).astype(float)
  return img512p, img

def get_model():
  ckpt_dir = packagepath / "pretrained_models/ltmnet_res_hdrplus_ds_model"
  model = tf.keras.models.load_model(ckpt_dir)
  return model

def demo(img_path, out_path):
  img_tuple = load_img_fn(img_path)
  model = get_model()
  out = enhance(model, img_tuple, res=True)
  out = out_to_rgbnpuint8(out)
  Image.fromarray(out).save(out_path)



if __name__=='__main__':
  paper_name="ltmnet"
  demo(
      img_path=thisfilepath.parent / "demo_images/a0094_480p.jpg",
      out_path=thisfilepath.parent / f"results/{paper_name}_480p.jpg",
  )
  demo(
      img_path=thisfilepath.parent / "demo_images/a0094_1080p.jpg",
      out_path=thisfilepath.parent / f"results/{paper_name}_1080p.jpg",
  )

