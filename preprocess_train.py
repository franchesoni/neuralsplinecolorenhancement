from pathlib import Path
import numpy as np
from PIL import Image

def norm_img(x):
    return (x - x.min()) / (x.max() - x.min())

def preprocess_train(datadir):
    rawdir = Path(datadir) / 'train' / 'raw'
    targetdir = Path(datadir) / 'train' / 'target'
    with open(Path(datadir) / 'train_images.txt', 'r') as f:
        train_imgs_list = [line.replace('\n', '') for line in f.readlines()]

    dstrawdir = Path(str(rawdir).replace('train', 'train_processed'))
    dsttargetdir = Path(str(targetdir).replace('train', 'train_processed'))
    dstrawdir.mkdir(parents=True, exist_ok=True)
    dsttargetdir.mkdir(parents=True, exist_ok=True)

    for imgpath in rawdir.glob('*.jpg'):
        if imgpath.name not in train_imgs_list:
            continue
        img = Image.open(imgpath)
        img = img.resize((448, 448))
        img = Image.fromarray((255*norm_img(np.array(img))).astype(np.uint8))
        img.save(dstrawdir / imgpath.name)

    for imgpath in targetdir.glob('*.jpg'):
        if imgpath.name not in train_imgs_list:
            continue
        img = Image.open(imgpath)
        img = img.resize((448, 448))
        img.save(dsttargetdir / imgpath.name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetdir', type=str, help="path to expert C")
    args = parser.parse_args()
    preprocess_train(args.datasetdir)
    


