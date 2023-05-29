from pathlib import Path
import argparse
import glob
import os

def list_images(DATASET_DIR):
  """ Create lists for train, val and test (using random250 as test
  and the last 250 of the non-test as validation)"""
  DATASET_DIR = Path(DATASET_DIR)

  # get all the images in the train folder
  images = [path.split('/')[-1] for path in sorted(glob.glob(os.path.join(DATASET_DIR, 'train', 'raw', '*.jpg')))]

  # select the last 250 as validation images
  train_images = images[:-250]
  val_images = images[-250:]

  # get the list of test images
  test_images = [path.split('/')[-1] for path in sorted(glob.glob(os.path.join(DATASET_DIR, 'test', 'raw', '*.jpg')))]

  # write lists to files
  with open(DATASET_DIR / 'train_images.txt', 'w') as f:
    f.write('\n'.join(train_images) + '\n')
  with open(DATASET_DIR / 'val_images.txt', 'w') as f:
    f.write('\n'.join(val_images) + '\n')
  with open(DATASET_DIR / 'test_images.txt', 'w') as f:
    f.write('\n'.join(test_images) + '\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datasetdir', type=str, help="path to expert C")
  args = parser.parse_args()
  list_images(args.datasetdir)