import os
import shutil
import random

def split_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    files = os.listdir(source_dir)
    random.shuffle(files)
    train_size = int(len(files) * split_ratio)

    train_files = files[:train_size]
    test_files = files[train_size:]

    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))

    for f in test_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))

if __name__ == "__main__":
    split_dataset('./data/images', './data/train/images', './data/test/images')
    split_dataset('./data/masks', './data/train/masks', './data/test/masks')
