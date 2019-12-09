#!/usr/bin/env python
import numpy as np
import os
import argparse
import shutil


def main(data_dir, test_fraction):
    files = os.listdir(data_dir)
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # randomly shuffle the files
    files = list(np.array(files)[np.random.permutation(len(files))])
    os.makedirs((train_dir))
    os.makedirs(test_dir)

    train_fraction = 1 - test_fraction
    for i, f in enumerate(files):
       file_path = os.path.join(data_dir, f)
       if len(files) * train_fraction >= i:
           shutil.move(file_path, train_dir)
       else:
           shutil.move(file_path, test_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Given a directory with input files, randomly partition the "
        "files into train and test sets.")
    parser.add_argument("data_dir", help="directory containing input files")
    parser.add_argument("test_fraction", type=float,
                        help="what fraction of file to use for testing")
    args = parser.parse_args()

    main(args.data_dir, args.test_fraction)
