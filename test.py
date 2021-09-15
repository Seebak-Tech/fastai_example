#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# This short example is based on the fastai GitHub
# Repository of vision examples
# https://github.com/fastai/fastai/blob/master/examples/vision.ipynb
# Modified here to show mlflow.fastai.autolog() capabilities
#
import argparse
import fastai.vision as vis
import mlflow.fastai
import dvc.api
import tarfile


def parse_args():
    parser = argparse.ArgumentParser(description="Fastai example")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate to update step size at each step (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs (default: 5). Note it takes about 1 min per epoch",
    )
    return parser.parse_args()


def extract(extract_path='~/.fastai/prueba'):
    tar = tarfile.open(
        mode='r',
        fileobj=dvc.api.read(
            path='data/mnist_tiny.tgz',
            repo='https://github.com/Seebak-Tech/fastai_example.git',
            remote='aws-remote',
            mode='rb'
        )
    )
    for item in tar:
        tar.extractall(extract_path)
        #  if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            #  extract(item.name, "./" + item.name[:item.name.rfind('/')])


def main():

    # Open file
    #  tar = tarfile.open(
        #  mode='r',
        #  fileobj=dvc.api.read(
            #  path='data/mnist_tiny.tgz',
            #  repo='https://github.com/Seebak-Tech/fastai_example',
            #  mode='rb'
        #  )
    #  )
    extract()


if __name__ == "__main__":
    main()
