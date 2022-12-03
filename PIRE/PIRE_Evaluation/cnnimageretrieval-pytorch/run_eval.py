import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

if (args.gpu):
    print('>Using GPU 0\n\n')
    gpu = '0'
else:
    print('>Using CPU\n\n')
    gpu = '-1'

print('resnet101-gem')
os.system("python -m cirtorch.examples.test --gpu-id '"+gpu+"' --network-offtheshelf 'resnet101-gem' --datasets 'oxford5k,paris6k'")

print('\n\nvgg16-gem')
os.system("python -m cirtorch.examples.test --gpu-id '"+gpu+"' --network-offtheshelf 'vgg16-gem' --datasets 'oxford5k,paris6k'")

print('\n\nretrievalSfM120k-resnet101-gem')
os.system(" python -m cirtorch.examples.test --gpu-id '"+gpu+"' --network-path 'retrievalSfM120k-resnet101-gem' --datasets 'oxford5k,paris6k'")
