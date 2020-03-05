import os
import argparse
from dataset import gen_pseudo_masks

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='MoNuSeg',
    help='dataset to use'
)

parser.add_argument(
    '--dataset-root', type=str, default='/work/fad11204/dataset/',
    help='directory containing dataset'
)

parser.add_argument(
    '--epsilon', type=float, default='0.2',
    help='perturbation strength'
)

parser.add_argument(
    '--run-number', type=int, default='1',
    help='run number'
)

def main(args):
    DATASET = args.dataset
    DATASET_ROOT = args.dataset_root
    EPSILON = args.epsilon
    RUN_NUMBER = args.run_number

    version = (int((EPSILON * 10) // 2) - 1) * 10 + RUN_NUMBER - 1 + 100

    if not os.path.isdir(DATASET_ROOT+DATASET+"/Train/version_{:02d}".format(version)):
        gen_pseudo_masks(dataset=DATASET, root=DATASET_ROOT+DATASET+"/", split='train', ver=version, itr=0, contain_both=True)
    if not os.path.isdir(DATASET_ROOT+DATASET+"/Test/version_{:02d}".format(version)):
        gen_pseudo_masks(dataset=DATASET, root=DATASET_ROOT+DATASET+"/", split='test', ver=version, itr=0, contain_both=True)

    print(version)

if __name__ == "__main__":
    main(args=parser.parse_args())