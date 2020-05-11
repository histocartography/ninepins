import os
import argparse
from histocartography.image.VorHoVerNet.inference import run, get_checkpoint

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir', type=str, default='/work/contluty01/IBM/VorHoVerNet/m_only_crf/checkpoints',
    help='model dir (default: /work/contluty01/IBM/VorHoVerNet/m_only_crf/checkpoints)'
)

parser.add_argument(
    '--user', type=str, default='user/',
    help='user name (default: user/)'
)

parser.add_argument(
    '-v', '--version', type=int, default=0,
    help='dataset version (default: 0)'
)

parser.add_argument(
    '--dataset', type=str, default='CoNSeP',
    help='dataset (default: CoNSeP)'
)

parser.add_argument(
    '--root', type=str, default='histocartography/image/VorHoVerNet/CoNSeP/',
    help='root path (default: histocartography/image/VorHoVerNet/CoNSeP/)'
)

parser.add_argument(
    '--use_dot_branch', type=int, default=0,
    help='whether use dot branch (default: 0)'
)

# parser.add_argument(
#     '--user', type=str, default='user/',
#     help='user name (default: user/)'
# )
def main(args):
    CKPT = args.model_dir
    USER = args.user
    VERSION = args.version
    DATASET = args.dataset
    ROOT = args.root
    USE_DOT_BRANCH = args.use_dot_branch
    # root = "/work/contluty01/IBM/VorHoVerNet/m_only_crf/checkpoints"
    for filename in os.listdir(CKPT):
        if filename.endswith(".ckpt"):
            model_name = filename

    print(model_name)
    checkpoint = get_checkpoint(CKPT, model_name)
    run(checkpoint, model_name, USER, with_plot=False, ver=VERSION, dataset=DATASET, root=ROOT, use_dot_branch=USE_DOT_BRANCH)

if __name__ == "__main__":
    main(args=parser.parse_args())
