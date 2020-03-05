import logging
import sys
import argparse
# import tempfile
import os
import glob

import torch
from torchvision import transforms as tfs
import mlflow
import numpy as np
import pytorch_lightning as pl

# from torch.utils.data import DataLoader
# from torch.utils.data.dataset import random_split
from histocartography.image.VorHoVerNet.dataset import data_reader, CoNSeP_cropped, AugmentedDataset, dataset_numpy_to_tensor, MixedDataset
# from brontes import Brontes
# from pl_net import plNet
from histocartography.image.VorHoVerNet.cus_brontes import CusBrontes
from histocartography.image.VorHoVerNet.model.vorhover_net import Net, CustomLoss
from histocartography.image.VorHoVerNet.utils import set_grad

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::Training')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--data_path', type=str, default='data/',
    help='path to data (default: data/)'
)
parser.add_argument(
    '-t', '--dataset', type=str, default='CoNSeP/', 
    help='dataset name (may include more than one directory)'
)
parser.add_argument(
    '-i', '--iteration', type=int, default=0, 
    help='dataset iteration (default=0)'
)
parser.add_argument(
    '-v', '--version', type=int, default=0, 
    help='dataset version (default=0)'
)
parser.add_argument(
    '--bucket', type=str, default='test-data', 
    help='s3 bucket'
)
parser.add_argument(
    '-p', '--number_of_workers', type=int, default=-1, 
    help='number of workers (default: 1)'
)
parser.add_argument(
    '-n', '--model_name', type=str, default='model', 
    help='model name (default: model)'
)
parser.add_argument(
    '--output_root', type=str, default='./',
    help='output root path (default: ./)'
)
parser.add_argument(
    '--batch_size', type=int, default=-1, metavar='N', 
    help='input batch size for training (default: 16)'
)
parser.add_argument(
    '--test_batch_size', type=int, default=16, metavar='N',
    help='input batch size for testing (default: 16)'
)
parser.add_argument(
    '--epochs', type=int, default=14, metavar='N',
    help='number of epochs to train (default: 10)'
)
parser.add_argument(
    '--lr', type=float, default=1e-4, metavar='LR',
    help='learning rate (default: {})'.format(1e-4)
)
parser.add_argument(
    '--log_interval', type=int, default=10, metavar='N', 
    help='how many batches to wait before logging training status'
)
parser.add_argument(
    '--early_stop_patience', type=int, default=5, metavar='N', 
    help='how many times to wait before early stop (default: 5)'
)
parser.add_argument(
    '--early_stop_monitor', type=str, default='val_loss', metavar='N', 
    help='criterion monitor for early stopping (default: val_loss)'
)
parser.add_argument(
    '--load_pretrained', type=str, default='False',
    help='whether load pretrained weights or not (default: False)'
)
parser.add_argument(
    '--pretrained_weights', type=str, default='../../histocartography/image/VorHoVerNet/savers/ImageNet-ResNet50-Preact.npz',
    help='output root path (default: ./)'
)
parser.add_argument(
    '--data_mix_rate', type=float, default=1.0, metavar='N', 
    help='training with how much propotion of pseudo labels (default: 1.0)'
)
parser.add_argument(
    '--stain_norm', type=str, default='True', metavar='N', 
    help='whether apply stain morm (default: True)'
)
parser.add_argument(
    '--down_sample', type=str, default='False', metavar='N',
    help='whether to down sample the images (default: False)'
)
# parser.add_argument('--inference_mode', type=bool, default=True, metavar='N', 
#                     help='save results of inference (default: True)')
# parser.add_argument('--vdir', type=str, default='train', 
#                     help='dir name of visualization images')

def main(args):
    """
    Train with pytorch_lightning

    Args:
        args (Namespace): parsed arguments
    """
    # load parameters from parser
    DATA_PATH = args.data_path
    BUCKET = args.bucket
    DATASET = args.dataset
    ITERATION = args.iteration
    VERSION = args.version
    NUMBER_OF_WORKERS = args.number_of_workers if args.number_of_workers > 0 else torch.cuda.device_count()
    MODEL_NAME = args.model_name
    BATCH_SIZE = args.batch_size if args.batch_size > 0 else NUMBER_OF_WORKERS * 12
    TEST_BATCH_SIZE = args.test_batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    LOG_INTERVAL = args.log_interval
    EARLY_STOP_PATIENCE = args.early_stop_patience
    EARLY_STOP_MONITOR = args.early_stop_monitor
    OUTPUT_ROOT = f'{args.output_root}/{args.model_name}'
    LOAD_PRETRAINED = True if 't' in args.load_pretrained.lower() else False
    PRETRAINED_WEIGHTS = args.pretrained_weights
    DATA_MIX_RATE = args.data_mix_rate
    STAIN_NORM = True if 't' in args.stain_norm.lower() else False
    DOWN_SAMPLE = True if 't' in args.down_sample.lower() else False

    assert 0 <= DATA_MIX_RATE <= 1.0, f"data mixed rate must be between 0 and 1.0, got {DATA_MIX_RATE}"

    # EXPERIMENT_NAME = f'{MODEL_NAME}_iter{ITERATION:02d}'
    # INFERENCE_MODE = True if NUMBER_OF_WORKERS == 1 else False
    INFERENCE_MODE = True

    # TODO: whether these args are needed
    VERBOSE = True

    # make sure data folder exists
    os.makedirs(DATA_PATH, exist_ok=True)

    if os.path.isdir(OUTPUT_ROOT):
        exit()

    """
    # data loaders for the GLEASON 2019 dataset
    utils.download_s3_dataset(
        utils.get_s3(), BUCKET, DATASET, DATA_PATH
    )
    # Get a list of all images
    all_img_files = glob.glob(
        os.path.join(DATA_PATH, DATASET, 'Train Imgs', '*.jpg')
    )
    label_image_pairs = {}
    for filename in all_img_files:
        slide, core = os.path.splitext(os.path.basename(filename)
                                       )[0].split('_')
        corresponding_img = f'{slide}_{core}.jpg'
        label_image_pairs[f'{slide}_{core}_classimg_nonconvex.png'
                          ] = os.path.join(
                              DATA_PATH, DATASET, 'Train Imgs',
                              corresponding_img
                          )

    # Choose a set of annotations
    annotation_subpath = 'Maps*'
    label_folder = np.random.choice(
        glob.glob(f'{os.path.join(DATA_PATH,DATASET, annotation_subpath)}')
    )
    label_folder_content = glob.glob(os.path.join(label_folder, '*.png'))

    # Find the names of the annotation files if
    #  label_image_pairs[os.path.basename(label_file)]
    pairs = [
        (label_file, label_image_pairs.get(os.path.basename(label_file)))
        for label_file in label_folder_content
        if os.path.basename(label_file) in label_image_pairs
    ]
    log.debug(pairs)
    """

    # augmentation
    # TODO: original mask at bigger size so that it can be cropped into disired size after rotation
    # augs_both = tfs.Compose([
    #     tfs.ToPILImage(),
    #     tfs.RandomHorizontalFlip(), 
    #     tfs.RandomVerticalFlip(),
    #     tfs.RandomRotation(45)
    # ])

    # augs_both = tfs.Compose([
    #     tfs.ToPILImage(),
    #     tfs.RandomHorizontalFlip(),
    #     tfs.RandomRotation(45, fill=1.0)
    # ])

#     augs_image = tfs.Compose([
#         tfs.ToPILImage(),
#         tfs.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
#     ])

    # prepare data_loaders
    # train_idx = [i for i in range(1, 28) if i not in (2, 4, 12, 15)]
    if DATASET == 'CoNSeP/':
        train_idx = [i for i in range(1, 28) if i not in (2, 4, 12, 15)]
    elif DATASET == 'CoNuSeg/':
        train_idx = [i for i in range(1, 44) if i not in (2, 4, 12, 15)]
    else:
        train_idx = None

    train_dataset = CoNSeP_cropped(*data_reader(dataset=DATASET[:-1], root=f'{DATA_PATH}/{DATASET}', split='train', ver=VERSION, itr=ITERATION, doflip=True, contain_both=True, part=train_idx, norm=STAIN_NORM, down_sample=DOWN_SAMPLE))
    # train_dataset = CoNSeP_cropped(*data_reader(dataset=DATASET[:-1], root=f'{DATA_PATH}/{DATASET}', split='train', ver=VERSION, itr=ITERATION, doflip=True, contain_both=True, part=[1], norm=STAIN_NORM))
    num_train = int(len(train_dataset) * 0.8)
    num_valid = len(train_dataset) - num_train
    train_data, valid_data = torch.utils.data.dataset.random_split(train_dataset, [num_train, num_valid])
    # train_data = AugmentedDataset(dataset=train_data, transform=augs_both, target_transform=augs_both)
    # train_data = AugmentedDataset(dataset=train_data, transform=augs_image, target_transform=None)
    
    if DATA_MIX_RATE < 1.0:
        train_data = MixedDataset(train_data, rate=DATA_MIX_RATE, channel_first=True)
        valid_data = MixedDataset(valid_data, rate=DATA_MIX_RATE, channel_first=True)

    dataset_loaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMBER_OF_WORKERS), 
        'val': torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMBER_OF_WORKERS)
    }

    inf_batch_train = dataset_numpy_to_tensor(train_data, batch_size=BATCH_SIZE//NUMBER_OF_WORKERS)
    inf_batch_valid = dataset_numpy_to_tensor(valid_data, batch_size=BATCH_SIZE//NUMBER_OF_WORKERS)

    # lr_lambda = lambda epoch: [1, 0.1, 1, 0.1][(epoch - 1)//1]
    def lr_rt(epoch, mults=(1, 0.1, 1, 0.1), step=7):
        if epoch < step * len(mults) - 1:
            e = epoch if epoch == 0 else epoch + 1
            mult = (1, 0.1, 1, 0.1)[e//step]
        else:
            e = epoch + 1
            mult = mults[-1]
        print(f'current epoch: {e}, current learning rate {LEARNING_RATE * mult}')
        return mult

    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        filepath=f'{OUTPUT_ROOT}/checkpoints',
        save_top_k=1,
        verbose=VERBOSE,
        monitor=EARLY_STOP_MONITOR,
        mode='min',
        prefix=MODEL_NAME
    )
    
    from pytorch_lightning.callbacks import EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor=EARLY_STOP_MONITOR,
        min_delta=0.001,
        patience=EARLY_STOP_PATIENCE,
        verbose=VERBOSE,
        mode='min'
    )

    # phase 1 (freeze pretrained weights)
    # define model and optimizer
    model = Net(batch_size=BATCH_SIZE)
    set_grad(model.conv0, False)
    set_grad(model.encoder, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_rt)

    cbrontes_model = CusBrontes(
        model=model,
        loss=CustomLoss(),
        data_loaders=dataset_loaders, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        metrics=None,
        training_log_interval=LOG_INTERVAL,
        tracker_type='mlflow',
        visualize=INFERENCE_MODE,
        inf_batches=[inf_batch_train, inf_batch_valid],
        model_name=MODEL_NAME,
        output_root=OUTPUT_ROOT,
        num_gpus=NUMBER_OF_WORKERS,
        load_pretrained=LOAD_PRETRAINED,
        pretrained_path=PRETRAINED_WEIGHTS
    )
    
    try:
        if torch.cuda.is_available():
            print('early_stop_minitor: {}'.format(EARLY_STOP_MONITOR))
            print('early_stop_patience: {}'.format(EARLY_STOP_PATIENCE))
            # print('EPOCHS:', EPOCHS, [i for i in range(NUMBER_OF_WORKERS)])
            print()
            trainer = pl.Trainer(
                accumulate_grad_batches=4, gpus=[i for i in range(NUMBER_OF_WORKERS)], 
                default_save_path=f'{OUTPUT_ROOT}/pl_logs',
                checkpoint_callback=checkpoint_callback, early_stop_callback=early_stop_callback,
                distributed_backend='dp', max_epochs=EPOCHS, 
                train_percent_check=1.0, val_percent_check=1.0)
                # , val_check_interval=0.25
        else:
            trainer = pl.Trainer(
                accumulate_grad_batches=4,
                checkpoint_callback=checkpoint_callback, early_stop_callback=early_stop_callback,
                distributed_backend='dp')
        trainer.fit(cbrontes_model)
    except KeyboardInterrupt:
        MODEL_NAME += '_ki'
    
    # log artifacts
    cbrontes_model.log_artifacts()

    # phase 2 (all parameters trainable)
    # define model and optimizer
    model = Net(batch_size=BATCH_SIZE)
    # set_grad(model.conv0, True)
    # set_grad(model.encoder, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer.add_param_group({'params': [p for p in model.conv0.parameters()]})
    # optimizer.add_param_group({'params': [p for p in model.encoder.parameters()]})
    # optimizer.add_param_group({
    #     'conv0': [p for p in model.conv0.parameters()],
    #     'encoder': [p for p in model.encoder.parameters()]
    # })
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_rt)

    cbrontes_model = CusBrontes(
        model=model,
        loss=CustomLoss(),
        data_loaders=dataset_loaders, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        metrics=None,
        training_log_interval=LOG_INTERVAL,
        tracker_type='mlflow',
        visualize=INFERENCE_MODE,
        inf_batches=[inf_batch_train, inf_batch_valid],
        model_name=MODEL_NAME,
        output_root=OUTPUT_ROOT,
        num_gpus=NUMBER_OF_WORKERS,
        load_pretrained=False,
        pretrained_path=None
    )
    
    try:
        if torch.cuda.is_available():
            print('early_stop_minitor: {}'.format(EARLY_STOP_MONITOR))
            print('early_stop_patience: {}'.format(EARLY_STOP_PATIENCE))
            # print('EPOCHS:', EPOCHS, [i for i in range(NUMBER_OF_WORKERS)])
            print()
            trainer = pl.Trainer(
                accumulate_grad_batches=4, gpus=[i for i in range(NUMBER_OF_WORKERS)], 
                default_save_path=f'{OUTPUT_ROOT}/pl_logs',
                checkpoint_callback=checkpoint_callback, early_stop_callback=early_stop_callback,
                distributed_backend='dp',
                train_percent_check=1.0, val_percent_check=1.0)
                # , val_check_interval=0.25
        else:
            trainer = pl.Trainer(
                accumulate_grad_batches=4,
                checkpoint_callback=checkpoint_callback, early_stop_callback=early_stop_callback,
                distributed_backend='dp')
        trainer.fit(cbrontes_model)
    except KeyboardInterrupt:
        MODEL_NAME += '_ki'
    

    # # save model
    # # SAVER_PATH = 'saver_pl/'
    # MODEL_NAME = MODEL_NAME + '_epoch_{}.ckpt'.format(cbrontes_model.current_epoch)
    # # os.makedirs(SAVER_PATH, exist_ok=True)
    # saved_model = OUTPUT_ROOT + '/checkpoints/' + MODEL_NAME
    # # torch.save({'state_dict': plmodel.state_dict()}, saved_model)
    # state_dict = {
    #     'epoch': cbrontes_model.current_epoch,
    #     'state_dict': cbrontes_model.state_dict()
    # }
    # torch.save(state_dict, saved_model)
    # # mlflow.log_artifact(save_model)
    # print('{} saved.'.format(saved_model))

    # log artifacts
    cbrontes_model.log_artifacts()

if __name__ == "__main__":
    main(args=parser.parse_args())
    # python train_pl.py -n model_009 --batch_size 12 --epochs 100 --early_stop_patience 10
