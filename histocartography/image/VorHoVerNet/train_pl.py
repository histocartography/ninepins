import logging
# import sys
import argparse
# import tempfile
import os
import glob

import torch
from torchvision import transforms as tfs
# import mlflow
import numpy as np
import pytorch_lightning as pl

# from torch.utils.data import DataLoader
# from torch.utils.data.dataset import random_split
from histocargraphy.image.VorHoVerNet.dataset import data_reader, CoNSeP_cropped, AugmentedDataset
# from brontes import Brontes
from histocargraphy.image.VorHoVerNet.pl_net import plNet
from histocargraphy.image.VorHoVerNet.model.vorhover_net import Net, CustomLoss

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='data/',
                    help='path to data (default: data/)')
parser.add_argument('-t', '--dataset', type=str, default='CoNSeP/', 
                    help='dataset name (may include more than one directory)')
parser.add_argument('--bucket', type=str, default='test-data', 
                    help='s3 bucket')
parser.add_argument('-p', '--number_of_workers', type=int, default='1', 
                    help='number of workers (default: 1)')
parser.add_argument('-n', '--model_name', type=str, default='model', 
                    help='model name (default: model)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', 
                    help='input batch size for training (default: 16)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='N',
                    help='input batch size for testing (default: 16)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: {})'.format(1e-4))
parser.add_argument('--log_interval', type=int, default=10, metavar='N', 
                    help='how many batches to wait before logging training status')
parser.add_argument('--early_stop_patience', type=int, default=5, metavar='N', 
                    help='how many times to wait before early stop (default: 5)')
parser.add_argument('--early_stop_monitor', type=str, default='val_loss', metavar='N', 
                    help='criterion monitor for early stopping (default: val_loss)')
parser.add_argument('--inference_mode', type=bool, default=True, metavar='N', 
                    help='save results of inference (default: True)')
# parser.add_argument('--vdir', type=str, default='train', 
#                     help='dir name of visualization images')

def main(args):
    """
    Train with brontes and pytorch_lightning
    @args: parsed arguments
    """
    # load parameters from parser
    DATA_PATH = args.data_path
    BUCKET = args.bucket
    DATASET = args.dataset
    NUMBER_OF_WORKERS = args.number_of_workers
    MODEL_NAME = args.model_name
    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    LOG_INTERVAL = args.log_interval
    EARLY_STOP_PATIENCE = args.early_stop_patience
    EARLY_STOP_MONITOR = args.early_stop_monitor
    INFERENCE_MODE = args.inference_mode

    # make sure data folder exists
    os.makedirs(DATA_PATH, exist_ok=True)

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

    augs_image = tfs.Compose([
        tfs.ToPILImage(),
        tfs.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
    ])

    # prepare data_loaders
    train_dataset = CoNSeP_cropped(*data_reader(root='CoNSeP/', split='train', contain_both=True, part=None))
    num_train = int(len(train_dataset) * 0.8)
    num_valid = len(train_dataset) - num_train
    train_data, valid_data = torch.utils.data.dataset.random_split(train_dataset, [num_train, num_valid])
    # train_data = AugmentedDataset(dataset=train_data, transform=augs_both, target_transform=augs_both)
    train_data = AugmentedDataset(dataset=train_data, transform=augs_image, target_transform=None)
    dataset_loaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMBER_OF_WORKERS), 
        'val': torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMBER_OF_WORKERS)
    }

    # define model and optimizer
    model = Net(batch_size=BATCH_SIZE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # mlflow set a tag
    # mlflow.set_tag('label_folder', label_folder)
    # mlflow log parameters
    # mlflow.log_params({
    #     'number_of_workers': NUMBER_OF_WORKERS,
    #     'seed': SEED,
    #     'batch_size': BATCH_SIZE,
    #     'learning_rate': LEARNING_RATE
    # })

    # set brontes
    plmodel = plNet(
        model=model,
        loss=CustomLoss(),
        data_loaders=dataset_loaders,
        lr=LEARNING_RATE,
        visualize=INFERENCE_MODE,
        vdir=MODEL_NAME
    )
    
    # start training
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        filepath='./savers_pl/{}'.format(MODEL_NAME),
        save_best_only=True,
        verbose=True,
        monitor=EARLY_STOP_MONITOR,
        mode='min',
        prefix=MODEL_NAME
    )
    
    from pytorch_lightning.callbacks import EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor=EARLY_STOP_MONITOR,
        min_delta=0.00,
        patience=EARLY_STOP_PATIENCE,
        verbose=True,
        mode='min'
    )
    
    try:
        if torch.cuda.is_available():
            print('early_stop_minitor: {}'.format(EARLY_STOP_MONITOR))
            print('early_stop_patience: {}'.format(EARLY_STOP_PATIENCE))
            print('EPOCHS:', EPOCHS, [i for i in range(NUMBER_OF_WORKERS)])
            trainer = pl.Trainer(accumulate_grad_batches=4, gpus=[i for i in range(NUMBER_OF_WORKERS)], 
                                 max_nb_epochs=EPOCHS, min_nb_epochs=30, 
                                 checkpoint_callback=checkpoint_callback, early_stop_callback=early_stop_callback)
        else:
            trainer = pl.Trainer(max_nb_epochs=EPOCHS, checkpoint_callback=checkpoint_callback)
        trainer.fit(plmodel)
    except KeyboardInterrupt:
        MODEL_NAME += '_ki'

    # save model
    SAVER_PATH = 'saver_pl/'
    MODEL_NAME = MODEL_NAME + '_epoch_{}.ckpt'.format(plmodel.epoch)
    os.makedirs(SAVER_PATH, exist_ok=True)
    saved_model = SAVER_PATH + MODEL_NAME
    # torch.save({'state_dict': plmodel.state_dict()}, saved_model)
    state_dict = {
        'epoch': plmodel.epoch,
        'state_dict': plmodel.state_dict()
    }
    torch.save(state_dict, saved_model)
    # mlflow.log_artifact(save_model)
    print('{} saved.'.format(saved_model))

if __name__ == "__main__":
    main(args=parser.parse_args())
    # python train_pl.py -n model_005 --batch_size 8 --epochs 100 --early_stop_patience 10
