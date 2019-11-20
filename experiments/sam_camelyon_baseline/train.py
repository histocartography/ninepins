import os
import torch
import logging
import numpy as np
import pytorch_lightning as pl

from brontes import Brontes
from torchsummary import summary
from torchvision import datasets, transforms
import histocartography.io.utils as utils

from histocartography.ml.models.resnet import ResNet
from histocartography.ml.datasets import WSIPatchSegmentationDataset

from sklearn.metrics import roc_auc_score, accuracy_score

log = logging.getLogger('Histocartography::experiments::sam')
log.setLevel(logging.DEBUG)

def auc(y_hat, y):
    y_hat = torch.argmax(y_hat, axis=1)
    if(len(np.unique(y)) == 1):
       return torch.tensor(accuracy_score(y, y_hat))
    return torch.tensor(roc_auc_score(np.array(y),np.array(y_hat)))


def label_fn(label):
    if(np.sum(label) > 0):
        return 1
    else:
        return 0

def main():

    SEED = 42
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    PATCH_SIZE = 256
    NUMBER_OF_WORKERS = 1

    DATA_PATH = '/Users/sam/Documents/coding/datasets/cam_patches'

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=data_transforms)


    train_length = int(len(full_dataset) * 0.8)
    val_length = len(full_dataset) - train_length

    train_data, val_data = torch.utils.data.random_split(
        full_dataset, [train_length, val_length])

    dataset_loaders = {
        'train': torch.utils.data.DataLoader(
            train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUMBER_OF_WORKERS
        ),
        'val': torch.utils.data.DataLoader(
            val_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUMBER_OF_WORKERS
        )
    }

    model = ResNet([2, 2], 2, 3, 10, 20, bottleneck=True)
    summary(model, (3, 224, 224))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5)

    brontes_model = Brontes(
        model=model,
        loss=torch.nn.functional.nll_loss,
        data_loaders=dataset_loaders,
        optimizers=optimizer,
        metrics={'roc': auc})

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=[0], max_nb_epochs=EPOCHS)
    else:
        trainer = pl.Trainer(max_nb_epochs=EPOCHS)

    trainer.fit(brontes_model)

if __name__ == "__main__":
    main()
