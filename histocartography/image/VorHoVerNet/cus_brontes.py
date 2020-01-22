import time
import torch
import random
import mlflow
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from brontes import Brontes
from utils import scale


class CusBrontes(Brontes):
    """
    Inherit Brontes class and add features needed.
    """
    def __init__(
        self,
        model, loss,
        data_loaders,
        optimizers,
        metrics=None,
        training_log_interval=100,
        tracker_type='logging',
        visualize=False,
        model_name='model_default',
        root_path='./',
        num_gpus=1
    ):
        """
        Args:
            model (torch.nn.Module): a model object.
            loss (callable fn): a loss object.
            data_loaders (dict): a dict of torch.utils.data.DataLoader objects.
                It has to contain 'train', 'val' and optionally a 'test'.
            optimizers (list of/or torch.optim.Optimizer): optimizer/s adopted.
            metrics (dict): additional metrics to compute apart from the loss.
                Defaults to None.
            training_log_interval (int): number of training steps for logging.
                Defaults to 100.
            tracker_type (str): type of tracker. Defaults to 'logging'.
        """
        super(CusBrontes, self).__init__()
        """
        inherited from Brontes via super():
            model, loss, data_loaders, optimizers, tracker;
            training_step_count, validation_step_count, validation_end_count
        """
        # new parameters
        self.visualize = visualize
        self.model_name = model_name
        self.root_path = f'{root_path}/{model_name}' # for log_artifacts
        self.num_gpus = num_gpus

        self.start_epoch = 0
        self.num_patches = [len(dl.dataset) for dl in data_loaders.values()]
        self.rints = [None, None]
        self.figpath = None
        self.inf_batch_train, self.inf_batch_valid = None, None
        self.inf_type = ('train', 'valid')

        self.check_params()

    def forward(self, x):
        return self.model(x)

    def check_params(self):
        print('\nArguments:')
        print(f'\tstart epoch: {self.start_epoch + 1}')
        print(f'\troot path: {self.root_path}')
        print(f'\tvisualize: {self.visualize}')
        if self.visualize:
            import os
            self.figpath = f'{root_path}/visualize'
            os.makedirs(self.figpath, exist_ok=True)
            print('\tinference image dir: {}'.format(self.figpath))

            # reminders
            if self.gpus > 1:
                print("It is better to pass num_gpus if it is greater than 1 with visualize set to True.")


    def inference(self, gap=None, psize=270, vsize=80):
        """
        """
        for typ, (imgs, gts) in enumerate((self.inf_batch_t, self.inf_batch_v)):     
            preds = self.model(imgs)

            # transpose and reshape
            imgs = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            gts = gts.permute(0, 2, 3, 1).detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            fig, ax = plt.subplots(4, 5, figsize=(12, 9))
            gap = (psize - vsize) // 2 if gap is None else gap
            for i in range(imgs.shape[0]):
                img = imgs[i, ...]
                gt = gts[i, ...]
                pred = preds[i, ...]

                for a in ax.ravel():
                    a.axis('off')
                fig.suptitle('Epoch {} on {} patch {}'.format(self.current_epoch, inference_on, i + 1), fontsize=16)
                
                # original image for every colume
                ori = (img * 255).astype(np.uint8)[gap:gap+vsize, gap:gap+vsize, :]
                ax[0, 0].imshow(ori)
                ax[1, 0].imshow(ori)
                ax[2, 0].imshow(ori)
                ax[3, 0].imshow(ori)
                # first row: original image and exhausted masks
                gt_dot = gt[..., 3]
                gtf0 = gt[..., 4]
                gtf0 = (np.stack((gtf0,)*3, axis=-1) * 255).astype(np.uint8)
                gtf0[gt_dot == 1] = (255, 0, 0)
                gtf1 = gt[..., 5]
                gtf2 = gt[..., 6]
                ax[0, 1].imshow(gt_dot)
                ax[0, 2].imshow(gtf0)
                ax[0, 3].imshow(gtf1)
                ax[0, 4].imshow(gtf2)
                # second row: original image and pseudo masks
                gt0 = scale(gt[..., 0], 1., 0)
                gt1 = scale(gt[..., 1], 1., -1.)
                gt2 = scale(gt[..., 2], 1., -1.)
                ax[1, 1].imshow(gt_dot)
                ax[1, 2].imshow(gt0)
                ax[1, 3].imshow(gt1)
                ax[1, 4].imshow(gt2)
                # third row: original image and pure predictions
                pred_seg = scale(pred[..., 0], 1., 0)
                pred_hor = scale(pred[..., 1], 1., -1.)
                pred_ver = scale(pred[..., 2], 1., -1.)
                pred_dot = scale(pred[..., 3], 1., 0)
                ax[2, 1].imshow(pred_dot)
                ax[2, 2].imshow(pred_seg)
                ax[2, 3].imshow(pred_hor)
                ax[2, 4].imshow(pred_ver)
                # fourth row: original image and masked predictions
                seg_thres = pred_seg >= 0.5
                ax[3, 1].imshow(pred_dot >= 0.5)
                ax[3, 2].imshow(seg_thres)
                ax[3, 3].imshow(np.where(seg_thres == 1, pred_hor, 0))
                ax[3, 4].imshow(np.where(seg_thres == 1, pred_ver, 0))
                plt.savefig(f'{self.figpath}/{self.inf_type[typ]}_patch{i + 1:02d}_epoch{self.current_epoch + 1:03d}.png', dpi=120)
                plt.cla()
            plt.close()

    def log_artifacts(self)：
        """
        Log directories as mlflow artifacts
        """
        pass

    def training_step(self, batch, idx, mode='train'):
        img, gts = batch
        preds = self.forward(img)

        training_dict = {}
        training_dict['loss'] = self.loss(preds, gts)
        for name, metric in self.metrics.items():
            training_dict[name] = metric(preds, gts)
        
        if mode == "train":
            # log info
            if idx % self.training_log_interval == 0:
                self.tracker.log_tensor_dict(
                    training_dict, step=self.training_step_count
                )
            self.training_step_count += 1

            # prepare for inference
            if self.visualize and self.current_epoch == self.start_epoch:
                if self.rints[0] is None:
                    self.rints[0] = random.randint(0, self.num_patches[0]//img.shape[0]//self.num_gpus)
                if idx == self.rints[0]:
                    self.inf_batch_train = batch
        return training_dict

    def validation_step(self, batch, idx):
        img, gts = batch
        preds = self.forward(img)

        validation_dict = {
            f'val_{name}': value for name, value in self.training_step(batch, idx, mode='valid')
        }
        self.tracker.log_tensor_dict(
            validation_dict, step=self.validation_step_count
        )
        self.validation_step_count += 1
        return validation_dict

    def validation_end(self, outputs):
        names = ['val_loss'] + [f'val_{name}' for name in self.metrics.keys()]
        avg_dict = {
            f'avg_{name}': torch.stack([x[name] for x in outputs]).mean() for name in names
        }
        validation_end_dict = {
            avg_dict,
            'progress_bar': {'val_loss': avg_dict['val_loss']}
        }
        self.tracker.log_tensor_dict(
            validation_end_dict, step=self.validation_end_count
        )
        self.validation_end_count += 1
        return validation_end_dict

    @pl.data_loader
    def train_dataloader(self):
        print('\nEnter custom train dataloader method.\n')
        return self.data_loaders['train']

    @pl.data_loader
    def val_dataloader(self):
        print('\nEnter custom val dataloader method.\n')
        return self.data_loaders['val']
