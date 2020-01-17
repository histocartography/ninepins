import time
import torch
import random
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from histocartography.image.VorHoVerNet.utils import scale

class plNet(pl.LightningModule):

    def __init__(self, model, loss, data_loaders, lr=1e-4, visualize=False, vdir='model', ckpt=None):
        super(plNet, self).__init__()

        self.model = model
        self.loss = loss
        self.data_loaders = data_loaders
        self.lr = lr
        self.visualize = visualize

        self.start_epoch = 0
        self.num_patches = [len(dl.dataset) for dl in data_loaders.values()]
        self.rints = [None, None]
        self.figpath = None
        self.inf_batch_v, self.inf_batch_t = None, None
        self.inf_type = ['train', 'valid']

        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # print(list(self.model.state_dict().keys())[:10])
        
        if ckpt is not None:
            print(f'Restore model from {ckpt}.')
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch'] + 1
            # print('b', self.current_epoch)
            # self.current_epoch = checkpoint['epoch']
            # print('a', self.current_epoch)
            self.global_step = checkpoint['global_step']
            # for key in checkpoint['state_dict'].keys():
            #     key = 'model.' + key
            # print(list(checkpoint['state_dict'].keys())[:10])
            # model.load_state_dict(checkpoint['state_dict'])
            self.model.load_state_dict(checkpoint['state_dict'])
            # self.model = model
            # self._load_state_dict(checkpoint['state_dict'])
            # if isinstance(self.optimizers, dict):
            #     for opt, opt_state in zip(self.optimizers, self.optimizer_states):
            #         opt.load_state_dict(opt_state)
            # else:
            #     self.optimizers.load_state_dict(checkpoint['optimizer_states'][0])

        if self.visualize:
            import os
            self.figpath = f'./visualize/{vdir}' if ckpt is None else f'./visualize/{ckpt}'
            os.makedirs(self.figpath, exist_ok=True)
        
        self.check_params()

    def forward(self, x):
        return self.model(x)

    def check_params(self):
        print('\nArguments:')
        print('learning rate: {}'.format(self.lr))
        print('visualize: {}'.format(self.visualize))
        if self.visualize:
            print('inference image dir: {}'.format(self.figpath))
        print('')
    
    def inference(self, inference_on='valid', gap=None, psize=270, vsize=80):
        assert inference_on in ('train', 'valid'), 'inference_on should be train or valid'
        
        for typ, (imgs, gts) in enumerate((self.inf_batch_t, self.inf_batch_v)):
        # if inference_on == 'valid':
        #     imgs, gts = self.inf_batch_v
        # else:
        #     imgs, gts = self.inf_batch_t
        
            preds = self.model(imgs)

            # transpose and reshape
            imgs = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            gts = gts.permute(0, 2, 3, 1).detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            fig, ax = plt.subplots(4, 4, figsize=(12, 9))
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
                gtf0 = scale(gt[..., 3], 1., 0)
                gtf1 = scale(gt[..., 4], 1., -1.)
                gtf2 = scale(gt[..., 5], 1., -1.)
                ax[0, 1].imshow(gtf0)
                ax[0, 2].imshow(gtf1)
                ax[0, 3].imshow(gtf2)
                # second row: original image and pseudo masks
                gt0 = scale(gt[..., 0], 1., 0)
                gt1 = scale(gt[..., 1], 1., -1.)
                gt2 = scale(gt[..., 2], 1., -1.)
                ax[1, 1].imshow(gt0)
                ax[1, 2].imshow(gt1)
                ax[1, 3].imshow(gt2)
                # third row: original image and pure predictions
                pred0 = scale(pred[..., 0], 1., 0)
                pred1 = scale(pred[..., 1], 1., -1.)
                pred2 = scale(pred[..., 2], 1., -1.)
                ax[2, 1].imshow(pred0)
                ax[2, 2].imshow(pred1)
                ax[2, 3].imshow(pred2)
                # fourth row: original image and masked predictions
                seg_thres = np.where(pred[..., 0] >= 0.5, 1., 0)
                ax[3, 1].imshow(seg_thres)
                ax[3, 2].imshow(np.where(seg_thres == 1, pred1, 0))
                ax[3, 3].imshow(np.where(seg_thres == 1, pred2, 0))
                plt.savefig('{}/{}_patch{:02d}_epoch{:03d}.png'.format(self.figpath, self.inf_type[typ], i + 1, self.current_epoch), dpi=120)
                plt.cla()
            plt.close()

    def state_dict(self):
        return self.model.state_dict()

    def save_model(self, checkpoint_path):
        # TODO: add info such as epochs
        checkpoint = {}
        checkpoint['state_dict'] = self.model.state_dict()
        torch.save(checkpoint, checkpoint_path)

    def training_step(self, batch, idx):
        img, gts = batch
        preds = self.forward(img)
        loss = self.loss(preds, gts)

        # get random inference batch
        if self.current_epoch == 0 and self.current_epoch != self.start_epoch:
            self.current_epoch = self.start_epoch
        if self.current_epoch == self.start_epoch:
            if self.rints[0] is None:
                self.rints[0] = random.randint(0, self.num_patches[0]//img.shape[0])
            if idx == self.rints[0]:
                self.inf_batch_t = batch
        return {'loss': loss}

    # def training_end(self, outputs):
    #     print(type(outputs))
    #     # for x in outputs.items():
    #     #     print(type(x), x)
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     print('')
    #     print('EPOCH {:03d}, avg_loss: {:.4f}'.format(self.epoch, avg_loss))
    #     return {'loss': avg_loss, 
    #             'progress_bar': {'loss', avg_loss}}
    
    def validation_step(self, batch, idx):
        img, gts = batch
        preds = self.forward(img)
        # print('valid', img.detach().cpu().numpy().max(), gts[..., 0].detach().cpu().numpy().max())

        # get random inference batch
        if self.current_epoch == self.start_epoch:
            if self.rints[1] is None:
                self.rints[1] = random.randint(0, self.num_patches[1]//img.shape[0])
            if idx == self.rints[1]:
                self.inf_batch_v = batch
        # if idx == self.rints[1]:
        #     print(batch[0, :, 50, 50], self.inf_batch_t[0, :, 50, 50])
        return {'val_loss': self.loss(preds, gts)}

    def validation_end(self, outputs):
        # print(outputs)
        # print(torch.cat([x['val_loss'] for x in outputs], dim=0))
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.current_epoch > self.start_epoch:
            print('')
            print('Epoch {}, avg_val_loss: {:.4f}  '.format(self.current_epoch, avg_val_loss))
            if self.visualize:
                # TODO: pass val_loss and show on figure
                self.inference()
        # self.epoch += 1
        return {'val_loss': avg_val_loss, 
                'progress_bar': {'val_loss': avg_val_loss}}
    
    # def test_step(self, batch, idx):
    #     img, gts = batch

    def configure_optimizers(self):
        return self.optimizers

    @pl.data_loader
    def train_dataloader(self):
        return self.data_loaders['train']

    @pl.data_loader
    def val_dataloader(self):
        return self.data_loaders['val']

    # @pl.data_loader
    # def test_dataloader(self):
    #     return self.data_loaders[2]

def get_exist_ckpt(model_name, itr=0, dir_p='./savers_pl'):
    import re
    model_dir = f'{dir_p}/{model_name}_{itr:02d}'
    
