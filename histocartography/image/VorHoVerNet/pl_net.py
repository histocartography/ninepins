import time
import torch
import random
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class plNet(pl.LightningModule):

    def __init__(self, model, loss, data_loaders, lr=1e-4, visualize=False, vdir='model'):
        super(plNet, self).__init__()

        self.model = model(batch_size=8)
        self.loss = loss
        self.data_loaders = data_loaders
        self.lr = lr
        self.visualize = visualize

        self.epoch = 0
        self.num_patches = [len(dl.dataset) for dl in data_loaders.values()]
        self.rints = [None, None]
        self.figpath = None
        self.inf_batch_v, self.inf_batch_t = None, None
        self.inf_type = ['train', 'valid']

        if self.visualize:
            import os
            self.figpath = './visualize/{}'.format(vdir)
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
        assert inference_on in ['train', 'valid'], 'inference_on should be train or valid'
        
        for typ, (imgs, gts) in enumerate((self.inf_batch_t, self.inf_batch_v)):
#         if inference_on == 'valid':
#             imgs, gts = self.inf_batch_v
#         else:
#             imgs, gts = self.inf_batch_t
        
            preds = self.model(imgs)

            # transpose and reshape
            imgs = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            gts = gts.permute(0, 2, 3, 1).detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            fig, ax = plt.subplots(3, 4, figsize=(12, 9))
            for i in range(imgs.shape[0]):
                img = imgs[i, ...]
                gt = gts[i, ...]
                pred = preds[i, ...]

                for a in ax.ravel():
                    a.axis('off')
                fig.suptitle('Epoch {} on {} patch {}'.format(self.epoch, inference_on, i + 1), fontsize=16)
                gap = (psize - vsize) // 2 if gap is None else gap
                ax[0, 0].imshow(img.astype(np.uint8)[gap:gap+vsize, gap:gap+vsize, :])
                ax[0, 1].imshow(gt[..., 0] * 255)
                ax[0, 2].imshow(gt[..., 1] * 255)
                ax[0, 3].imshow(gt[..., 2] * 255)
                ax[1, 1].imshow(pred[..., 0] * 255)
                ax[1, 2].imshow(pred[..., 1] * 255)
                ax[1, 3].imshow(pred[..., 2] * 255)
                seg_thres = np.where(pred[..., 0] >= 0.5, 255, 0)
                ax[2, 1].imshow(seg_thres)
                ax[2, 2].imshow(np.where(seg_thres == 255, pred[..., 1], 0))
                ax[2, 3].imshow(np.where(seg_thres == 255, pred[..., 2], 0))
                plt.draw()
        #         plt.show()

                plt.savefig('{}/{}_patch{:02d}_epoch{:03d}.png'.format(self.figpath, self.inf_type[typ], i + 1, self.epoch), dpi=120)
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
#         if idx == 0:
#             self.t = time.time()
        img, gts = batch
        preds = self.forward(img)
        loss = self.loss(preds, gts)
#         print('EPOCH {:03d}, {:03d}/{:03d} batches, {:.2f}s per batch, loss: {:.4f}'.format(self.epoch, idx + 1, self.num_patches[0], (time.time() - self.t)/(idx + 1), loss), end='\r')
        
        # get random inference batch
        if self.epoch == 1:
            if self.rints[0] is None:
                self.rints[0] = random.randint(0, self.num_patches[0]//img.shape[0])
            if idx == self.rints[0]:
                self.inf_batch_t = batch
        return {'loss': loss}

#     def training_end(self, outputs):
#         print(type(outputs))
# #         for x in outputs.items():
# #             print(type(x), x)
#         avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
#         print('')
#         print('EPOCH {:03d}, avg_loss: {:.4f}'.format(self.epoch, avg_loss))
#         return {'loss': avg_loss, 
#                 'progress_bar': {'loss', avg_loss}}
    
    def validation_step(self, batch, idx):
        img, gts = batch
        preds = self.forward(img)

        # get random inference batch
        if self.epoch == 1:
            if self.rints[1] is None:
                self.rints[1] = random.randint(0, self.num_patches[1]//img.shape[0])
            if idx == self.rints[1]:
                self.inf_batch_v = batch
#         if idx == self.rints[1]:
#             print(batch[0, :, 50, 50], self.inf_batch_t[0, :, 50, 50])
        return {'val_loss': self.loss(preds, gts)}

    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.epoch > 0:
            print('')
            print('Epoch {}, avg_val_loss: {:.4f}  '.format(self.epoch, avg_val_loss))
            if self.visualize:
                # TODO: pass val_loss and show on figure
                self.inference()
        self.epoch += 1
        return {'val_loss': avg_val_loss, 
                'progress_bar': {'val_loss': avg_val_loss}}
    
    # def test_step(self, batch, idx):
    #     img, gts = batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @pl.data_loader
    def train_dataloader(self):
        return self.data_loaders['train']

    @pl.data_loader
    def val_dataloader(self):
        return self.data_loaders['val']

    # @pl.data_loader
    # def test_dataloader(self):
    #     return self.data_loaders[2]


