import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import CoNSeP_cropped, data_reader
from model.vorhover_net import Net, CustomLoss


def inference(model, data_loader, gap=0):
    for idx, (img, gt) in enumerate(data_loader):
        # get prediction
        pred = model(img)
        
        # transpose and reshape
        img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        gt = gt.squeeze(0).permute(1, 2, 0)
        pred = pred.squeeze(0).detach().cpu().numpy()

        # create figure and show
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        plt.ion()
        for a in ax.ravel():
            a.axis('off')
        fig.suptitle('No.{} patch'.format(idx), fontsize=16)
        ax[0, 0].imshow(img.astype(np.uint8)[gap:gap+80, gap:gap+80, :])
        ax[0, 1].imshow(gt[..., 0] * 255)
        ax[0, 2].imshow(gt[..., 1] * 255)
        ax[0, 3].imshow(gt[..., 2] * 255)
        ax[1, 1].imshow(np.where(pred[..., 0] > 0.5, 255, 0))
        ax[1, 2].imshow(pred[..., 1] * 255)
        ax[1, 3].imshow(pred[..., 2] * 255)
        plt.draw()
        plt.waitforbuttonpress()
        plt.pause(0.3)
    plt.ioff()
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    # load model
    model = Net()
    checkpoint = torch.load('saver/model_5.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)
    
    # create test data loader
    from torch.utils.data import DataLoader
    test_data = CoNSeP_cropped(*data_reader(root='CoNSeP/', split='test', part=None))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # inference
    gap = (270 - 80) // 2
    inference(model, test_loader, gap=gap)
