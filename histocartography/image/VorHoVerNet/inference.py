import os
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.io import imsave
from histocartography.image.VorHoVerNet.dataset import CoNSeP_cropped, data_reader
from histocartography.image.VorHoVerNet.model.vorhover_net import CustomLoss, Net
from histocartography.image.VorHoVerNet.utils import scale, shift_and_scale


def inference(model, data_loader, figpath_fix='', gap=None, psize=270, vsize=80):
    """
    Run massive inference on given model with provided data.
    Combine the output with annotations to form a plot for comparison.
    Args:
        model (torch.nn.Module): the model.
        data_loader (torch.nn.DataLoader): the data loader to retrieve data from.
        figpath_fix (str): the name of the folder to store the figures.
        gap (int): the offset between the origins of input image and output.
        psize (int): the size of input image patch. If gap is set, this is ignored.
        vsize (int): the size of output patch. If gap is set, this is ignored.
    """
    import matplotlib
    matplotlib.use('Agg')

    os.makedirs('./inference/{}'.format(figpath_fix), exist_ok=True)
    
    gap = (psize - vsize) // 2 if gap is None else gap
    fig, ax = plt.subplots(4, 4, figsize=(12, 9))
    # plt.ion()
    for idx, (img, gt) in enumerate(data_loader):
        print('Current patch: {:04d}'.format(idx + 1), end='\r')

        filename = './inference/{}/patch{:04d}.png'.format(figpath_fix, idx + 1)
        if os.path.isfile(filename):
            continue

        # get prediction
        with torch.no_grad():
            pred = model(img)
        
        # transpose and reshape
        img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        gt = gt.squeeze(0).permute(1, 2, 0)
        pred = pred.squeeze(0).detach().cpu().numpy()

        for a in ax.ravel():
            a.axis('off')
        fig.suptitle('Patch {}'.format(idx + 1), fontsize=16)
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
        # plt.draw()
        # plt.waitforbuttonpress()
        plt.savefig(filename, dpi=120)
        # plt.pause(1.0)
        # plt.cla()
    # plt.ioff()
    # plt.show()
    plt.close()

def inference_without_plot(model, data_loader, figpath_fix='', gap=None, psize=270, vsize=80):
    """
    Run massive inference on given model with provided data.
    Save individual result into files.
    Args:
        model (torch.nn.Module): the model.
        data_loader (torch.nn.DataLoader): the data loader to retrieve data from.
        figpath_fix (str): the name of the folder to store the figures.
        gap (int): the offset between the origins of input image and output.
        psize (int): the size of input image patch. If gap is set, this is ignored.
        vsize (int): the size of output patch. If gap is set, this is ignored.
    """
    
    os.makedirs('./inference/{}'.format(figpath_fix), exist_ok=True)

    subs = ["seg", "dist1", "dist2"]
    pngsubs = subs + list(map(lambda x: x+"_gt", subs))
    npysubs = subs

    savedir = './inference/{}/patch{:04d}'

    gap = (psize - vsize) // 2 if gap is None else gap
    for idx, (img, gt) in enumerate(data_loader):
        print('Current patch: {:04d}'.format(idx + 1), end='\r')

        imgdir = savedir.format(figpath_fix, idx + 1)
        os.makedirs(imgdir, exist_ok=True)
        filename = imgdir + '/{}.png'
        filename_npy = imgdir + '/{}.npy'

        if reduce(lambda x, y: x and os.path.isfile(filename.format(y)), pngsubs, True) and \
            reduce(lambda x, y: x and os.path.isfile(filename_npy.format(y)), npysubs, True):
            continue

        # get prediction
        with torch.no_grad():
            # pred = model(img * 255) # for old checkpoints that were trained with 0-255 images
            pred = model(img)
        
        # transpose and reshape
        img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        gt = gt.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        pred = pred.squeeze(0).detach().cpu().numpy()

        gt0 = shift_and_scale(gt[..., 0], 255, 0).astype(np.uint8)
        gt1 = shift_and_scale(gt[..., 1], 255, 0).astype(np.uint8)
        gt2 = shift_and_scale(gt[..., 2], 255, 0).astype(np.uint8)

        imsave(filename.format("seg_gt"), gt0)
        imsave(filename.format("dist1_gt"), gt1)
        imsave(filename.format("dist2_gt"), gt2)
        ori = (img * 255).astype(np.uint8)[gap:gap+vsize, gap:gap+vsize, :]
        imsave(filename.format("ori"), ori)
        pred0 = shift_and_scale(pred[..., 0], 255, 0).astype(np.uint8)
        pred1 = shift_and_scale(pred[..., 1], 255, 0).astype(np.uint8)
        pred2 = shift_and_scale(pred[..., 2], 255, 0).astype(np.uint8)

        imsave(filename.format("seg"), pred0)
        imsave(filename.format("dist1"), pred1)
        imsave(filename.format("dist2"), pred2)
        np.save(filename_npy.format("seg"), pred[..., 0])
        np.save(filename_npy.format("dist1"), pred[..., 1])
        np.save(filename_npy.format("dist2"), pred[..., 2])

if __name__ == '__main__':
    # load model
    model_name = 'model_009_ckpt_epoch_18.ckpt'
    checkpoint = torch.load('savers_pl/{}'.format(model_name), map_location=torch.device('cpu'))
    print('model_name: {}'.format(model_name))
    model = Net()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # create test data loader
    from torch.utils.data import DataLoader
    test_data = CoNSeP_cropped(*data_reader(root='CoNSeP/', split='train', doflip=False, contain_both=True, part=None))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # inference
    inference_without_plot(model, test_loader, figpath_fix=model_name.split('.')[-2])
    # inference(model, test_loader, figpath_fix=model_name.split('.')[-2])
