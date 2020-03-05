from histocartography.image.VorHoVerNet.dataset_readers.CoNSeP import CoNSeP
from histocartography.image.VorHoVerNet.dataset_readers.MoNuSeg import MoNuSeg
from histocartography.image.VorHoVerNet.dataset_readers.CoNuSeg import CoNuSeg
from histocartography.image.VorHoVerNet.dataset_readers.DownSampling import DownSampling

if __name__ == "__main__":
    c = CoNSeP(ver=1)
    c.read_image(1, 'test')
    c.read_labels(1, 'test')
    c.read_points(1, 'test')
    c.read_pseudo_labels(1, 'test')

    m = MoNuSeg(ver=1)
    m.read_image(1, 'test')
    m.read_labels(1, 'test')
    m.read_points(1, 'test')
    m.read_pseudo_labels(1, 'test')

    cm = CoNuSeg(ver=1)
    cm.read_image(1, 'test')
    cm.read_labels(1, 'test')
    cm.read_points(1, 'test')
    cm.read_pseudo_labels(1, 'test')
    cm.read_image(43, 'train')
    cm.read_labels(43, 'train')
    cm.read_points(43, 'train')
    cm.read_pseudo_labels(43, 'train')