from histocartography.image.VorHoVerNet.dataset_readers.CoNSeP import CoNSeP
from histocartography.image.VorHoVerNet.dataset_readers.MoNuSeg import MoNuSeg

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