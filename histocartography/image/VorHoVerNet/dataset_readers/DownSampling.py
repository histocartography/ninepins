import cv2

class DownSampling:
    def __init__(self, dataset, ratio=2, same_shape=True):
        self.dataset = dataset
        self.ratio = ratio
        self.same_shape = same_shape

    def __getattribute__(self, name):
        if name in ['dataset', 'ratio', 'same_shape', '_read_image', 'read_ori']:
            return super().__getattribute__(name)
        if name == "read_image":
            return self._read_image
        else:
            return getattr(self.dataset, name)

    def _read_image(self, *args, **kwargs):
        image = self.dataset.read_image(*args, **kwargs)
        h, w = image.shape[:2]
        th, tw = h // self.ratio, w // self.ratio
        image = cv2.resize(image, (tw, th))
        if self.same_shape:
            image = cv2.resize(image, (w, h))
        return image

    def read_ori(self, *args, **kwargs):
        return self.dataset.read_image(*args, **kwargs)

if __name__ == "__main__":
    from MoNuSeg import MoNuSeg
    from skimage.io import imsave
    d = MoNuSeg(ver=1, root='../MoNuSeg/')
    d = DownSampling(d)
    imsave("nodownsample.png", d.read_ori(2, 'test'))
    imsave("downsample.png", d.read_image(2, 'test'))
