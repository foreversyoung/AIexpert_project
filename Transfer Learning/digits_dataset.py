from torch.utils.data import Dataset
import os
from PIL import Image

class MnistWithPrintWriterLabels(Dataset):

    def cached_loader(self, root):
        cache = dict()

        def load_internal(path, transform):
            if path in cache:
                return cache[path]

            with open(os.path.join(root, path), 'rb') as f:
                image = Image.open(f)
                image.load()
                if transform != None:
                    image = transform(image)
                cache[path] = image
                return image

        return load_internal

    def __init__(self, delegate, root, digit_transform, label_transform=lambda l: l):
        self.delegate = delegate
        self.root = root
        self.digit_transform = digit_transform
        self.label_transform = label_transform
        self.loader = self.cached_loader(root)

    def __getitem__(self, index):
        image, label = self.delegate.__getitem__(index)
        digit_name = self.label_transform(label)
        digit = self.loader("{:d}.png".format(digit_name), self.digit_transform)
        return image, digit

    def __len__(self):
        return self.delegate.__len__()
