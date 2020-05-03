import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from estinet.utils import onehot
from PIL import Image


class Addition(torch.utils.data.Dataset):
    def __init__(self, *, num_samples, arg_size=10):
        self.data = torch.randint(0, 10, size=(num_samples, arg_size))
        self.onehot_data = onehot(self.data, 10)
        self.targets = torch.sum(self.data, axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.onehot_data[index], self.targets[index]


class MNISTAddition(torchvision.datasets.MNIST):
    def __init__(self, root, *, num_samples, arg_size=10, **kwargs):
        super().__init__(root, **kwargs)
        self.num_samples = num_samples
        self.arg_size = arg_size
        self._initialize()

    def _initialize(self):
        """Transform MNIST dataset to MNISTAddition dataset, where training sample consists of
        a list of MNIST images of length num_samples (x) and the target is the sum of digits represented in the images.
        the shape of self.data is (num_samples, arg_size, 28, 28).
        """
        data, self.data = self.data, []
        targets, self.targets = self.targets, []
        dataset = []
        for img, target in zip(data, targets):
            img = Image.fromarray(img.numpy(), mode='L')
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            dataset.append((img, target))

        for _ in range(self.num_samples):
            imgs, nums = zip(*(dataset[i] for i in np.random.randint(0, len(dataset), self.arg_size)))
            self.data.append(torch.cat(list(imgs), dim=0))
            self.targets.append(sum(nums))
        self.data = torch.stack(self.data)

    def __getitem__(self, index):
        """
        Arguments:
            index {int} -- index

        Returns:
            Tuple[torch.Tensor, int] -- pair of (x, y). shape of the former is (arg_size, 28, 28)
        """
        imgs, target = self.data[index], float(self.targets[index])
        return imgs, target
