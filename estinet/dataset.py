from typing import Tuple
import numpy as np
import torch
import torchvision
from PIL import Image

from estinet.utils import onehot


class Addition(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        num_samples: int,
        arg_size: int = 10
    ):
        self.data = torch.softmax(
            torch.randn(num_samples, arg_size, 10),
            dim=2
        )
        self.targets = torch.sum(self.data.argmax(dim=2), axis=1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.data[index], self.targets[index]


class MNISTAddition(torchvision.datasets.MNIST):
    def __init__(self, root, *, num_samples: int, arg_size: int = 10, **kwargs):
        super().__init__(root, **kwargs)
        self.num_samples = num_samples
        self.arg_size = arg_size
        self._initialize()

    def _initialize(self):
        """
        Transform MNIST dataset to MNISTAddition dataset, where training sample
        consists of a list of MNIST images of length num_samples (x) and the
        target is the sum of digits represented in the images.  the shape of
        self.data is (num_samples, arg_size, 28, 28).
        """
        data, self.data = self.data, []
        targets, self.targets = self.targets, []
        dataset = []
        for img, target in zip(data, targets):
            img = Image.fromarray(img.numpy(), mode="L")
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            dataset.append((img, target))

        for _ in range(self.num_samples):
            imgs, nums = zip(
                *(dataset[i] for i in np.random.randint(
                    0, len(dataset), self.arg_size))
            )
            self.data.append(torch.cat(list(imgs), dim=0))
            self.targets.append(sum(nums))
        self.data = torch.stack(self.data)

    def __getitem__(self, index):
        """
        Arguments:
            index {int} -- index

        Returns:
            Tuple[torch.Tensor, int] -- pair of (x, y).
            shape of the former is (arg_size, 28, 28)
        """
        imgs, target = self.data[index], float(self.targets[index])
        return imgs, target
