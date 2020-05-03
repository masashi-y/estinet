import hydra
import logging
import numpy as np
from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from estinet.dataset import Addition, MNISTAddition
from estinet.nalu import StackedNALU
import estinet.utils as utils


logger = logging.getLogger(__file__)


class SumEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=10,
            hidden_size=50,
            num_layers=1,
            dropout=0.5)
        self.nalu = StackedNALU(
            n_layers=2,
            in_dim=50,
            out_dim=1,
            hidden_dim=100)

    @overrides
    def forward(self, x):
        _, (hs, _) = self.rnn(x)
        hs = torch.squeeze(hs, dim=0)
        preds = self.nalu(hs)
        return preds.view((-1,))

    def loss(self, x, y):
        preds = self.forward(x)
        loss = F.mse_loss(preds, y)
        return loss


class EstiNet(nn.Module):
    def __init__(self, sum_estimator, *, entropy_threshold, entropy_weight):
        super().__init__()
        self.sum_estimator = sum_estimator
        self.argument_extractor = nn.Sequential(
            nn.Conv2d(1, 10, 5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=0),
            nn.Dropout2d(0.25),
            nn.Conv2d(10, 20, 5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=0),   # (batch_size, 20, 4, 4)
            nn.Dropout2d(0.50),
            utils.Lambda(lambda x: x.view(-1, 320)),
            nn.Linear(320, 10))
        self.entropy_threshold = entropy_threshold
        self.entropy_weight = entropy_weight

    @overrides
    def forward(self, x, y=None):
        """
        Arguments:
            x {torch.Tensor} -- (arg_size, batch_size, 28, 28)
            y {torch.Tensor} -- (batch_size,)
        """
        arg_size, batch_size, _, _ = x.size()
        logits = self.argument_extractor(x.view((-1, 1, 28, 28)))
        digits = F.gumbel_softmax(logits, tau=1).view((arg_size, batch_size, 10))
        preds = self.sum_estimator(digits)
        if y is None:
            return preds, None

        sampled_sum = digits.argmax(dim=2).sum(dim=0) \
                    .float().detach().requires_grad_(False)
        entropy_term = torch.relu(utils.entropy(digits, dim=(0, 2)) - self.entropy_threshold)
        loss = F.mse_loss(preds, y) \
            + self.sum_estimator.loss(digits, sampled_sum) \
            + self.entropy_weight * entropy_term
        return preds, loss.mean()

    def loss(self, x, y):
        _, loss = self(x, y)
        return loss

    def pred(self, x, use_estimator=False):
        if use_estimator:
            pred, _ = self(x)
            return pred
        arg_size, batch_size, _, _ = x.size()
        digits = self.argument_extractor(x.view((-1, 1, 28, 28))).argmax(dim=1)
        return digits.view((arg_size, batch_size)).sum(dim=0)


def collate_fn(batch):
    """
    Arguments:
        batch {List[Tuple[torch.Tensor, int]]} -- a list of dataset examples

    Returns:
        torch.Tensor -- (arg size, batch size, 28, 28)
        torch.Tensor -- (batch size,)
        List[Dict[str, str]] -- list of meta data
    """
    xs, ys = zip(*batch)
    xs = torch.nn.utils.rnn.pad_sequence(xs)
    ys = torch.tensor(list(ys), dtype=torch.float)
    return xs, ys


def train(
        model, data_loader, device, *,
        num_epochs, optimizer,
        test_fun=None,
        val_interval=100,
        clip_grad=None,
        clip_grad_norm=None):
    optimizer = utils.optimizer_of(
        optimizer,
        [param for param in model.parameters() if param.requires_grad])
    for epoch in range(1, num_epochs + 1):
        if epoch % val_interval == 0 and test_fun is not None:
            test_fun()
        avg_loss = 0.
        for count, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            model.zero_grad()
            loss = model.loss(x, y)
            logger.info('epoch %d loss of batch %d/%d: %f',
                        epoch, count, len(data_loader), loss.item())
            loss.backward()
            avg_loss += loss
            count += 1
            if clip_grad:
                nn.utils.clip_grad_value_(model.parameters(), clip_grad)
            elif clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm)
            optimizer.step()
    if test_fun is not None:
        test_fun()


@hydra.main(config_path='config.yaml')
def main(cfg):
    logger.info(cfg.pretty())

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    device = utils.get_device(cfg.device)

    blackbox_estimator = SumEstimator().to(device)

    if cfg.pretrained.blackbox_estimator is not None:
        logger.info('loading pretrained blackbox estimator at %s',
                    cfg.pretrained.blackbox_estimator)
        utils.load_model(
            blackbox_estimator, cfg.pretrained.blackbox_estimator)
    else:
        logger.info('start pretraining blackbox estimator')
        pretrain_data_loader = DataLoader(
            Addition(
                num_samples=cfg.addition.num_samples,
                arg_size=cfg.addition.arg_size),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn)
        train(
            blackbox_estimator,
            pretrain_data_loader,
            device,
            num_epochs=cfg.addition.num_epochs,
            optimizer=cfg.optimizer,
            val_interval=cfg.val_interval)

    utils.save_model(blackbox_estimator, './blackbox_estimator.th')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_data_loader = DataLoader(
        MNISTAddition(
            './data',
            train=True,
            download=cfg.download,
            transform=transform,
            num_samples=cfg.mnist.train.num_samples,
            arg_size=cfg.mnist.train.arg_size),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn)

    test_small_loader = DataLoader(
        MNISTAddition(
            './data',
            train=False,
            transform=transform,
            num_samples=cfg.mnist.test_small.num_samples,
            arg_size=cfg.mnist.test_small.arg_size),
        batch_size=cfg.batch_size,
        collate_fn=collate_fn)

    test_large_loader = DataLoader(
        MNISTAddition(
            './data',
            train=False,
            transform=transform,
            num_samples=cfg.mnist.test_large.num_samples,
            arg_size=cfg.mnist.test_large.arg_size),
        batch_size=cfg.batch_size,
        collate_fn=collate_fn)

    estinet = EstiNet(
        blackbox_estimator,
        entropy_threshold=0.15,
        entropy_weight=0.15).to(device)

    def compare_estinet_vs_nalu(data_loader, msg):
        error_estinet = 0.
        error_nalu = 0.
        total = 0
        estinet.eval()
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            estinet_pred = estinet.pred(x, use_estimator=False)
            error_estinet += (estinet_pred - y).abs().sum().item()
            nalu_pred = estinet.pred(x, use_estimator=True)
            error_nalu += (nalu_pred - y).abs().sum().item()
            total += y.numel()
        mae_estinet = error_estinet / total
        mae_nalu = error_nalu / total
        logger.info(msg)
        logger.info('Mean absolute error (Estinet): %f', mae_estinet)
        logger.info('Mean absolute error (NALU): %f', mae_nalu)

    train(
        estinet,
        train_data_loader,
        device,
        test_fun=lambda: compare_estinet_vs_nalu(
            test_small_loader,
            f'Results on dataset with {cfg.mnist.test_small.arg_size} arg images'),
        num_epochs=cfg.mnist.num_epochs,
        optimizer=cfg.optimizer,
        val_interval=cfg.val_interval)

    compare_estinet_vs_nalu(
        test_large_loader,
        f'Results on dataset with {cfg.mnist.test_large.arg_size} arg images')


if __name__ == "__main__":
    main()
