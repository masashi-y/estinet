import hydra
import logging
from overrides import overrides

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from estinet.dataset import Addition, MNISTAddition
from estinet.nalu import StackedNALU
from estinet.utils import optimizer_of


logger = logging.getLogger(__file__)


class SumEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=10,
            hidden_size=50,
            num_layers=1)
        self.nalu = StackedNALU(
            n_layers=2,
            in_dim=50,
            out_dim=1,
            hidden_dim=100)

    @overrides
    def forward(self, x):
        _, (hs, _) = self.rnn(x)
        hs = torch.squeeze(hs, axis=0)
        preds = self.nalu(hs)
        return preds.view((-1,))

    def loss(self, x, y):
        preds = self.forward(x)
        loss = torch.nn.functional.mse_loss(preds, y)
        return loss


class EstiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.argument_extractor = nn.Sequential(
            nn.Conv2d(1, 10, 5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=0),
            nn.Conv2d(10, 20, 5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=0),
            nn.Linear(320, 10))

    @overrides
    def forward(self):
        pass


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


@hydra.main(config_path='config.yaml')
def main(cfg):
    logger.info(cfg.pretty())

    # MNISTAddition('./data', num_samples=cfg.addition.num_samples, arg_size=cfg.addition.arg_size)
    dataset = Addition(
        num_samples=cfg.addition.num_samples,
        arg_size=cfg.addition.arg_size)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn)
    model = SumEstimator()

    optimizer = optimizer_of(
        cfg.optimizer,
        [param for param in model.parameters() if param.requires_grad]
    )
    for epoch in range(1, cfg.num_epochs + 1):
        logger.info('epoch %d', epoch)
        # train_logger.update_epoch()
        # val_logger.update_epoch()
        if epoch % cfg.val_interval == 0:
            pass  # validation
        avg_loss, count = 0, 0
        for x, y in data_loader:
            model.train()
            model.zero_grad()
            loss = model.loss(x, y)
            logger.info('loss of batch %d/%d: %f', count, len(data_loader), loss.item())
            loss.backward()
            avg_loss += loss
            count += 1
            # if cfg.clip_grad:
            #     nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad)
            # elif cfg.clip_grad_norm:
            #     nn.utils.clip_grad_norm_(
            #         model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
        # train_logger.plot_obj_val((avg_loss / count).item())
    # validation

if __name__ == "__main__":
    main()