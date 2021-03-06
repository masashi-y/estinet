import logging
from typing import Optional, Tuple

import wandb
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import estinet.utils as utils
from estinet.dataset import Addition, MNISTAddition
from estinet.nalu import StackedNALU

logger = logging.getLogger(__file__)


def make_argument_extractor_cnn():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 10, 5, stride=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=None),
        torch.nn.Conv2d(10, 20, 5, stride=1),
        torch.nn.ReLU(),
        # (batch_size, 20, 4, 4)
        torch.nn.MaxPool2d(kernel_size=2, stride=None),
        utils.Lambda(lambda x: x.view(-1, 320)),
        torch.nn.Linear(320, 10),
    )


class SumEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=10, hidden_size=50, num_layers=1)
        self.nalu = StackedNALU(n_layers=2, in_dim=50,
                                out_dim=1, hidden_dim=100)

    def forward(self, x):
        _, (hs, _) = self.rnn(x)
        # shape (hs): (batch_size, hidden_size)
        hs = torch.squeeze(hs, dim=0)
        preds = self.nalu(hs)
        # shape (hs): (batch_size, hidden_size)
        return preds.view((-1,))

    def loss(self, x, y):
        preds = self(x)
        loss = torch.nn.functional.mse_loss(preds, y)
        return loss


class EstiNet(torch.nn.Module):
    def __init__(self, sum_estimator, *, entropy_threshold, entropy_weight):
        super().__init__()
        self.sum_estimator = sum_estimator
        self.argument_extractor = make_argument_extractor_cnn()
        self.entropy_threshold = entropy_threshold
        self.entropy_weight = entropy_weight

    def forward(
        self,
        x: torch.FloatTensor,
        y: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Arguments:
            x {torch.Tensor} -- (arg_size, batch_size, 28, 28)
            y {torch.Tensor} -- (batch_size,)
        """
        arg_size, batch_size, _, _ = x.size()

        # shape (logits): (arg_size * batch_size, 10)
        logits = self.argument_extractor(x.view((-1, 1, 28, 28)))
        # shape (digits): (arg_size, batch_size, 10)
        digits = torch.softmax(logits, dim=1).view((arg_size, batch_size, 10))

        # shape (preds): (batch_size,)
        self.sum_estimator.requires_grad_(False)
        preds = self.sum_estimator(digits)
        if y is None:
            return preds, None

        # shape (sampled_digits): (arg_size, batch_size, 10)
        sampled_digits = digits.detach().requires_grad_(False)
        # shape (sampled_sum): (batch_size,)
        sampled_sum = sampled_digits.argmax(dim=2).float().sum(dim=0)
        entropy_term = torch.relu(
            utils.entropy(digits, dim=(0, 2)) - self.entropy_threshold
        ).mean()

        self.sum_estimator.requires_grad_(True)
        loss = (
            torch.nn.functional.mse_loss(preds, y)
            + self.sum_estimator.loss(sampled_digits, sampled_sum)
            - self.entropy_weight * entropy_term
        )

        return preds, loss

    def loss(
        self,
        x: torch.FloatTensor,
        y: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        _, loss = self(x, y)
        return loss

    def pred(
        self,
        x: torch.FloatTensor,
        use_estimator: bool = False
    ) -> torch.FloatTensor:

        if use_estimator:
            pred, _ = self(x)
            return pred
        arg_size, batch_size, _, _ = x.size()
        digits = self.argument_extractor(x.view((-1, 1, 28, 28))).argmax(dim=1)
        return digits.view((arg_size, batch_size)).sum(dim=0)

    def pred_mnist(self, x):
        return torch.log_softmax(self.argument_extractor(x), dim=1)


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
    model,
    data_loader,
    device,
    *,
    num_epochs,
    optimizer,
    test_fun=None,
    val_interval=100,
    clip_grad=None,
    clip_grad_norm=None,
    stage=None,
):
    optimizer = utils.optimizer_of(
        optimizer, [param for param in model.parameters()
                    if param.requires_grad]
    )
    for epoch in range(1, num_epochs + 1):
        if epoch % val_interval == 0 and test_fun is not None:
            test_fun()
        avg_loss = 0.0
        for step, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            model.zero_grad()
            loss = model.loss(x, y)
            logger.info(
                "stage: \"%s\" epoch %d loss of batch %d/%d: %f",
                stage or "",
                epoch,
                step,
                len(data_loader),
                loss.item(),
            )
            loss.backward()
            avg_loss += loss
            if clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
            elif clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm)
            optimizer.step()
            wandb.log({"epoch": epoch, "loss": loss.item()})
        epoch_loss = avg_loss / len(data_loader)
        wandb.log({"epoch": epoch, "epoch_loss": epoch_loss})
    if test_fun is not None:
        test_fun()


@hydra.main(config_path="config.yaml")
def main(cfg):
    logger.info(cfg.pretty())

    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.seed)

    device = utils.get_device(cfg.device)

    blackbox_estimator = SumEstimator().to(device)

    if cfg.pretrained.blackbox_estimator is not None:
        logger.info(
            "loading pretrained blackbox estimator at %s",
            cfg.pretrained.blackbox_estimator,
        )
        utils.load_model(
            blackbox_estimator,
            hydra.utils.to_absolute_path(cfg.pretrained.blackbox_estimator),
            device=device,
        )
    else:
        logger.info("start pretraining blackbox estimator")
        pretrain_data_loader = DataLoader(
            Addition(
                num_samples=cfg.addition.num_samples,
                arg_size=cfg.addition.arg_size
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        with wandb.init(project="estinet_pretrain"):
            train(
                blackbox_estimator,
                pretrain_data_loader,
                device,
                num_epochs=cfg.addition.num_epochs,
                optimizer=cfg.optimizer,
                val_interval=cfg.val_interval,
                stage="pretrain",
            )

    utils.save_model(blackbox_estimator, "./blackbox_estimator.th")

    wandb.init(
        project="estinet", config=dict(cfg), reinit=True,
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data_loader = DataLoader(
        MNISTAddition(
            "./data",
            train=True,
            download=cfg.download,
            transform=transform,
            num_samples=cfg.mnist.train.num_samples,
            arg_size=cfg.mnist.train.arg_size,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    test_small_loader = DataLoader(
        MNISTAddition(
            "./data",
            train=False,
            transform=transform,
            num_samples=cfg.mnist.test_small.num_samples,
            arg_size=cfg.mnist.test_small.arg_size,
        ),
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
    )

    test_large_loader = DataLoader(
        MNISTAddition(
            "./data",
            train=False,
            transform=transform,
            num_samples=cfg.mnist.test_large.num_samples,
            arg_size=cfg.mnist.test_large.arg_size,
        ),
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
    )

    mnist_test_loader = DataLoader(
        datasets.MNIST(
            "./data", train=False, download=cfg.download, transform=transform
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    estinet = EstiNet(
        blackbox_estimator, entropy_threshold=0.15, entropy_weight=0.15
    ).to(device)

    wandb.watch(estinet, log="all", log_freq=10)

    def mnist_test():
        correct = 0
        estinet.eval()
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x, y = x.to(device), y.to(device)
                output = estinet.pred_mnist(x)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()

        accuracy = 100.0 * correct / len(mnist_test_loader.dataset)
        logger.info(
            "MNIST test accuracy: %d/%d (%f%%)",
            correct,
            len(mnist_test_loader.dataset),
            accuracy,
        )
        return accuracy

    best_mae = float("inf")

    def compare_estinet_vs_nalu(data_loader, split):
        nonlocal best_mae
        error_estinet = 0.0
        error_nalu = 0.0
        estinet.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                estinet_pred = estinet.pred(x, use_estimator=False)
                error_estinet += (estinet_pred - y).abs().sum().item()
                nalu_pred = estinet.pred(x, use_estimator=True)
                error_nalu += (nalu_pred - y).abs().sum().item()
        mae_estinet = error_estinet / len(data_loader.dataset)
        mae_nalu = error_nalu / len(data_loader.dataset)
        logger.info(
            "Results on dataset with %d arg images",
            data_loader.dataset.arg_size
        )
        logger.info("Mean absolute error (Estinet): %f", mae_estinet)
        logger.info("Mean absolute error (NALU): %f", mae_nalu)

        if mae_estinet < best_mae:
            logger.info("The metric improved. The parameters are saved")
            utils.save_model(estinet, "./best_model.th")
            best_mae = mae_estinet
        accuracy = mnist_test()
        wandb.log(
            {
                f"{split}/mae_estinet": mae_estinet,
                f"{split}/mae_nalu": mae_nalu,
                f"{split}/accuracy": accuracy
            }
        )

    train(
        estinet,
        train_data_loader,
        device,
        test_fun=lambda: compare_estinet_vs_nalu(test_small_loader, "small"),
        num_epochs=cfg.mnist.num_epochs,
        optimizer=cfg.optimizer,
        val_interval=cfg.val_interval,
        stage="main",
    )

    logger.info("The final results using the best model")
    utils.load_model(estinet, "./best_model.th")
    compare_estinet_vs_nalu(test_small_loader, "small")
    compare_estinet_vs_nalu(test_large_loader, "large")


if __name__ == "__main__":
    main()
