import torch


EPS = 1e-6


def get_device(gpu_id):
    if gpu_id is not None and gpu_id >= 0:
        return torch.device('cuda', gpu_id)
    return torch.device('cpu')


def onehot(x, n):
    x0 = x.view(-1, 1)
    x1 = x.new_zeros(len(x0), n, dtype=torch.float)
    x1.scatter_(1, x0, 1)
    return x1.view(x.size() + (n,))


def entropy(x, dim=-1):
    return - (x * torch.log(x + EPS)).sum(dim=dim)


def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model(model, file_path):
    with open(file_path, 'rb') as f:
        model.load_state_dict(torch.load(f))


__optimizers = {
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'lbfgs': torch.optim.LBFGS,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'adamax': torch.optim.Adamax,
    'asgd': torch.optim.ASGD,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
    'rprop': torch.optim.Rprop,
}


def optimizer_of(string, params):
    """Returns a optimizer object based on input string, e.g., adagrad(lr=0.01, lr_decay=0)
    Arguments:
        string {str} -- string expression of an optimizer
        params {List[torch.Tensor]} -- parameters to learn

    Returns:
        torch.optim.Optimizer -- optimizer
    """
    index = string.find('(')
    if index >= 0:
        assert string[-1] == ')', f'invalid format for the optimizer string: {string}'
    else:
        string += '()'
        index = -2
    try:
        optim_class = __optimizers[string[:index]]
    except KeyError as e:
        raise Exception(
            f'Optimizer class "{string[:index]}" does not exist.\n'
            f'Please choose one among: {list(__optimizers.keys())}'
        ) from e
    kwargs = eval(f'dict{string[index:]}')
    return optim_class(params, **kwargs)


def random_probabilities(batch_size, n, m, device=None, requires_grad=True):
    """returns a tensor with shape (batch size, n, m), that sums to one at the last dimension

    Arguments:
        batch_size {int} -- batch size
        n, m {int} -- number of items

    Returns:
        torch.Tensor -- (batch size, n, m)
    """
    x = torch.rand(n, m + 1, device=device)
    x[:, 0] = 0.
    x[:, -1] = 1.
    x, _ = x.sort(1)
    return (x[:, 1:] - x[:, :-1]).unsqueeze(0) \
                                    .expand(batch_size, -1, -1) \
                                    .contiguous() \
                                    .requires_grad_(requires_grad)
