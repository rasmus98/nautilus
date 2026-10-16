"""Optional PyTorch implementations of the host-side numerical kernels.

This module is imported only when a sampler is created with
``backend='torch'``. It provides drop-in replacements for the emulator
training and prediction. Results are statistically, not bitwise, equivalent
to the default numpy/scikit-learn backend.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor

try:
    import torch
except ImportError as error:
    raise ImportError(
        "backend='torch' requires PyTorch. Install it with "
        "'pip install nautilus-sampler[torch]'.") from error

# Network weights and activations. Ellipsoid algebra stays in float64.
NN_DTYPE = torch.float32
# Maximum number of points passed through the networks at once.
PREDICT_CHUNK = 65536


def train_networks(x, y, n_networks, neural_network_kwargs, device):
    """Train an ensemble of networks like `MLPRegressor` with Adam.

    All networks are trained simultaneously as one batched model. Each network
    has its own initialization and shuffling and stops after its own
    `n_iter_no_change` criterion, mirroring the scikit-learn defaults used by
    nautilus (ReLU, squared loss, constant learning rate, no early stopping).

    Parameters
    ----------
    x : numpy.ndarray
        Normalized input coordinates.
    y : numpy.ndarray
        Target values.
    n_networks : int
        Number of networks.
    neural_network_kwargs : dict
        Keyword arguments for `MLPRegressor`. Only the Adam/ReLU subset is
        supported.
    device : str
        Torch device.

    Returns
    -------
    networks : list of sklearn.neural_network.MLPRegressor
        Fitted networks holding the trained weights, such that predictions,
        checkpoints and pickling work as for the default backend.

    """
    network = MLPRegressor(**neural_network_kwargs)
    unsupported = {
        key: value for key, value in dict(
            activation='relu', solver='adam', learning_rate='constant',
            early_stopping=False, shuffle=True, warm_start=False).items()
        if getattr(network, key) != value}
    if unsupported:
        raise ValueError("The torch backend does not support the network " +
                         "settings {}.".format(unsupported))

    device = torch.device(device)
    n_points, n_dim = x.shape
    units = [n_dim] + list(np.atleast_1d(network.hidden_layer_sizes)) + [1]
    if network.batch_size == 'auto':
        batch_size = min(200, n_points)
    else:
        batch_size = int(np.clip(network.batch_size, 1, n_points))

    # Weights and biases of all networks, stacked along the first axis.
    gen = torch.Generator(device=device).manual_seed(0)
    params = []
    for fan_in, fan_out in zip(units[:-1], units[1:]):
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        for shape in [(fan_in, fan_out), (1, fan_out)]:
            params.append(torch.empty(
                (n_networks, ) + shape, dtype=NN_DTYPE, device=device).uniform_(
                -bound, bound, generator=gen).requires_grad_(True))

    def forward(h):
        for k in range(0, len(params), 2):
            h = torch.baddbmm(params[k + 1], h, params[k])
            if k < len(params) - 2:
                h = torch.relu(h)
        return h[..., 0]

    x = torch.as_tensor(x, dtype=NN_DTYPE, device=device)
    y = torch.as_tensor(y, dtype=NN_DTYPE, device=device)
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    active = torch.ones((n_networks, 1, 1), dtype=NN_DTYPE, device=device)
    lr, beta_1, beta_2 = (network.learning_rate_init, network.beta_1,
                          network.beta_2)
    alpha, eps = network.alpha, network.epsilon

    n_iter = np.zeros(n_networks, dtype=int)
    best_loss = np.full(n_networks, np.inf)
    no_improvement = np.zeros(n_networks, dtype=int)
    loss_curve = [[] for i in range(n_networks)]
    t = 0

    for epoch in range(network.max_iter):
        perm = torch.argsort(torch.rand(
            (n_networks, n_points), generator=gen, device=device), dim=1)
        x_perm, y_perm = x[perm], y[perm]
        loss_sum = torch.zeros(n_networks, dtype=NN_DTYPE, device=device)
        for start in range(0, n_points, batch_size):
            stop = min(start + batch_size, n_points)
            y_pred = forward(x_perm[:, start:stop])
            loss = 0.5 * torch.mean(
                (y_pred - y_perm[:, start:stop])**2, dim=1)
            if alpha > 0:
                loss = loss + 0.5 * alpha * sum(
                    torch.sum(w**2, dim=(1, 2)) for w in params[::2]) / (
                        stop - start)
            grad = torch.autograd.grad(loss.sum(), params)
            # Adam exactly as in scikit-learn, frozen for stopped networks.
            t += 1
            with torch.no_grad():
                torch._foreach_mul_(m, beta_1)
                torch._foreach_add_(m, grad, alpha=1 - beta_1)
                torch._foreach_mul_(v, beta_2)
                torch._foreach_addcmul_(v, grad, grad, value=1 - beta_2)
                denom = torch._foreach_sqrt(v)
                torch._foreach_add_(denom, eps)
                lr_t = lr * np.sqrt(1 - beta_2**t) / (1 - beta_1**t)
                torch._foreach_addcdiv_(
                    params, torch._foreach_mul(m, [active] * len(m)), denom,
                    value=-lr_t)
                loss_sum += loss.detach() * (stop - start)

        loss_epoch = (loss_sum / n_points).tolist()
        for i in np.flatnonzero(active[:, 0, 0].cpu().numpy()):
            n_iter[i] += 1
            loss_curve[i].append(loss_epoch[i])
            if loss_epoch[i] > best_loss[i] - network.tol:
                no_improvement[i] += 1
            else:
                no_improvement[i] = 0
            best_loss[i] = min(best_loss[i], loss_epoch[i])
            if no_improvement[i] > network.n_iter_no_change:
                active[i] = 0
        if not torch.any(active):
            break

    weights = [p.detach().to('cpu', torch.float64).numpy() for p in params]
    networks = []
    for i in range(n_networks):
        network = MLPRegressor(**neural_network_kwargs)
        network.coefs_ = [w[i] for w in weights[::2]]
        network.intercepts_ = [b[i, 0] for b in weights[1::2]]
        network.n_features_in_ = n_dim
        network.n_outputs_ = 1
        network.n_layers_ = len(units)
        network.out_activation_ = 'identity'
        network.n_iter_ = n_iter[i]
        network.t_ = n_iter[i] * n_points
        network.loss_ = loss_curve[i][-1]
        network.best_loss_ = best_loss[i]
        network.loss_curve_ = loss_curve[i]
        networks.append(network)

    return networks


def predict(emulator, x):
    """Average prediction of an emulator's networks on a torch device.

    Parameters
    ----------
    emulator : nautilus.neural.NeuralNetworkEmulator
        Emulator whose `device` is a torch device.
    x : numpy.ndarray or torch.Tensor
        Input coordinates (not normalized).

    Returns
    -------
    y_emu : numpy.ndarray or torch.Tensor
        Mean prediction, of the same type as `x`.

    """
    device = torch.device(emulator.device)
    if getattr(emulator, '_torch_params', None) is None:
        params = []
        for k in range(len(emulator.neural_networks[0].coefs_)):
            params.append(torch.as_tensor(np.stack(
                [nn.coefs_[k] for nn in emulator.neural_networks]),
                dtype=NN_DTYPE, device=device))
            params.append(torch.as_tensor(np.stack(
                [nn.intercepts_[k] for nn in emulator.neural_networks])[
                    :, np.newaxis], dtype=NN_DTYPE, device=device))
        emulator._torch_params = params
        emulator._torch_norm = [torch.as_tensor(
            a, dtype=torch.float64, device=device) for a in
            (emulator.mean, emulator.scale)]
    params = emulator._torch_params
    n_networks, n_dim, n_hidden = params[0].shape

    is_numpy = not isinstance(x, torch.Tensor)
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    y = torch.empty(len(x), dtype=NN_DTYPE, device=device)
    with torch.inference_mode():
        for start in range(0, len(x), PREDICT_CHUNK):
            h = ((x[start:start + PREDICT_CHUNK] - emulator._torch_norm[0]) /
                 emulator._torch_norm[1]).to(NN_DTYPE)
            # First layer of all networks as a single matrix product.
            h = (h @ params[0].permute(1, 0, 2).reshape(n_dim, -1)).view(
                len(h), n_networks, n_hidden).transpose(0, 1) + params[1]
            for k in range(2, len(params), 2):
                h = torch.baddbmm(params[k + 1], torch.relu(h), params[k])
            y[start:start + len(h[0])] = torch.mean(h[..., 0], dim=0)
    if is_numpy:
        return y.cpu().numpy().astype(np.float64)
    return y.double()

