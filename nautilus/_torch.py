"""Optional PyTorch implementations of the host-side numerical kernels.

This module is imported only when a sampler is created with
``backend='torch'``. It provides drop-in replacements for the emulator
training and prediction and for drawing points from, and testing membership
in, the neural network-based bounds. Results are statistically, not bitwise,
equivalent to the default numpy/scikit-learn backend.
"""

from functools import partial

import numpy as np
from scipy.special import logsumexp
from sklearn.neural_network import MLPRegressor

from .bounds.basic import Ellipsoid
from .bounds.union import oversample_size

try:
    import torch
except ImportError as error:
    raise ImportError(
        "backend='torch' requires PyTorch. Install it with "
        "'pip install nautilus-sampler[torch]'.") from error

# Network weights and activations. Initial weights are drawn in float64 and
# cast, so float32 and float64 training start from the same values.
NN_DTYPE = torch.float64
# Maximum number of points passed through the networks at once.
PREDICT_CHUNK = 65536
# Maximum number of minibatch steps per CUDA graph. An epoch is captured as
# consecutive graphs sharing one memory pool, which bounds the pool to the
# temporaries of this many steps instead of a whole epoch.
GRAPH_STEPS = 16
# Training epochs between host checks of whether all networks stopped, on
# CUDA devices, where each check waits for the device. Stopped networks are
# frozen, so the extra epochs do not change the result.
CHECK_EVERY = 4


def train_networks(x, y, n_networks, neural_network_kwargs, device, seed=0):
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
    seed : int, optional
        Seed of the initial weights and the shuffling. Default is 0.

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
    gen = torch.Generator(device=device).manual_seed(seed)
    params = []
    for fan_in, fan_out in zip(units[:-1], units[1:]):
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        for shape in [(fan_in, fan_out), (1, fan_out)]:
            params.append(torch.empty(
                (n_networks, ) + shape, dtype=torch.float64,
                device=device).uniform_(-bound, bound, generator=gen).to(
                NN_DTYPE).requires_grad_(True))

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
    lr, beta_1, beta_2 = (network.learning_rate_init, network.beta_1,
                          network.beta_2)
    alpha, eps = network.alpha, network.epsilon
    slices = [(start, min(start + batch_size, n_points)) for start in
              range(0, n_points, batch_size)]

    # State read and written by an epoch, kept on the device in fixed buffers
    # such that an epoch can be replayed as a CUDA graph.
    x_perm = torch.empty((n_networks, n_points, n_dim), dtype=NN_DTYPE,
                         device=device)
    y_perm = torch.empty((n_networks, n_points), dtype=NN_DTYPE,
                         device=device)
    lr_epoch = torch.empty(len(slices), dtype=NN_DTYPE, device=device)
    active = torch.ones(n_networks, dtype=torch.bool, device=device)
    best_loss = torch.full((n_networks, ), np.inf, dtype=NN_DTYPE,
                           device=device)
    no_improvement = torch.zeros(n_networks, dtype=torch.long, device=device)
    loss_sum = torch.empty(n_networks, dtype=NN_DTYPE, device=device)
    loss_epoch = torch.empty(n_networks, dtype=NN_DTYPE, device=device)
    active_curve = torch.zeros((network.max_iter, n_networks),
                               dtype=torch.bool, device=device)
    loss_curve = torch.zeros((network.max_iter, n_networks), dtype=NN_DTYPE,
                             device=device)

    def steps(k_0, k_1):
        # Adam exactly as in scikit-learn, frozen for stopped networks.
        mask = active.to(NN_DTYPE)[:, None, None]
        if k_0 == 0:
            loss_sum.zero_()
        for k in range(k_0, k_1):
            start, stop = slices[k]
            y_pred = forward(x_perm[:, start:stop])
            loss = 0.5 * torch.mean(
                (y_pred - y_perm[:, start:stop])**2, dim=1)
            if alpha > 0:
                loss = loss + 0.5 * alpha * sum(
                    torch.sum(w**2, dim=(1, 2)) for w in params[::2]) / (
                        stop - start)
            grad = torch.autograd.grad(loss.sum(), params)
            with torch.no_grad():
                torch._foreach_mul_(m, beta_1)
                torch._foreach_add_(m, grad, alpha=1 - beta_1)
                torch._foreach_mul_(v, beta_2)
                torch._foreach_addcmul_(v, grad, grad, value=1 - beta_2)
                denom = torch._foreach_sqrt(v)
                torch._foreach_add_(denom, eps)
                step = torch._foreach_mul(m, [mask * lr_epoch[k]] * len(m))
                torch._foreach_div_(step, denom)
                torch._foreach_sub_(params, step)
                loss_sum.add_(loss.detach() * (stop - start))
        if k_1 < len(slices):
            return
        # Early stopping as in scikit-learn (tol, n_iter_no_change).
        with torch.no_grad():
            loss_epoch.copy_(loss_sum / n_points)
            no_improvement.copy_(torch.where(active, torch.where(
                loss_epoch > best_loss - network.tol, no_improvement + 1, 0),
                no_improvement))
            best_loss.copy_(torch.where(
                active, torch.fmin(best_loss, loss_epoch), best_loss))
            active.logical_and_(no_improvement <= network.n_iter_no_change)

    epoch = [partial(steps, k, min(k + GRAPH_STEPS, len(slices))) for k in
             range(0, len(slices), GRAPH_STEPS)]
    replay = None
    check_every = CHECK_EVERY if device.type == 'cuda' else 1
    for epoch_0 in range(0, network.max_iter, check_every):
        n_epochs = min(check_every, network.max_iter - epoch_0)
        t = np.arange(epoch_0 * len(slices) + 1,
                      (epoch_0 + n_epochs) * len(slices) + 1)
        lr_block = torch.as_tensor(
            lr * np.sqrt(1 - beta_2**t) / (1 - beta_1**t), dtype=NN_DTYPE,
            device=device).view(n_epochs, len(slices))
        for i in range(epoch_0, epoch_0 + n_epochs):
            perm = torch.argsort(torch.rand(
                (n_networks, n_points), generator=gen, device=device), dim=1)
            torch.index_select(x, 0, perm.view(-1),
                               out=x_perm.view(-1, n_dim))
            torch.index_select(y, 0, perm.view(-1), out=y_perm.view(-1))
            lr_epoch.copy_(lr_block[i - epoch_0])
            active_curve[i].copy_(active)
            if replay is not None:
                replay()
            elif device.type == 'cuda':
                replay = capture(epoch)
            else:
                for function in epoch:
                    function()
            loss_curve[i].copy_(loss_epoch)
        if not torch.any(active):
            break

    active_curve = active_curve.cpu().numpy()
    loss_curve = loss_curve.cpu().numpy()
    best_loss = best_loss.cpu().numpy()
    n_iter = np.sum(active_curve, axis=0)
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
        network.loss_curve_ = loss_curve[active_curve[:, i], i].tolist()
        network.loss_ = network.loss_curve_[-1]
        network.best_loss_ = best_loss[i]
        networks.append(network)

    return networks


def capture(functions):
    """Run functions once and return a replay of them as CUDA graphs.

    The first call runs eagerly on a side stream, which also initializes
    autograd and the BLAS handles, as required before capture. Each function
    is then captured as its own graph. The graphs share one memory pool and
    are always replayed in order. The functions must read and write only
    tensors that persist between calls.

    Parameters
    ----------
    functions : list of callable
        Functions without arguments, launching only CUDA kernels.

    Returns
    -------
    replay : callable
        Replays the captured kernels of all functions in order.

    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for function in functions:
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graphs = []
    for function in functions:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graphs[0].pool() if graphs else
                              None):
            function()
        graphs.append(graph)

    def replay():
        for graph in graphs:
            graph.replay()

    return replay


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



def _params(bound, device, keys):
    """Return attributes of a bound as tensors, cached on the bound."""
    cache = bound.__dict__.setdefault('_torch_cache', {})
    if keys not in cache:
        cache[keys] = [torch.as_tensor(getattr(bound, key), device=device)
                       for key in keys]
    return cache[keys]


def _transform(ellipsoid, x):
    """`Ellipsoid.transform` for a tensor."""
    c, B_inv = _params(ellipsoid, x.device, ('c', 'B_inv'))
    return (x - c) @ B_inv.T


def _contains(bound, x):
    """`contains` of an ellipsoid or a cube-ellipsoid mixture for a tensor."""
    if isinstance(bound, Ellipsoid):
        return torch.sum(_transform(bound, x)**2, dim=-1) < 1
    in_bound = torch.ones(len(x), dtype=torch.bool, device=x.device)
    dim_cube, = _params(bound, x.device, ('dim_cube', ))
    if bound.cube is not None:
        x_cube = x[:, dim_cube]
        in_bound &= torch.all((x_cube >= 0) & (x_cube < 1), dim=-1)
    if bound.ellipsoid is not None:
        in_bound &= _contains(bound.ellipsoid, x[:, ~dim_cube])
    return in_bound


def _sample(bound, n_points, gen):
    """`sample` of an ellipsoid or a cube-ellipsoid mixture on a device."""
    device = gen.device
    if isinstance(bound, Ellipsoid):
        c, B = _params(bound, device, ('c', 'B'))
        x = torch.randn((n_points, bound.n_dim), generator=gen,
                        dtype=torch.float64, device=device)
        x /= torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        x *= torch.rand((n_points, 1), generator=gen, dtype=torch.float64,
                        device=device)**(1.0 / bound.n_dim)
        return x @ B.T + c
    x = torch.empty((n_points, bound.n_dim), dtype=torch.float64,
                    device=device)
    dim_cube, = _params(bound, device, ('dim_cube', ))
    if bound.cube is not None:
        x[:, dim_cube] = torch.rand(
            (n_points, int(np.sum(bound.dim_cube))), generator=gen,
            dtype=torch.float64, device=device)
    if bound.ellipsoid is not None:
        x[:, ~dim_cube] = _sample(bound.ellipsoid, n_points, gen)
    return x


def _union_sample(union, n_points, gen):
    """One pass of `Union.sample` on a device, keeping no points in reserve.

    Returns the accepted points of `n_points` draws and updates the union's
    sampling counters.
    """
    p = np.exp(np.array(union.log_v_all) - logsumexp(union.log_v_all))
    n_per_bound = union.rng.multinomial(n_points, p)
    x = torch.cat([_sample(bound, int(n), gen) for bound, n in
                   zip(union.bounds, n_per_bound)])
    if union.cube is not None:
        x = x[torch.all((x >= 0) & (x < 1), dim=-1)]
    x = x[torch.randperm(len(x), generator=gen, device=x.device)]
    n_bound = sum(_contains(bound, x).to(torch.float64) for bound in
                  union.bounds)
    x = x[torch.rand(len(x), generator=gen, dtype=torch.float64,
                     device=x.device) > 1 - 1.0 / n_bound]
    union.n_sample += n_points
    union.n_reject += n_points - len(x)
    return x


def _neural_contains(bound, x):
    """`NeuralBound.contains` for a tensor."""
    in_bound = _contains(bound.outer_bound, x)
    if bound.emulator is not None:
        idx = torch.nonzero(in_bound)[:, 0]
        if len(idx) > 0:
            in_bound[idx] = predict(
                bound.emulator, _transform(bound.outer_bound, x[idx])) > (
                    bound.score_predict_min - 1e-9)
    return in_bound


def _in_any_neural(bounds, x):
    """Whether points lie in any neural bound, testing each bound only on the
    points not inside a previous one."""
    in_any = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    for bound in bounds:
        idx = torch.nonzero(~in_any)[:, 0]
        if len(idx) == 0:
            break
        in_any[idx] = _neural_contains(bound, x[idx])
    return in_any


def sample(bound, n_points):
    """One pass of `NautilusBound.sample` with the points on the device.

    Draws from the outer bound, tests membership in the neural network-based
    bounds and copies only the accepted points to the host. Random numbers
    come from a torch generator seeded by the bound's numpy generator.

    Parameters
    ----------
    bound : nautilus.bounds.NautilusBound
        Bound to sample, whose `device` is a torch device.
    n_points : int
        Number of points from the outer bound to test.

    Returns
    -------
    points : numpy.ndarray
        Accepted points.
    n_sample : int
        Number of points from the outer bound that were tested.

    """
    gen = torch.Generator(device=bound.device).manual_seed(
        int(bound.rng.integers(2**63)))
    union = bound.outer_bound
    x = _union_sample(union, oversample_size(
        n_points, union.n_sample, union.n_reject, margin=1, n_min=n_points),
        gen)
    in_bound = _in_any_neural(bound.neural_bounds, x)
    return x[in_bound].cpu().numpy(), len(x)


def contains(bound, points):
    """`NautilusBound.contains`, after any phase shift, on the device.

    Parameters
    ----------
    bound : nautilus.bounds.NautilusBound
        Bound whose `device` is a torch device.
    points : numpy.ndarray
        A 1-D or 2-D array of points, each row being a point.

    Returns
    -------
    in_bound : bool or numpy.ndarray
        Whether each point is contained in the bound.

    """
    x = torch.as_tensor(points, dtype=torch.float64, device=bound.device)
    x = x.reshape(-1, bound.n_dim)
    union = bound.outer_bound
    in_bound = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    for outer_bound in union.bounds:
        in_bound |= _contains(outer_bound, x)
    if union.cube is not None:
        in_bound &= torch.all((x >= 0) & (x < 1), dim=-1)
    if len(bound.neural_bounds) > 0:
        idx = torch.nonzero(in_bound)[:, 0]
        in_bound[idx] = _in_any_neural(bound.neural_bounds, x[idx])
    return in_bound.cpu().numpy().reshape(np.shape(points)[:-1])
