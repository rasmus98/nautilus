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
# Maximum number of transformed coordinates (points x ellipsoids x
# dimensions) held at once when testing bounds in `highest_bound`.
ELLIPSOID_CHUNK = 2**23
# Maximum number of first-layer activations (neural bounds x points x
# networks x units) held at once when testing neural bounds.
NETWORK_CHUNK = 2**24
# Number of neural bounds whose networks `highest_bound` evaluates in its
# first block; each further block is twice as large.
FIRST_BLOCK = 1
# Maximum number of minibatch steps per CUDA graph. An epoch is captured as
# consecutive graphs sharing one memory pool, which bounds the pool to the
# temporaries of this many steps instead of a whole epoch.
GRAPH_STEPS = 16
# Training epochs between host checks of whether all networks stopped, on
# CUDA devices, where each check waits for the device. Stopped networks are
# frozen, so the extra epochs do not change the result.
CHECK_EVERY = 4
# Train with the split-minibatch HIP kernel (nautilus/_megakernel.py) instead
# of the CUDA graphs. Opt-in; AMD GPUs only. The result equals the graph
# trainer's to rounding. Cases the kernel does not support (other devices, no
# hipcc, alpha != 0, shapes that do not fit, a failed launch or a barrier
# timeout) use the graph trainer, with one warning per reason.
TRAIN_KERNEL = False


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

    x = torch.as_tensor(x, dtype=NN_DTYPE, device=device)
    y = torch.as_tensor(y, dtype=NN_DTYPE, device=device)
    result = None
    if TRAIN_KERNEL:
        result = _train_kernel(x, y, units, n_networks, network, seed,
                               batch_size)
    if result is None:
        gen, params = _init(units, n_networks, device, seed)
        result = (params, ) + _train_graphs(x, y, params, network, gen,
                                            batch_size)
    params, loss_curve, active_curve, best_loss = result

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


def _init(units, n_networks, device, seed):
    """Generator and initial weights and biases of all networks, stacked
    along the first axis (Glorot uniform, drawn in float64)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    params = []
    for fan_in, fan_out in zip(units[:-1], units[1:]):
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        for shape in [(fan_in, fan_out), (1, fan_out)]:
            params.append(torch.empty(
                (n_networks, ) + shape, dtype=torch.float64,
                device=device).uniform_(-bound, bound, generator=gen).to(
                NN_DTYPE).requires_grad_(True))
    return gen, params


def _step_sizes(network, n_steps, device, epoch_0, n_epochs):
    """Bias-corrected Adam step sizes of epochs epoch_0, ...,
    epoch_0 + n_epochs - 1, shape (n_epochs, n_steps)."""
    t = np.arange(epoch_0 * n_steps + 1, (epoch_0 + n_epochs) * n_steps + 1)
    return torch.as_tensor(
        network.learning_rate_init * np.sqrt(1 - network.beta_2**t) /
        (1 - network.beta_1**t), dtype=NN_DTYPE, device=device).view(
            n_epochs, n_steps)


def _train_kernel(x, y, units, n_networks, network, seed, batch_size):
    """Train with the HIP kernel (TRAIN_KERNEL). Returns the parameters and
    curves as `_train_graphs`, or None if the kernel does not support the
    case."""
    from . import _megakernel
    try:
        if network.alpha != 0:
            raise _megakernel.Unsupported('alpha != 0')
        if n_networks < 1:
            raise _megakernel.Unsupported('no networks')
        _megakernel.library(x.device, NN_DTYPE)
        gen, params = _init(units, n_networks, x.device, seed)
        curves = _megakernel.train(
            x, y, params, network, gen, batch_size, CHECK_EVERY, partial(
                _step_sizes, network, -(-len(x) // batch_size), x.device))
    except _megakernel.Unsupported as error:
        _megakernel.warn_once(str(error))
        return None
    return (params, ) + curves


def _train_graphs(x, y, params, network, gen, batch_size):
    """Train the stacked parameters in place with Adam, replaying each epoch
    as CUDA graphs on CUDA devices; see `train_networks`.

    Returns the loss curve, the active curve and the best loss.
    """
    device = x.device
    n_networks, (n_points, n_dim) = len(params[0]), x.shape

    def forward(h):
        for k in range(0, len(params), 2):
            h = torch.baddbmm(params[k + 1], h, params[k])
            if k < len(params) - 2:
                h = torch.relu(h)
        return h[..., 0]

    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    beta_1, beta_2 = network.beta_1, network.beta_2
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
        lr_block = _step_sizes(network, len(slices), device, epoch_0,
                               n_epochs)
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
    return loss_curve, active_curve, best_loss


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


def _gaussian_parameters(x, resp, reg_covar):
    """`_estimate_gaussian_parameters` of scikit-learn (full covariances).

    `resp` has one row per run and component. Every row goes through the same
    operations as its own batch element, so two runs whose components are a
    permutation of each other give identical numbers, as in scikit-learn.

    Returns the component sums `nk`, the means, the covariances, the Cholesky
    factors of the precisions and whether a covariance is not positive
    definite.
    """
    x_b = x.expand(len(resp), -1, -1)
    nk = torch.sum(resp, dim=1) + 10 * torch.finfo(resp.dtype).eps
    means = torch.bmm(resp[:, None, :], x_b)[:, 0] / nk[:, None]
    diff = x_b - means[:, None, :]
    covariances = torch.bmm((diff * resp[..., None]).mT, diff) / nk[
        :, None, None]
    covariances.diagonal(dim1=-2, dim2=-1).add_(reg_covar)
    cov_chol, info = torch.linalg.cholesky_ex(covariances)
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    precisions_chol = torch.linalg.solve_triangular(
        cov_chol, eye.expand_as(cov_chol), upper=False).mT
    return nk, means, covariances, precisions_chol, info > 0


def gaussian_mixture(points, random_state, device, n_components=2, n_init=10,
                     tol=1e-3, reg_covar=1e-6, max_iter=100):
    """`GaussianMixture(n_components, n_init=n_init,
    random_state=random_state).fit(points)`, with the EM runs on a device.

    The algorithm is scikit-learn's: full covariances, one k-means
    initialisation per run (scikit-learn's `KMeans`, drawing from one random
    state in the same order), EM steps until the change of the mean
    log-likelihood is below `tol` or for `max_iter` steps, and the run with
    the highest final lower bound wins, the first one on ties. Only the EM
    steps of the `n_init` runs are batched, so the result equals
    scikit-learn's up to floating-point rounding. Runs that converge to the
    same solution with permuted components tie exactly in scikit-learn; they
    tie here too (see `_gaussian_parameters`), and lower bounds within a
    relative 1e-12 count as ties in case rounding breaks one.

    Returns
    -------
    weights, means, covariances : numpy.ndarray
        Parameters of the best run, as `weights_`, `means_` and
        `covariances_` of `GaussianMixture`.

    Raises
    ------
    ValueError
        If a covariance is not positive definite, as in scikit-learn.

    """
    from sklearn.cluster import KMeans
    from sklearn.utils import check_array, check_random_state

    points = check_array(points, dtype=[np.float64, np.float32],
                         ensure_min_samples=2)
    n_points, n_dim = points.shape
    k = n_components
    random_state = check_random_state(random_state)
    resp = np.zeros((n_init, k, n_points))
    for i in range(n_init):
        labels = KMeans(n_clusters=k, n_init=1,
                        random_state=random_state).fit(points).labels_
        resp[i, labels, np.arange(n_points)] = 1

    device = torch.device(device)
    x = torch.as_tensor(points, dtype=torch.float64, device=device)
    # Parameters per run and component, flattened: row i * k + j.
    nk, means, covariances, precisions_chol, failed = _gaussian_parameters(
        x, torch.as_tensor(resp.reshape(n_init * k, n_points), device=device),
        reg_covar)
    weights = nk.view(n_init, k) / n_points
    const = n_dim * np.log(2 * np.pi)
    lower_bound = np.full(n_init, -np.inf)
    active = np.arange(n_init)
    failed = bool(torch.any(failed))
    for _ in range(max_iter):
        if failed:
            break
        runs = torch.as_tensor(active, device=device)
        rows = torch.as_tensor((active[:, None] * k + np.arange(k)).ravel(),
                               device=device)
        p_chol = precisions_chol[rows]
        # E-step: log of the weighted probability of each point under each
        # component, normalised over the components.
        y = torch.bmm(x.expand(len(rows), -1, -1), p_chol) - torch.bmm(
            means[rows][:, None, :], p_chol)
        log_det = torch.sum(torch.log(torch.diagonal(
            p_chol, dim1=-2, dim2=-1)), dim=-1)
        log_prob = (-0.5 * (const + torch.sum(y**2, dim=-1)) +
                    log_det[:, None]) + torch.log(weights[runs]).view(-1, 1)
        log_prob = log_prob.view(len(active), k, n_points)
        log_prob_norm = torch.logsumexp(log_prob, dim=1)
        # M-step.
        nk_a, means[rows], covariances[rows], precisions_chol[rows], \
            failed_a = _gaussian_parameters(x, torch.exp(
                log_prob - log_prob_norm[:, None]).view(-1, n_points),
                reg_covar)
        nk_a = nk_a.view(len(active), k)
        weights[runs] = nk_a / torch.sum(nk_a, dim=1, keepdim=True)
        # One copy to the host per step: the lower bounds and failures.
        lower_bound_new = torch.cat([torch.mean(log_prob_norm, dim=1),
                                     failed_a.to(torch.float64)]).cpu().numpy()
        failed = np.any(lower_bound_new[len(active):] > 0)
        lower_bound_new = lower_bound_new[:len(active)]
        change = lower_bound_new - lower_bound[active]
        lower_bound[active] = lower_bound_new
        active = active[~(np.abs(change) < tol)]
        if len(active) == 0:
            break
    if failed:
        raise ValueError(
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by "
            "singleton or collapsed samples). Try to decrease the number of "
            "components, increase reg_covar, or scale the input data.")

    best, max_lower_bound = 0, -np.inf
    for i in range(n_init):
        if (max_lower_bound == -np.inf or lower_bound[i] - max_lower_bound >
                1e-12 * abs(max_lower_bound)):
            best, max_lower_bound = i, lower_bound[i]
    return tuple(a.cpu().numpy() for a in (
        weights[best], means[best * k:(best + 1) * k],
        covariances[best * k:(best + 1) * k]))


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
    if (str(device), keys) not in cache:
        cache[str(device), keys] = [
            torch.as_tensor(getattr(bound, key), device=device) for key in
            keys]
    return cache[str(device), keys]


def _dims(mixture, device):
    """Indices of the cube and of the ellipsoid dimensions of a cube-ellipsoid
    mixture as tensors, cached on the mixture. Indexing with them, unlike with
    boolean masks, does not wait for the device."""
    cache = mixture.__dict__.setdefault('_torch_cache', {})
    if (str(device), 'dims') not in cache:
        cache[str(device), 'dims'] = [torch.as_tensor(
            np.flatnonzero(select), device=device) for select in
            (mixture.dim_cube, ~mixture.dim_cube)]
    return cache[str(device), 'dims']


def _transform(ellipsoid, x):
    """`Ellipsoid.transform` for a tensor."""
    c, B_inv = _params(ellipsoid, x.device, ('c', 'B_inv'))
    return (x - c) @ B_inv.T


def _contains(bound, x):
    """`contains` of an ellipsoid or a cube-ellipsoid mixture for a tensor."""
    if isinstance(bound, Ellipsoid):
        return torch.sum(_transform(bound, x)**2, dim=-1) < 1
    in_bound = torch.ones(len(x), dtype=torch.bool, device=x.device)
    dim_cube, dim_ellipsoid = _dims(bound, x.device)
    if bound.cube is not None:
        x_cube = x.index_select(1, dim_cube)
        in_bound &= torch.all((x_cube >= 0) & (x_cube < 1), dim=-1)
    if bound.ellipsoid is not None:
        in_bound &= _contains(bound.ellipsoid, x.index_select(1, dim_ellipsoid))
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
    dim_cube, dim_ellipsoid = _dims(bound, device)
    if bound.cube is not None:
        x.index_copy_(1, dim_cube, torch.rand(
            (n_points, len(dim_cube)), generator=gen, dtype=torch.float64,
            device=device))
    if bound.ellipsoid is not None:
        x.index_copy_(1, dim_ellipsoid, _sample(bound.ellipsoid, n_points,
                                                gen))
    return x


def _union_draw(union, n_points, gen):
    """Draw points from a union and decide which ones `Union.sample` keeps.

    Each point is drawn from one of the union's bounds, chosen with
    probability proportional to its volume, and kept with probability one
    over the number of bounds containing it (and only inside the unit cube,
    if the union has one). The union's sampling counters are not updated.

    Returns
    -------
    x : torch.Tensor
        All drawn points, grouped by bound.
    keep : torch.Tensor
        Whether each point is kept.

    """
    p = np.exp(np.array(union.log_v_all) - logsumexp(union.log_v_all))
    n_per_bound = union.rng.multinomial(n_points, p)
    x = torch.cat([_sample(bound, int(n), gen) for bound, n in
                   zip(union.bounds, n_per_bound)])
    n_bound = sum(_contains(bound, x).to(torch.float64) for bound in
                  union.bounds)
    keep = torch.rand(n_points, generator=gen, dtype=torch.float64,
                      device=x.device) > 1 - 1.0 / n_bound
    if union.cube is not None:
        keep &= torch.all((x >= 0) & (x < 1), dim=-1)
    return x, keep


def _stack_networks(bounds, device):
    """Networks of neural bounds, stacked along the first axis.

    Returns None if the bounds have no emulators. Otherwise a dictionary with
    the emulators' input normalization (`mean`, `scale`), the first layer of
    all networks of a bound as one matrix (`w_0`, `b_0`), the later layers
    (`w`, `b`, lists over layers of arrays of shape (bounds, networks, in,
    out)), the score thresholds (`threshold`) and `n_networks`.
    """
    if bounds[0].emulator is None:
        return None
    emulators = [bound.emulator for bound in bounds]
    n_layers = len(emulators[0].neural_networks[0].coefs_)

    def layer(k, name):
        return np.array([[getattr(network, name)[k] for network in
                          emulator.neural_networks] for emulator in emulators])

    w_0 = layer(0, 'coefs_')
    n_bounds, n_networks, n_dim, n_hidden = w_0.shape
    networks = dict(
        mean=np.array([emulator.mean for emulator in emulators]),
        scale=np.array([emulator.scale for emulator in emulators]),
        w_0=w_0.transpose(0, 2, 1, 3).reshape(
            n_bounds, n_dim, n_networks * n_hidden),
        b_0=layer(0, 'intercepts_').reshape(n_bounds, 1, -1),
        w=[layer(k, 'coefs_') for k in range(1, n_layers)],
        b=[layer(k, 'intercepts_')[:, :, np.newaxis] for k in
           range(1, n_layers)],
        threshold=np.array([bound.score_predict_min - 1e-9 for bound in
                            bounds]))
    networks = {key: [torch.as_tensor(a, dtype=NN_DTYPE, device=device) for
                      a in value] if isinstance(value, list) else
                torch.as_tensor(value, device=device, dtype=torch.float64 if
                                key in ('mean', 'scale') else NN_DTYPE)
                for key, value in networks.items()}
    networks['n_networks'] = n_networks
    return networks


def _bound_stack(bound):
    """Tensors describing a `NautilusBound` on its device, cached on it.

    The ellipsoids of the outer bound's cube-ellipsoid mixtures are embedded
    in the full space, with zero rows and columns along the cube dimensions
    (`c_m`, `B_inv_T_m`, `dim_cube`). The outer ellipsoids (`c_n`,
    `B_inv_T_n`) and the networks (`networks`, see `_stack_networks`) of the
    neural bounds are stacked.
    """
    key = (str(bound.device), NN_DTYPE)
    cache = getattr(bound, '_torch_bound', None)
    if cache is not None and cache[0] == key:
        return cache[1]
    n_dim = bound.n_dim
    mixtures = bound.outer_bound.bounds
    c_m = np.zeros((len(mixtures), n_dim))
    B_inv_m = np.zeros((len(mixtures), n_dim, n_dim))
    for i, mixture in enumerate(mixtures):
        if mixture.ellipsoid is not None:
            select = ~mixture.dim_cube
            c_m[i, select] = mixture.ellipsoid.c
            B_inv_m[i][np.ix_(select, select)] = mixture.ellipsoid.B_inv
    device = torch.device(bound.device)
    neural = bound.neural_bounds
    stack = dict(
        c_m=torch.as_tensor(c_m, device=device),
        B_inv_T_m=torch.as_tensor(B_inv_m.transpose(0, 2, 1).copy(),
                                  device=device),
        dim_cube=torch.as_tensor(np.array([
            mixture.dim_cube for mixture in mixtures]), device=device),
        cube=bound.outer_bound.cube is not None,
        c_n=torch.as_tensor(np.array([
            b.outer_bound.c for b in neural]), device=device),
        B_inv_T_n=torch.as_tensor(np.array([
            b.outer_bound.B_inv.T for b in neural]), device=device),
        networks=_stack_networks(neural, device))
    bound._torch_bound = (key, stack)
    return stack


def _scores(networks, k_0, k_1, x_t):
    """Mean network predictions of neural bounds `k_0` to `k_1 - 1`.

    Parameters
    ----------
    networks : dict
        Stacked networks, see `_stack_networks`.
    k_0, k_1 : int
        Range of neural bounds.
    x_t : torch.Tensor
        Points transformed by each bound's outer ellipsoid, of shape
        (k_1 - k_0, n_points, n_dim).

    Returns
    -------
    score : torch.Tensor
        Predictions of shape (k_1 - k_0, n_points).

    """
    n_bounds, n_points, n_dim = x_t.shape
    n_networks = networks['n_networks']
    n_rows = max(1, NETWORK_CHUNK // (n_bounds * networks['w_0'].shape[2]))
    mean = networks['mean'][k_0:k_1, None]
    scale = networks['scale'][k_0:k_1, None]
    w_0, b_0 = networks['w_0'][k_0:k_1], networks['b_0'][k_0:k_1]
    w = [w[k_0:k_1].flatten(0, 1) for w in networks['w']]
    b = [b[k_0:k_1].flatten(0, 1) for b in networks['b']]
    score = torch.empty((n_bounds, n_points), dtype=NN_DTYPE,
                        device=x_t.device)
    for start in range(0, n_points, n_rows):
        h = ((x_t[:, start:start + n_rows] - mean) / scale).to(NN_DTYPE)
        m = h.shape[1]
        # First layer of all networks of a bound as one matrix product.
        h = torch.baddbmm(b_0, h, w_0).view(
            n_bounds, m, n_networks, -1).transpose(1, 2).reshape(
                n_bounds * n_networks, m, -1)
        for w_k, b_k in zip(w, b):
            h = torch.baddbmm(b_k, torch.relu(h), w_k)
        score[:, start:start + m] = torch.mean(
            h.view(n_bounds, n_networks, m), dim=1)
    return score


def _neural_accept(stack, x):
    """Whether points lie in any of a bound's neural bounds.

    Each network sees the points inside its outer ellipsoid, all networks
    after one compaction.
    """
    c, B_inv_T = stack['c_n'], stack['B_inv_T_n']
    todo = torch.stack([torch.sum(((x - c[k]) @ B_inv_T[k])**2, dim=-1) < 1
                        for k in range(len(c))])
    networks = stack['networks']
    if networks is None:
        return torch.any(todo, dim=0)
    pairs = torch.nonzero(todo)
    counts = torch.bincount(pairs[:, 0], minlength=len(c)).tolist()
    accept = torch.zeros(len(x), dtype=torch.int32, device=x.device)
    start = 0
    for k, count in enumerate(counts):
        if count > 0:
            rows = pairs[start:start + count, 1]
            score = _scores(networks, k, k + 1,
                            ((x[rows] - c[k]) @ B_inv_T[k])[None])[0]
            accept.index_add_(0, rows, (score > networks['threshold'][k]).to(
                torch.int32))
            start += count
    return accept > 0


def _stack_bounds(sampler):
    """Tensors of all of a sampler's bounds after the unit cube, stacked.

    Concatenates the `_bound_stack` of each bound. `owner_m` and `owner_n`
    give the bound (counted from the first bound after the unit cube) of each
    mixture and neural bound, and `start_m` and `start_n` the first mixture
    and neural bound of each bound. The result is cached on the sampler until
    its bounds change.
    """
    bounds = sampler.bounds[1:]
    key = (str(sampler.device), NN_DTYPE)
    stack = getattr(sampler, '_torch_stack', None)
    if stack is not None and stack['key'] == key and len(
            stack['bounds']) == len(bounds) and all(
            a is b for a, b in zip(stack['bounds'], bounds)):
        return stack
    device = torch.device(sampler.device)
    parts = [_bound_stack(bound) for bound in bounds]
    n_m = [len(part['c_m']) for part in parts]
    n_n = [len(part['c_n']) for part in parts]
    stack = dict(key=key, bounds=list(bounds),
                 start_m=np.append(0, np.cumsum(n_m)).tolist(),
                 start_n=np.append(0, np.cumsum(n_n)).tolist(),
                 owner_m=torch.as_tensor(np.repeat(np.arange(len(parts)), n_m),
                                         device=device),
                 owner_n=torch.as_tensor(np.repeat(np.arange(len(parts)), n_n),
                                         device=device),
                 cube=torch.as_tensor([part['cube'] for part in parts],
                                      device=device))
    for name in ['c_m', 'B_inv_T_m', 'dim_cube', 'c_n', 'B_inv_T_n']:
        stack[name] = torch.cat([part[name] for part in parts])
    if any(part['networks'] is None for part in parts):
        stack['networks'] = None
    else:
        stack['networks'] = {name: [torch.cat([
            part['networks'][name][k] for part in parts]) for k in range(len(
                value))] if isinstance(value, list) else torch.cat([
                    part['networks'][name] for part in parts]) if isinstance(
                        value, torch.Tensor) else value for name, value in
            parts[0]['networks'].items()}
    sampler._torch_stack = stack
    return stack


def _shift(shift, x):
    """`PhaseShift.transform` for a tensor."""
    if shift is None:
        return x
    x = x.clone()
    x[:, shift.periodic] = (x[:, shift.periodic] + torch.as_tensor(
        -shift.centers + 0.5, device=x.device)) % 1
    return x


def _highest(networks, k_0, x_t, todo, level):
    """Highest level of the neural bounds containing each point.

    Parameters
    ----------
    networks : dict or None
        Stacked networks, see `_stack_networks`.
    k_0 : int
        Index of the first neural bound in `networks`.
    x_t : torch.Tensor
        Points transformed by each neural bound's outer ellipsoid, of shape
        (n_bounds, n_points, n_dim), in order of increasing level.
    todo : torch.Tensor
        Whether each point lies inside each neural bound's outer ellipsoid and
        its bound's outer union, of shape (n_bounds, n_points).
    level : torch.Tensor
        Level (bound index) of each neural bound, non-decreasing.

    Returns
    -------
    result : torch.Tensor
        Highest level of a neural bound containing each point, or -1.

    """
    if networks is None:
        return torch.amax(torch.where(todo, level[:, None], -1), dim=0)
    n_bounds, n_points = todo.shape
    # Networks are evaluated in blocks of neural bounds from the highest
    # level down, each block twice as large as the previous one, on the
    # points not yet in a higher bound that lie in the outer ellipsoid of a
    # neural bound of the block. Selecting these points waits for the device
    # once per block. An extra dummy point, inside no outer ellipsoid, is
    # selected whenever lower blocks have such points left; without any
    # selected point, the search stops.
    todo = torch.cat([todo, torch.zeros_like(todo[:, :1])], dim=1)
    result = torch.full((n_points + 1, ), -1, dtype=level.dtype,
                        device=todo.device)
    k_1, size = n_bounds, FIRST_BLOCK
    while k_1 > 0:
        k = max(k_1 - size, 0)
        todo = todo[:k_1] & (result < 0)
        select = torch.any(todo[k:], dim=0)
        select[n_points] = torch.any(todo[:k])
        rows = torch.nonzero(select)[:, 0]
        if len(rows) == 0:
            break
        accept = todo[k:, rows] & (_scores(
            networks, k_0 + k, k_0 + k_1,
            x_t[k:k_1, rows.clamp(max=n_points - 1)]) > networks[
                'threshold'][k_0 + k:k_0 + k_1, None])
        result[rows] = torch.amax(torch.where(accept, level[k:k_1, None], -1),
                                  dim=0)
        k_1, size = k, 2 * size
    return result[:n_points]


def highest_bound(sampler, points, start, stop):
    """Index of the last of the sampler's bounds `start` to `stop - 1`
    containing each point, on the device.

    Replaces the loops over bounds in `Sampler.sample_shell` and
    `Sampler.shell_association`. The ellipsoids of all bounds are tested in
    one batched transform. The networks of the neural bounds are then
    evaluated in a few blocks from the last bound down (see `_highest`). One
    copy goes to the device and one back.

    Parameters
    ----------
    sampler : nautilus.Sampler
        Sampler whose `device` is a torch device.
    points : numpy.ndarray
        A 2-D array where each row represents a point.
    start, stop : int
        Range of bound indices to test.

    Returns
    -------
    index : numpy.ndarray
        For each point, the largest index of a bound containing it, or -1.

    """
    index = np.full(len(points), -1)
    if start >= stop or len(points) == 0:
        return index
    x_all = torch.as_tensor(points, dtype=torch.float64,
                            device=sampler.device)
    result = torch.full((len(x_all), ), -1, device=x_all.device)
    first = max(start, 1)
    if first < stop:
        stack = _stack_bounds(sampler)
        bounds = stack['bounds'][first - 1:stop - 1]
        m_0, m_1 = stack['start_m'][first - 1], stack['start_m'][stop - 1]
        k_0, k_1 = stack['start_n'][first - 1], stack['start_n'][stop - 1]
        c_m, B_inv_T_m, dim_cube = (stack[name][m_0:m_1] for name in (
            'c_m', 'B_inv_T_m', 'dim_cube'))
        c_n, B_inv_T_n = stack['c_n'][k_0:k_1], stack['B_inv_T_n'][k_0:k_1]
        owner_m = stack['owner_m'][m_0:m_1] - (first - 1)
        owner_n = stack['owner_n'][k_0:k_1] - (first - 1)
        cube = stack['cube'][first - 1:stop - 1]
        n_rows = max(1, ELLIPSOID_CHUNK // (
            (m_1 - m_0 + k_1 - k_0) * sampler.n_dim))
        for i in range(0, len(x_all), n_rows):
            x = x_all[i:i + n_rows]
            if all(bound.shift is None for bound in bounds):
                x = x.expand(len(bounds), -1, -1)
            else:
                x = torch.stack([_shift(bound.shift, x) for bound in bounds])
            outside = (x < 0) | (x >= 1)
            in_mixture = (torch.sum(torch.bmm(
                x[owner_m] - c_m[:, None], B_inv_T_m)**2, dim=-1) < 1) & \
                ~torch.any(outside[owner_m] & dim_cube[:, None], dim=-1)
            in_union = torch.zeros(
                (len(bounds), x.shape[1]), dtype=torch.float64,
                device=x.device).index_add_(0, owner_m,
                                            in_mixture.double()) > 0
            in_union &= ~(torch.any(outside, dim=-1) & cube[:, None])
            x_t = torch.bmm(x[owner_n] - c_n[:, None], B_inv_T_n)
            todo = (torch.sum(x_t**2, dim=-1) < 1) & in_union[owner_n]
            result[i:i + n_rows] = _highest(
                stack['networks'], k_0, x_t, todo, owner_n + first)
    if start == 0:
        result = torch.where((result < 0) & torch.all(
            (x_all >= 0) & (x_all < 1), dim=-1), 0, result)
    return result.cpu().numpy()


def sample(bound, n_points):
    """One pass of `NautilusBound.sample` with the points on the device.

    Draws from the outer bound, tests membership in the neural network-based
    bounds and copies only the accepted points, in random order, to the host.
    Random numbers come from a torch generator seeded by the bound's numpy
    generator. Updates the outer bound's sampling counters.

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
    n_draw = oversample_size(n_points, union.n_sample, union.n_reject,
                             margin=1, n_min=n_points)
    x, keep = _union_draw(union, n_draw, gen)
    x = x[keep]
    n_keep = len(x)
    if len(bound.neural_bounds) > 0:
        x = x[_neural_accept(_bound_stack(bound), x)]
    x = x[torch.randperm(len(x), generator=gen, device=x.device)]
    union.n_sample += n_draw
    union.n_reject += n_draw - n_keep
    return x.cpu().numpy(), n_keep


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
        in_bound[idx] = _neural_accept(_bound_stack(bound), x[idx])
    return in_bound.cpu().numpy().reshape(np.shape(points)[:-1])
