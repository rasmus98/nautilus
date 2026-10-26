import subprocess
import sys

import h5py
import numpy as np
import pytest

from nautilus import Sampler
from nautilus.neural import NeuralNetworkEmulator


def test_import_does_not_import_torch():
    # Importing and running nautilus with the default backend never imports
    # torch.
    code = ("import sys, nautilus; s = nautilus.Sampler(lambda x: x, " +
            "lambda x: -sum(x**2), n_dim=2, n_live=50, seed=0); " +
            "s.run(n_eff=50, verbose=False); assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check=True)


def test_missing_torch_error():
    # Without torch, requesting the torch backend raises an ImportError that
    # names the extra.
    code = ("import sys, nautilus; sys.modules['torch'] = None\n" +
            "try:\n    nautilus.Sampler(lambda x: x, lambda x: 0, n_dim=2, " +
            "backend='torch')\nexcept ImportError as e:\n" +
            "    assert 'nautilus-sampler[torch]' in str(e)\n" +
            "else:\n    raise AssertionError")
    subprocess.run([sys.executable, '-c', code], check=True)


def test_backend_arguments():
    with pytest.raises(ValueError):
        Sampler(lambda x: x, lambda x: 0, n_dim=2, backend='jax')
    with pytest.raises(ValueError):
        Sampler(lambda x: x, lambda x: 0, n_dim=2, device='cpu')


def test_torch_emulator_checkpoint(tmp_path):
    # A torch-trained emulator is written in the scikit-learn layout and gives
    # the same predictions when read by either backend.
    pytest.importorskip('torch')
    rng = np.random.default_rng(0)
    x = rng.random((1000, 5))
    y = np.linalg.norm(x - 0.5, axis=1)
    y = np.argsort(np.argsort(y)) / float(len(y))
    emu = NeuralNetworkEmulator.train(x, y, n_networks=2, device='cpu')
    assert np.sqrt(np.mean((y - emu.predict(x))**2)) < 0.3 * np.std(y)

    with h5py.File(tmp_path / 'emu.h5', 'w') as fstream:
        emu.write(fstream.create_group('emulator'))
    with h5py.File(tmp_path / 'emu.h5', 'r') as fstream:
        emu_np = NeuralNetworkEmulator.read(fstream['emulator'])
        emu_torch = NeuralNetworkEmulator.read(fstream['emulator'],
                                               device='cpu')
    assert np.allclose(emu_np.predict(x), emu.predict(x), atol=1e-5)
    assert np.allclose(emu_torch.predict(x), emu.predict(x), atol=1e-5)


@pytest.mark.parametrize('first,second', [('numpy', 'torch'),
                                          ('torch', 'numpy')])
def test_checkpoint_cross_backend(tmp_path, first, second):
    # A run checkpointed with one backend can be resumed with the other.
    pytest.importorskip('torch')

    def likelihood(x):
        return -np.sum((x - 0.5)**2 / 0.01, axis=-1)

    def kwargs(backend):
        return dict(n_dim=3, n_live=200, vectorized=True, seed=0,
                    filepath=tmp_path / 'checkpoint.hdf5', backend=backend)

    sampler = Sampler(lambda x: x, likelihood, **kwargs(first))
    sampler.run(n_like_max=2000, verbose=False)
    n_bounds = len(sampler.bounds)
    assert n_bounds > 2

    sampler = Sampler(lambda x: x, likelihood, **kwargs(second))
    assert len(sampler.bounds) == n_bounds
    sampler.run(n_eff=1000, verbose=False)
    assert len(sampler.bounds) > n_bounds
    assert np.abs(sampler.log_z - np.log((2 * np.pi * 0.005)**1.5)) < 0.2


def test_torch_sampler():
    # The torch backend recovers the evidence of a correlated Gaussian.
    pytest.importorskip('torch')
    n_dim = 4
    cov = 0.01 * (0.5 * np.eye(n_dim) + 0.5)

    def likelihood(x):
        d = x - 0.5
        return -0.5 * np.einsum('...i,ij,...j', d, np.linalg.inv(cov), d)

    sampler = Sampler(lambda x: x, likelihood, n_dim=n_dim, n_live=500,
                      vectorized=True, seed=0, backend='torch')
    sampler.run(n_eff=2000, verbose=False)
    log_z = 0.5 * np.linalg.slogdet(2 * np.pi * cov)[1]
    assert np.abs(sampler.log_z - log_z) < 0.15
    points, log_w, log_l = sampler.posterior()
    mean = np.average(points, weights=np.exp(log_w), axis=0)
    assert np.all(np.abs(mean - 0.5) < 0.02)


def test_oversampling_is_uniform():
    # Drawing many points at once (torch backend) samples the union of two
    # overlapping ellipsoids uniformly, like the default backend. This needs
    # no torch: the device only switches the batch sizing.
    from nautilus.bounds import Union
    from nautilus.bounds.union import oversample_size
    rng = np.random.default_rng(0)
    points = np.concatenate([rng.normal(0.4, 0.05, size=(500, 2)),
                             rng.normal(0.55, 0.05, size=(500, 2))])
    results = []
    for device in [None, 'cpu']:
        bound = Union.compute(points, rng=np.random.default_rng(1),
                              device=device)
        assert bound.split()
        x = bound.sample(40000)
        results.append([np.mean(x, axis=0), np.mean(
            bound.bounds[0].contains(x) & bound.bounds[1].contains(x))])
    assert np.allclose(results[0][0], results[1][0], atol=3e-3)
    assert np.abs(results[0][1] - results[1][1]) < 0.01
    assert oversample_size(256, 0, 0) == 1000
    assert oversample_size(256, 10**5, 10**5 - 100) > 10**5
    assert oversample_size(10**6, 10**5, 10**5 - 100) == 2**18


def test_device_bound_matches_numpy():
    # The device versions of NautilusBound.contains and NautilusBound.sample
    # agree with the numpy versions: identical membership, and uniform draws
    # with the same volume estimate.
    pytest.importorskip('torch')
    from nautilus.bounds import NautilusBound
    rng = np.random.default_rng(0)
    points = rng.random((4000, 3))
    log_l = np.maximum(-np.sum((points - 0.25)**2, axis=1),
                       -np.sum((points - 0.75)**2, axis=1)) / 0.02
    bound = NautilusBound.compute(
        points, log_l, np.quantile(log_l, 0.9), np.log(0.05),
        n_points_min=50, split_threshold=1, rng=np.random.default_rng(1),
        device='cpu')
    assert len(bound.outer_bound.bounds) > 1

    x = rng.random((20000, 3))
    in_device = bound.contains(x)
    bound.device = None
    assert np.array_equal(in_device, bound.contains(x))
    assert bound.contains(x[0]) == in_device[0]

    results = []
    for device in [None, 'cpu']:
        bound.device = device
        bound.reset(np.random.default_rng(2))
        x = bound.sample(40000)
        assert np.all(bound.contains(x))
        results.append([np.mean(x, axis=0), np.mean(x[:, 0] < 0.5),
                        bound.log_v])
    assert np.allclose(results[0][0], results[1][0], atol=4e-3)
    assert np.abs(results[0][1] - results[1][1]) < 0.015
    assert np.abs(results[0][2] - results[1][2]) < 0.03


def test_device_union_is_uniform():
    # Device draws from the union of two overlapping ellipsoids are uniform:
    # same mean and overlap fraction as the numpy draws, and the union volume
    # estimate agrees.
    torch = pytest.importorskip('torch')
    from nautilus import _torch
    from nautilus.bounds import Union
    rng = np.random.default_rng(0)
    points = np.concatenate([rng.normal(0.4, 0.05, size=(500, 2)),
                             rng.normal(0.55, 0.05, size=(500, 2))])
    results = []
    for device in [None, 'cpu']:
        bound = Union.compute(points, rng=np.random.default_rng(1))
        assert bound.split()
        if device is None:
            x = bound.sample(40000)
        else:
            x, keep = _torch._union_draw(
                bound, 50000, torch.Generator().manual_seed(0))
            bound.n_sample, bound.n_reject = 50000, 50000 - int(keep.sum())
            x = x[keep].numpy()
            assert len(x) > 40000
        results.append([np.mean(x, axis=0), np.mean(
            bound.bounds[0].contains(x) & bound.bounds[1].contains(x)),
            bound.log_v])
    assert np.allclose(results[0][0], results[1][0], atol=3e-3)
    assert np.abs(results[0][1] - results[1][1]) < 0.01
    assert np.abs(results[0][2] - results[1][2]) < 0.01


def test_torch_seed_per_bound_resume(tmp_path):
    # Each neural bound draws its torch seed from the sampler's generator, so
    # bounds differ in their initial weights and a run resumed from a
    # checkpoint repeats the uninterrupted run.
    pytest.importorskip('torch')

    def likelihood(x):
        return -np.sum((x - 0.5)**2 / 0.01, axis=-1)

    def sampler(filepath=None):
        return Sampler(lambda x: x, likelihood, n_dim=3, n_live=200,
                       vectorized=True, seed=0, filepath=filepath,
                       backend='torch')

    full = sampler()
    full.run(n_like_max=3000, verbose=False)
    part = sampler(tmp_path / 'checkpoint.hdf5')
    part.run(n_like_max=1500, verbose=False)
    part = sampler(tmp_path / 'checkpoint.hdf5')
    part.run(n_like_max=3000, verbose=False)
    assert len(full.bounds) > 3
    assert part.n_like == full.n_like
    assert part.log_z == full.log_z
    assert np.array_equal(part.posterior()[0], full.posterior()[0])
    weights = [b.neural_bounds[0].emulator.neural_networks[0].coefs_[-1]
               for b in full.bounds[1:3]]
    assert not np.allclose(weights[0], weights[1])


@pytest.mark.parametrize('first_block,chunk', [(1, None), (3, 2**10),
                                                (1000, None)])
def test_highest_bound_matches_loop(monkeypatch, first_block, chunk):
    # The batched device test of the bounds in sample_shell and
    # shell_association agrees with testing each bound in turn, including
    # periodic phase shifts and points outside the unit cube, for any block
    # size of the network evaluation and with chunked evaluation.
    pytest.importorskip('torch')
    from nautilus import _torch
    monkeypatch.setattr(_torch, 'FIRST_BLOCK', first_block)
    if chunk is not None:
        monkeypatch.setattr(_torch, 'NETWORK_CHUNK', chunk)
        monkeypatch.setattr(_torch, 'ELLIPSOID_CHUNK', 64 * chunk)

    def likelihood(x):
        return np.logaddexp(-np.sum((x - 0.3)**2, axis=-1) / 0.005,
                            -np.sum((x - 0.75)**2, axis=-1) / 0.005)

    sampler = Sampler(lambda x: x, likelihood, n_dim=3, n_live=200,
                      vectorized=True, seed=0, periodic=[0],
                      backend='torch')
    sampler.run(n_like_max=3000, verbose=False)
    assert len(sampler.bounds) > 4
    assert all(b.shift is not None for b in sampler.bounds[1:])
    assert any(len(b.neural_bounds) > 1 for b in sampler.bounds[1:])
    x = np.random.default_rng(0).random((5000, 3)) * 1.2 - 0.1
    for index in range(len(sampler.bounds)):
        loop = np.zeros(len(x), dtype=bool)
        for bound in sampler.bounds[index + 1:]:
            loop |= bound.contains(x)
        assert np.array_equal(_torch.highest_bound(
            sampler, x, index + 1, len(sampler.bounds)) >= 0, loop)
    for n_max in range(len(sampler.bounds) + 1):
        shell = sampler.shell_association(x, n_max=n_max)
        sampler.device = None
        assert np.array_equal(shell, sampler.shell_association(x, n_max=n_max))
        sampler.device = 'cpu'


def test_device_bound_with_cube_dimensions():
    # A bound whose outer union treats a dimension as unit cube: device
    # membership is identical to numpy and device draws are uniform in the
    # bound, with the same volume estimates of the outer union and the bound.
    pytest.importorskip('torch')
    from nautilus.bounds import NautilusBound
    rng = np.random.default_rng(0)
    points = rng.random((6000, 3))
    log_l = -np.sum((points[:, :2] - 0.5)**2, axis=1) / 0.02
    bound = NautilusBound.compute(
        points, log_l, np.quantile(log_l, 0.7), np.log(0.3), n_points_min=50,
        rng=np.random.default_rng(1), device='cpu')
    assert any(np.any(m.dim_cube) for m in bound.outer_bound.bounds)

    x = rng.random((20000, 3))
    in_device = bound.contains(x)
    bound.device = None
    assert np.array_equal(in_device, bound.contains(x))

    results = []
    for device in [None, 'cpu']:
        bound.device = device
        bound.reset(np.random.default_rng(2))
        x = bound.sample(40000)
        assert np.all(bound.contains(x))
        results.append([np.mean(x, axis=0), np.mean(x[:, 2] < 0.25),
                        bound.outer_bound.log_v, bound.log_v])
    assert np.allclose(results[0][0], results[1][0], atol=4e-3)
    assert np.abs(results[0][1] - results[1][1]) < 0.015
    assert np.abs(results[0][2] - results[1][2]) < 0.02
    assert np.abs(results[0][3] - results[1][3]) < 0.03


def test_device_counters():
    # One device pass updates the outer union's counters with all drawn
    # points and returns the number of points kept by the union, of which
    # the accepted points are a subset in random order.
    torch = pytest.importorskip('torch')
    from nautilus import _torch
    from nautilus.bounds import NautilusBound
    rng = np.random.default_rng(0)
    points = rng.random((4000, 3))
    log_l = np.maximum(-np.sum((points - 0.25)**2, axis=1),
                       -np.sum((points - 0.75)**2, axis=1)) / 0.02
    bound = NautilusBound.compute(
        points, log_l, np.quantile(log_l, 0.9), np.log(0.05),
        n_points_min=50, split_threshold=1, rng=np.random.default_rng(1),
        device='cpu')
    union = bound.outer_bound
    n_sample, n_reject = union.n_sample, union.n_reject
    n_draw = _torch.oversample_size(5000, n_sample, n_reject, margin=1,
                                    n_min=5000)
    x, n_keep = _torch.sample(bound, 5000)
    assert union.n_sample - n_sample == n_draw
    assert union.n_reject - n_reject == n_draw - n_keep
    assert len(x) <= n_keep <= n_draw
    assert np.all(bound.contains(x))
    # Accepted points come in random order, not grouped by union member.
    first = np.argmax(np.stack([b.contains(x) for b in union.bounds]), axis=0)
    assert np.mean(first[:len(x) // 2]) == pytest.approx(
        np.mean(first[len(x) // 2:]), abs=0.1)
    # The same draws, with the union's keep decision, reproduce n_keep.
    bound.reset(np.random.default_rng(3))
    gen = torch.Generator().manual_seed(0)
    x_all, keep = _torch._union_draw(union, 10000, gen)
    n_bound = np.sum([b.contains(x_all.numpy()) for b in union.bounds],
                     axis=0)
    assert np.all(n_bound >= 1)
    assert np.all(~keep.numpy() | bound.outer_bound.contains(x_all.numpy()))


def test_device_without_networks():
    # With n_networks=0 the neural bounds are their outer ellipsoids; the
    # device tests agree with the default backend.
    pytest.importorskip('torch')
    from nautilus import _torch

    def likelihood(x):
        return np.logaddexp(-np.sum((x - 0.3)**2, axis=-1) / 0.005,
                            -np.sum((x - 0.75)**2, axis=-1) / 0.005)

    sampler = Sampler(lambda x: x, likelihood, n_dim=3, n_live=200,
                      n_networks=0, vectorized=True, seed=0, backend='torch')
    sampler.run(n_like_max=3000, verbose=False)
    assert len(sampler.bounds) > 3
    x = np.random.default_rng(0).random((5000, 3))
    for n_max in range(len(sampler.bounds) + 1):
        shell = sampler.shell_association(x, n_max=n_max)
        sampler.device = None
        assert np.array_equal(shell, sampler.shell_association(x, n_max=n_max))
        sampler.device = 'cpu'
    bound = sampler.bounds[-1]
    points, n_keep = _torch.sample(bound, 1000)
    assert len(points) > 0 and np.all(bound.contains(points))


def _kernel_case(n_points=600, n_dim=5):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n_points, n_dim))
    y = np.argsort(np.argsort(np.sum(x**2, axis=1))) / float(n_points)
    kwargs = dict(hidden_layer_sizes=(12, 7), alpha=0, max_iter=30,
                  learning_rate_init=1e-2, batch_size=50, tol=3e-4,
                  n_iter_no_change=1)
    return x, y, kwargs


def _same_networks(nets_a, nets_b, rtol=0, atol=0):
    for a, b in zip(nets_a, nets_b):
        assert a.n_iter_ == b.n_iter_
        assert np.allclose(a.loss_curve_, b.loss_curve_, rtol=rtol, atol=atol)
        for w_a, w_b in zip(a.coefs_ + a.intercepts_,
                            b.coefs_ + b.intercepts_):
            assert np.allclose(w_a, w_b, rtol=rtol, atol=atol)


def test_train_kernel_switch_and_fallback(monkeypatch):
    # TRAIN_KERNEL is off by default. When on, cases the HIP kernel does not
    # support (here: a CPU device, alpha > 0, a kernel failure) give exactly
    # the graph trainer's networks, with one warning per reason.
    torch = pytest.importorskip('torch')
    import warnings
    from nautilus import _megakernel, _torch
    assert _torch.TRAIN_KERNEL is False
    monkeypatch.setattr(_megakernel, '_WARNED', set())
    x, y, kwargs = _kernel_case()
    graph = _torch.train_networks(x, y, 3, kwargs, 'cpu', 5)
    assert len(set(net.n_iter_ for net in graph)) > 1

    monkeypatch.setattr(_torch, 'TRAIN_KERNEL', True)
    with pytest.warns(RuntimeWarning, match='AMD GPU'):
        _same_networks(_torch.train_networks(x, y, 3, kwargs, 'cpu', 5),
                       graph)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # the same reason warns only once
        _torch.train_networks(x, y, 3, kwargs, 'cpu', 5)
    with pytest.warns(RuntimeWarning, match='alpha'):
        _torch.train_networks(x, y, 3, kwargs | dict(alpha=1e-4), 'cpu', 5)

    # A kernel that fails after starting (launch error, barrier timeout):
    # fresh initialization, same result as the graph trainer.
    def fail(x, y, params, *args):
        with torch.no_grad():
            params[0].add_(1.0)
        raise _megakernel.Unsupported('barrier timeout')

    monkeypatch.setattr(_megakernel, 'library', lambda *args: None)
    monkeypatch.setattr(_megakernel, 'train', fail)
    with pytest.warns(RuntimeWarning, match='barrier timeout'):
        _same_networks(_torch.train_networks(x, y, 3, kwargs, 'cpu', 5),
                       graph)


def test_train_kernel_cache_and_layout(tmp_path):
    # The compiled kernel is cached per source, architecture and dtype; the
    # work split respects the LDS and co-residency limits.
    torch = pytest.importorskip('torch')
    from nautilus import _megakernel
    path = _megakernel.cache_path(b'src', 'gfx90a:sramecc+:xnack-',
                                  torch.float64, tmp_path)
    assert path.parent == tmp_path and path.name.endswith('-f64.so')
    assert 'gfx90a_sramecc_xnack' in path.name
    assert path == _megakernel.cache_path(
        b'src', 'gfx90a:sramecc+:xnack-', torch.float64, tmp_path)
    assert len({path, _megakernel.cache_path(
        b'src2', 'gfx90a:sramecc+:xnack-', torch.float64, tmp_path),
        _megakernel.cache_path(b'src', 'gfx942', torch.float64, tmp_path),
        _megakernel.cache_path(b'src', 'gfx90a:sramecc+:xnack-',
                               torch.float32, tmp_path)}) == 4

    calls = []

    def max_blocks(ldsw, threads, lds):
        calls.append(lds)
        return 110

    units = [33, 100, 50, 20, 1]
    plan = _megakernel.layout(units, 200, 4, torch.float64, max_blocks)
    assert plan['NP'] == 9491 and not plan['ldsw']
    assert (plan['G'], plan['R'], plan['RC'], plan['nch']) == (16, 13, 16, 1)
    assert plan['act'] == [0, 16 * 33, 16 * 133, 16 * 183, 16 * 203]
    assert plan['lds'] == 8 * (16 * 206 + 4) <= _megakernel.LDS_BYTES
    assert plan['ow'][1] == 3400 and plan['ob'][3] == 9490
    plan = _megakernel.layout(units, 200, 4, torch.float32, max_blocks)
    assert plan['ldsw'] and (plan['G'], plan['R'], plan['RC']) == (27, 8, 8)
    assert plan['lds'] == 4 * (9491 + 8 * 206 + 4)
    # Eight networks: fewer workgroups each, so that all are co-resident.
    assert _megakernel.layout(units, 200, 8, torch.float32,
                              max_blocks)['G'] == 13
    # Small batches: at least 4 rows per workgroup.
    assert _megakernel.layout(units, 30, 4, torch.float64,
                              max_blocks)['G'] == 8
    # Wide inputs: weights in global memory, rows in several chunks.
    plan = _megakernel.layout([1500, 100, 1], 200, 4, torch.float64,
                              max_blocks)
    assert not plan['ldsw'] and plan['RC'] == 4 and plan['nch'] == 4
    assert plan['S'] == plan['G'] * plan['nch']
    with pytest.raises(_megakernel.Unsupported):
        _megakernel.layout([10000, 100, 1], 200, 4, torch.float64, max_blocks)
    with pytest.raises(_megakernel.Unsupported):
        _megakernel.layout(units, 200, 111, torch.float64, max_blocks)
    with pytest.raises(_megakernel.Unsupported):
        _megakernel.layout([5] + [8] * 9 + [1], 200, 4, torch.float64,
                           max_blocks)


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_train_kernel_matches_graphs(monkeypatch, dtype):
    # On an AMD GPU, the HIP kernel gives the graph trainer's networks: in
    # float64 the same epochs and weights to rounding, including early
    # stopping; in float32 (rounding-chaotic training) the same loss curve
    # over the first epochs.
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('needs an AMD GPU')
    import warnings
    from nautilus import _torch
    monkeypatch.setattr(_torch, 'NN_DTYPE', getattr(torch, dtype))
    x, y, kwargs = _kernel_case(3000, 33)
    nets = []
    for kernel in [False, True]:
        monkeypatch.setattr(_torch, 'TRAIN_KERNEL', kernel)
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # no fallback
            nets.append(_torch.train_networks(x, y, 4, kwargs, 'cuda', 3))
    if dtype == 'float64':
        assert len(set(net.n_iter_ for net in nets[0])) > 1
        _same_networks(*nets, rtol=1e-10, atol=1e-13)
    else:
        for a, b in zip(*nets):
            assert np.allclose(a.loss_curve_[:3], b.loss_curve_[:3],
                               rtol=1e-3)


def test_train_kernel_timeout(monkeypatch):
    # A barrier timeout (forced by a 100 ns limit) ends the kernel; it is
    # then disabled for the process and the call returns the graph trainer's
    # networks.
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('needs an AMD GPU')
    from nautilus import _megakernel, _torch
    monkeypatch.setattr(_megakernel, 'TIMEOUT', 1e-7)
    monkeypatch.setattr(_megakernel, '_FAILED', [])
    monkeypatch.setattr(_megakernel, '_WARNED', set())
    x, y, kwargs = _kernel_case(3000, 33)
    graph = _torch.train_networks(x, y, 4, kwargs, 'cuda', 3)
    monkeypatch.setattr(_torch, 'TRAIN_KERNEL', True)
    with pytest.warns(RuntimeWarning, match='barrier timeout'):
        nets = _torch.train_networks(x, y, 4, kwargs, 'cuda', 3)
    _same_networks(nets, graph, rtol=1e-12, atol=1e-15)
    assert _megakernel._FAILED == ['barrier timeout']


@pytest.mark.parametrize('case', ['separated', 'overlapping', 'elongated'])
def test_gaussian_mixture_matches_sklearn(case):
    # The device Gaussian mixture used to split ellipsoids gives the fit of
    # scikit-learn's GaussianMixture (same k-means initialisations, EM steps
    # and choice of the best run, including ties between runs that converge
    # to the same solution with swapped components) up to rounding.
    pytest.importorskip('torch')
    from scipy.stats import multivariate_normal
    from sklearn.mixture import GaussianMixture
    from nautilus._torch import gaussian_mixture
    rng = np.random.default_rng(0)
    if case == 'separated':
        points = np.concatenate([rng.normal(size=(300, 4)),
                                 rng.normal(size=(200, 4)) + 6])
    elif case == 'overlapping':
        points = np.concatenate([rng.normal(size=(250, 3)),
                                 rng.normal(size=(250, 3)) + 1.5])
    else:
        points = rng.normal(size=(500, 8)) * np.geomspace(1, 0.01, 8)
    for seed in range(3):
        gmm = GaussianMixture(n_components=2, n_init=10,
                              random_state=seed).fit(points)
        weights, means, covariances = gaussian_mixture(points, seed, 'cpu')
        assert np.allclose(weights, gmm.weights_, rtol=1e-9, atol=0)
        assert np.allclose(means, gmm.means_, rtol=1e-9, atol=1e-12)
        assert np.allclose(covariances, gmm.covariances_, rtol=1e-9,
                           atol=1e-12)
        labels = [np.argmax(np.array([multivariate_normal.logpdf(
            points, mean=m, cov=c) for m, c in zip(mu, cov)]), axis=0)
            for mu, cov in [(gmm.means_, gmm.covariances_),
                            (means, covariances)]]
        assert np.array_equal(labels[0], labels[1])


def test_union_split_device_matches_numpy():
    # Splitting an ellipsoid union on the device path (batched Gaussian
    # mixture, batched MVEE) gives the same ellipsoids as the numpy path and
    # leaves the random generator in the same state.
    pytest.importorskip('torch')
    from nautilus import bounds
    rng = np.random.default_rng(0)
    points = np.concatenate([rng.normal(size=(400, 3)) * 0.05 + 0.3,
                             rng.normal(size=(400, 3)) * 0.05 + 0.7])
    unions = []
    rngs = []
    for device in [None, 'cpu']:
        rngs.append(np.random.default_rng(1))
        union = bounds.Union.compute(points, n_points_min=50,
                                     rng=rngs[-1], device=device)
        while union.split(allow_overlap=False):
            pass
        unions.append(union)
    bounds.basic._MVEE_CACHE.clear()
    assert len(unions[0].bounds) == len(unions[1].bounds) > 1
    for a, b in zip(unions[0].bounds, unions[1].bounds):
        assert np.allclose(a.c, b.c, rtol=1e-9)
        assert np.allclose(a.A, b.A, rtol=1e-7)
    assert rngs[0].bit_generator.state == rngs[1].bit_generator.state


def test_warm_up_keeps_no_state():
    # The warm-up compiles and loads only: the MVEE cache is as before.
    pytest.importorskip('torch')
    from nautilus import _torch
    from nautilus.bounds import basic
    basic.minimum_volume_enclosing_ellipsoid_batched(np.eye(6, 2) + 0.5)
    cache = [id(entry) for entry in basic._MVEE_CACHE]
    seconds = _torch.warm_up('cpu')
    assert set(seconds) == {'mvee'}
    assert [id(entry) for entry in basic._MVEE_CACHE] == cache


def test_transfer_association_from_lowest_candidate_shell(monkeypatch):
    # sample_shell on the torch backend associates drawn points with shells
    # only from the lowest shell still holding transfer candidates up (and
    # not at all once none is left). The run is bitwise the one with the
    # association over all shells.
    pytest.importorskip('torch')
    from nautilus import _torch

    def likelihood(x):
        return np.logaddexp(-np.sum((x - 0.3)**2, axis=-1) / 0.005,
                            -np.sum((x - 0.75)**2, axis=-1) / 0.005)

    highest_bound = _torch.highest_bound
    starts = []

    def run(full):
        def wrapped(sampler, points, start, stop):
            if stop == len(sampler.bounds) - 1:  # the transfer association
                starts.append(start)
                if full:
                    start = 0
            return highest_bound(sampler, points, start, stop)

        monkeypatch.setattr(_torch, 'highest_bound', wrapped)
        sampler = Sampler(lambda x: x, likelihood, n_dim=3, n_live=100,
                          n_networks=1, vectorized=True, seed=0,
                          backend='torch')
        sampler.run(n_eff=200, verbose=False)
        return sampler

    full = run(True)
    starts.clear()
    part = run(False)
    assert max(starts) > 0
    assert part.n_like == full.n_like
    assert len(part.bounds) == len(full.bounds)
    assert part.log_z == full.log_z
    for a, b in zip(part.posterior(), full.posterior()):
        assert np.array_equal(a, b)
