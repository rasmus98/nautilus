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
            x = _torch._union_sample(
                bound, 50000, torch.Generator().manual_seed(0)).numpy()
            assert len(x) > 40000
        results.append([np.mean(x, axis=0), np.mean(
            bound.bounds[0].contains(x) & bound.bounds[1].contains(x)),
            bound.log_v])
    assert np.allclose(results[0][0], results[1][0], atol=3e-3)
    assert np.abs(results[0][1] - results[1][1]) < 0.01
    assert np.abs(results[0][2] - results[1][2]) < 0.01
