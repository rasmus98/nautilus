"""Split-minibatch HIP training kernel for the torch backend (opt-in).

`nautilus._torch.train_networks` uses this module only when
``nautilus._torch.TRAIN_KERNEL`` is set. It trains all networks of an
emulator in one persistent kernel per block of epochs (`_megakernel.hip`) and
returns the same weights, number of epochs and loss curves as the CUDA-graph
trainer, to rounding: the same initial weights, permutations (same generator
calls), Adam and early stopping. Anything the kernel does not support (no AMD
GPU, no ``hipcc``, ``alpha != 0``, shapes that do not fit, a failed build or
launch, a barrier timeout) raises `Unsupported`, and the caller falls back to
the graph trainer after one warning per reason.

The kernel is compiled with ``hipcc`` on first use, once per (source, GPU
architecture, dtype), into the directory ``$NAUTILUS_KERNEL_CACHE`` (default
``~/.cache/nautilus``), and loaded with ctypes. Shapes are runtime arguments.
"""

import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

SOURCE = Path(__file__).with_name('_megakernel.hip')
# Workgroups per network: each one takes 1/G of every minibatch's rows and
# applies Adam to 1/G of the parameters (MI250X optima of the kill test:
# float64 16, float32 27, both at 1024 threads).
GROUPS = {torch.float32: 27, torch.float64: 16}
THREADS = 1024
# LDS available to one workgroup, minus room for the kernel's static LDS.
LDS_BYTES = 65536 - 64
# Maximum number of weight layers (hidden layers + output layer).
MAX_LAYERS = 8
# Barrier timeout in seconds.
TIMEOUT = 5.0
HIPCC_FLAGS = ['-O3', '-fPIC', '-shared']

_LIBRARIES = {}
_WARNED = set()
_FAILED = []


class Unsupported(Exception):
    """The kernel cannot train this case; use the graph trainer."""


def warn_once(reason):
    """Warn once per distinct reason that the graph trainer is used."""
    if reason not in _WARNED:
        _WARNED.add(reason)
        warnings.warn('TRAIN_KERNEL: using the CUDA-graph trainer instead ' +
                      'of the HIP kernel ({}).'.format(reason), RuntimeWarning,
                      stacklevel=3)


def cache_path(source, arch, dtype, cache=None):
    """Path of the compiled library for a source text, architecture, dtype.

    Parameters
    ----------
    source : bytes
        Kernel source.
    arch : str
        GPU architecture as reported by torch, e.g. 'gfx90a:sramecc+:xnack-'.
    dtype : torch.dtype
        torch.float32 or torch.float64.
    cache : str or Path, optional
        Cache directory. Default is ``$NAUTILUS_KERNEL_CACHE`` or
        ``~/.cache/nautilus``.

    Returns
    -------
    path : Path
        Library path; the name contains a hash of the source and the flags.

    """
    if cache is None:
        cache = os.environ.get('NAUTILUS_KERNEL_CACHE', Path.home() /
                               '.cache' / 'nautilus')
    digest = hashlib.sha256(source + ' '.join(HIPCC_FLAGS).encode()).hexdigest()
    return Path(cache) / 'megakernel-{}-{}-{}.so'.format(
        digest[:16], re.sub('[^0-9A-Za-z]+', '_', arch).strip('_'),
        {torch.float32: 'f32', torch.float64: 'f64'}[dtype])


def layout(units, batch_size, n_networks, dtype, max_blocks, groups=None,
           threads=THREADS, lds_bytes=LDS_BYTES):
    """Choose the kernel's work split and memory layout.

    Parameters
    ----------
    units : list of int
        Layer widths from the input to the single output.
    batch_size : int
        Minibatch size.
    n_networks : int
        Number of networks.
    dtype : torch.dtype
        Network dtype.
    max_blocks : callable
        ``max_blocks(ldsw, threads, lds)`` returns the number of workgroups
        that can be co-resident on the device.
    groups : int, optional
        Maximum workgroups per network. Default is ``GROUPS[dtype]``.
    threads : int, optional
        Threads per workgroup.
    lds_bytes : int, optional
        LDS available to one workgroup.

    Returns
    -------
    plan : dict
        G (workgroups per network), R (rows per workgroup), RC (rows per
        chunk), nch (chunks), S (gradient slots per network), NP (parameters
        per network), ow/ob (weight and bias offsets), act (activation
        offsets), ASZ (activation size), ldsw (weights in LDS), lds (bytes).

    Raises
    ------
    Unsupported
        If the layers do not fit or the networks cannot be co-resident.

    """
    if units[-1] != 1 or len(units) < 2 or len(units) - 1 > MAX_LAYERS:
        raise Unsupported('network shape {}'.format(units))
    size = torch.finfo(dtype).bits // 8
    ow, ob, n_params = [], [], 0
    for fan_in, fan_out in zip(units[:-1], units[1:]):
        ow.append(n_params)
        n_params += fan_in * fan_out
        ob.append(n_params)
        n_params += fan_out
    # Per chunk row: activations of every layer, target and residual.
    per_row = sum(units) + 2
    ldsw = (dtype == torch.float32 and
            size * (n_params + 4 * per_row + 4) <= lds_bytes)
    rc_max = (lds_bytes // size - 4 - ldsw * n_params) // per_row // 4 * 4
    if rc_max < 4:
        raise Unsupported('layers too wide for LDS: {}'.format(units))
    lds = None
    for g in range(min(groups or GROUPS[dtype], -(-batch_size // 4)), 0, -1):
        rows = -(-batch_size // g)
        rc = min(-(-rows // 4) * 4, rc_max)
        lds = size * (ldsw * n_params + rc * per_row + 4)
        if n_networks * g <= max_blocks(ldsw, threads, lds):
            break
    else:
        raise Unsupported('{} networks cannot be co-resident'.format(
            n_networks))
    act = list(np.cumsum([0] + [rc * u for u in units]))
    return dict(G=g, R=rows, RC=rc, nch=-(-rows // rc), S=g * -(-rows // rc),
                NP=n_params, ow=ow, ob=ob, act=[int(a) for a in act[:-1]],
                ASZ=int(act[-1]) + 4, ldsw=bool(ldsw), lds=lds,
                threads=threads)


def _hipcc():
    for path in [shutil.which('hipcc'), os.path.join(os.environ.get(
            'ROCM_PATH', '/opt/rocm'), 'bin', 'hipcc')]:
        if path and os.access(path, os.X_OK):
            return path
    raise Unsupported('hipcc not found')


def _hip_runtimes():
    """Distinct HIP runtime libraries mapped into this process."""
    try:
        with open('/proc/self/maps') as stream:
            return {os.path.realpath(line.split()[-1]) for line in stream if
                    'libamdhip64' in line}
    except OSError:
        return set()


def library(device, dtype):
    """Compiled kernel library and device properties, built on first use.

    Raises `Unsupported` if the device is not an AMD GPU or the library cannot
    be built or loaded. The outcome is remembered per device and dtype.
    """
    if device.type != 'cuda' or torch.version.hip is None:
        raise Unsupported('needs an AMD GPU and a ROCm build of torch')
    key = (str(device), dtype)
    if key not in _LIBRARIES:
        try:
            _LIBRARIES[key] = _load(device, dtype)
        except Unsupported as error:
            _LIBRARIES[key] = error
    if isinstance(_LIBRARIES[key], Unsupported):
        raise _LIBRARIES[key]
    return _LIBRARIES[key]


def _load(device, dtype):
    source = SOURCE.read_bytes()
    arch = torch.cuda.get_device_properties(device).gcnArchName
    path = cache_path(source, arch, dtype)
    if not path.exists():
        hipcc = _hipcc()
        path.parent.mkdir(parents=True, exist_ok=True)
        real = {torch.float32: 'float', torch.float64: 'double'}[dtype]
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            out = Path(tmp) / path.name
            result = subprocess.run(
                [hipcc, *HIPCC_FLAGS, '--offload-arch=' + arch,
                 '-DMK_REAL=' + real, '-o', str(out), str(SOURCE)],
                capture_output=True, text=True)
            if result.returncode != 0:
                raise Unsupported('hipcc failed: ' + (
                    result.stderr or result.stdout)[-500:])
            os.replace(out, path)  # atomic: concurrent builds are harmless
    torch.cuda.init()
    runtimes = _hip_runtimes()
    lib = ctypes.CDLL(str(path))
    if len(_hip_runtimes()) > max(1, len(runtimes)):
        raise Unsupported('the kernel loads a second HIP runtime')
    lib.mk_error.restype = ctypes.c_char_p
    lib.mk_query.argtypes = [ctypes.c_int] * 3 + [ctypes.POINTER(
        ctypes.c_int)] * 3
    lib.mk_train.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(
        ctypes.c_double), ctypes.POINTER(ctypes.c_void_p), ctypes.c_longlong,
        ctypes.c_void_p]
    if lib.mk_real_size() != torch.finfo(dtype).bits // 8:
        raise Unsupported('library dtype mismatch')
    return lib


def _check(lib, code, what):
    if code != 0:
        raise Unsupported('{}: {}'.format(what, lib.mk_error(code).decode()))


def train(x, y, params, network, gen, batch_size, check_every, lr_block):
    """Train stacked parameters in place with the kernel.

    Parameters
    ----------
    x, y : torch.Tensor
        Inputs and targets on the device, in the network dtype.
    params : list of torch.Tensor
        Stacked weights and biases (n_networks, fan_in, fan_out) and
        (n_networks, 1, fan_out), initialized; overwritten with the result.
    network : sklearn.neural_network.MLPRegressor
        Settings (max_iter, tol, n_iter_no_change, Adam parameters).
    gen : torch.Generator
        Generator of the permutations, in the state after the initialization.
    batch_size : int
        Minibatch size.
    check_every : int
        Epochs per launch; the host checks for stopped networks in between.
    lr_block : callable
        ``lr_block(epoch_0, n_epochs)`` returns the step sizes of the epochs
        as a tensor of shape (n_epochs, steps per epoch).

    Returns
    -------
    loss_curve, active_curve, best_loss : torch.Tensor
        As returned by the graph trainer.

    Raises
    ------
    Unsupported
        If the kernel cannot run this case or fails (launch error, barrier
        timeout); the parameters are then undefined.

    """
    if _FAILED:
        raise Unsupported(_FAILED[0])
    device, dtype = x.device, x.dtype
    n_networks, (n_points, n_dim) = len(params[0]), x.shape
    units = [n_dim] + [int(p.shape[2]) for p in params[::2]]
    lib = library(device, dtype)

    with torch.cuda.device(device):
        props = [ctypes.c_int() for _ in range(3)]

        @lru_cache(maxsize=None)
        def max_blocks(ldsw, threads, lds):
            _check(lib, lib.mk_query(int(ldsw), threads, lds,
                                     *map(ctypes.byref, props)),
                   'occupancy query')
            return props[0].value * props[1].value

        plan = layout(units, batch_size, n_networks, dtype, max_blocks)
        timeout = int(TIMEOUT * props[2].value * 1000)
        g, n_params = plan['G'], plan['NP']
        x, y = x.contiguous(), y.contiguous()
        w = torch.cat([p.detach().reshape(n_networks, -1) for p in params],
                      dim=1).contiguous()
        m, v = torch.zeros_like(w), torch.zeros_like(w)
        part = torch.empty(n_networks * plan['S'] * n_params, dtype=dtype,
                           device=device)
        lpart = torch.zeros(n_networks * g, dtype=dtype, device=device)
        best_loss = torch.full((n_networks, ), np.inf, dtype=dtype,
                               device=device)
        istate = torch.zeros(2 * n_networks + 1, dtype=torch.int32,
                             device=device)
        istate[n_networks:2 * n_networks] = 1
        bar = torch.zeros(2 * n_networks, dtype=torch.int32, device=device)
        loss_curve = torch.zeros((network.max_iter, n_networks), dtype=dtype,
                                 device=device)
        active_curve = torch.zeros((network.max_iter, n_networks),
                                   dtype=torch.bool, device=device)
        perm = torch.empty((check_every, n_networks, n_points),
                           dtype=torch.int32, device=device)
        ints = [n_networks, g, plan['threads'], plan['lds'], plan['ldsw'],
                n_points, batch_size, 0, 0, len(units) - 1, n_params,
                plan['R'], plan['RC'], plan['nch'], plan['S'],
                min(network.n_iter_no_change, 2**31 - 1), plan['ASZ'],
                *units, *plan['ow'], *plan['ob'], *plan['act']]
        ints = (ctypes.c_int * len(ints))(*map(int, ints))
        floats = (ctypes.c_double * 4)(network.beta_1, network.beta_2,
                                       network.epsilon, network.tol)
        stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)

        for epoch_0 in range(0, network.max_iter, check_every):
            n_epochs = min(check_every, network.max_iter - epoch_0)
            lr = lr_block(epoch_0, n_epochs).contiguous()
            for i in range(n_epochs):
                perm[i].copy_(torch.argsort(torch.rand(
                    (n_networks, n_points), generator=gen, device=device),
                    dim=1))
            ints[7], ints[8] = n_epochs, epoch_0
            pointers = (ctypes.c_void_p * 14)(*[t.data_ptr() for t in (
                x, y, perm, lr, w, m, v, part, lpart, best_loss, loss_curve,
                istate, bar, active_curve)])
            code = lib.mk_train(ints, floats, pointers, timeout, stream)
            if code != 0:
                _FAILED.append('launch failed: ' + lib.mk_error(code).decode())
                raise Unsupported(_FAILED[0])
            state = istate.cpu()
            if state[-1]:
                _FAILED.append('barrier timeout')
                raise Unsupported(_FAILED[0])
            if not torch.any(state[n_networks:2 * n_networks]):
                break

        with torch.no_grad():
            start = 0
            for p in params:
                stop = start + p[0].numel()
                p.copy_(w[:, start:stop].view(p.shape))
                start = stop
    return loss_curve, active_curve, best_loss
