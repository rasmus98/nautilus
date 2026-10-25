"""Fused HIP membership kernels for the torch backend's draws (opt-in).

`nautilus._torch.sample` uses this module only when
``nautilus._torch.DRAW_KERNEL`` is set, and `nautilus._torch.highest_bound`
only when `HIGHEST` is set as well. `accept` replaces the union
multiplicity and keep test, the unit-cube test, the outer-ellipsoid tests and
the networks of the neural bounds of one draw pass with one kernel launch;
`highest` replaces the batched bound tests and the block-wise network search
of `highest_bound` with one launch that tests the bounds from the last down
(`_drawkernel.hip`). The points themselves are still drawn by torch with the
same generator calls, so with identical decisions the samples are identical.

Ellipsoid tests run in float64 and networks in the network dtype
(``nautilus._torch.NN_DTYPE``), as in the torch path; only the summation
order differs, so decisions agree with the torch path to rounding.

Anything the kernel does not support (no AMD GPU, no ``hipcc``, a failed
build or launch, more than 64 dimensions or neural bounds, networks other
than three hidden layers of which the second and third have at most 64 and
32 units, periodic phase shifts) raises `Unsupported`, and the caller uses the
torch path after one warning per reason.

The kernels are compiled with ``hipcc`` on first use, once per (source, GPU
architecture, network dtype, padded dimension, second and third hidden
widths), into ``$NAUTILUS_KERNEL_CACHE`` (default ``~/.cache/nautilus``),
and loaded with ctypes. The register arrays need these sizes at compile time;
all other shapes (points, bounds, networks, first hidden width) are runtime
arguments.
"""

import ctypes
import hashlib
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

from ._megakernel import _hip_runtimes, _hipcc

SOURCE = Path(__file__).with_name('_drawkernel.hip')
HIPCC_FLAGS = ['-O3', '-fPIC', '-shared']
# Dimensions are padded to a multiple of this in registers.
PAD = 4
MAX_DIM = 64
MAX_NEURAL = 64
MAX_H2, MAX_H3 = 64, 32
# LDS bytes for one staged chunk of network parameters.
LDS_BUDGET = 24576
# Use the `highest` kernel in `highest_bound`. Off: on a 33-dimensional
# production run (62 shells) its per-point top-down search is slower than
# the torch path's batched tests (each thread walks the bounds serially, and
# shell association passes only 256 points), so `DRAW_KERNEL` covers the
# draws only unless this is set.
HIGHEST = False

_LIBRARIES = {}
_WARNED = set()


class Unsupported(Exception):
    """The kernel cannot handle this case; use the torch path."""


def warn_once(reason):
    """Warn once per distinct reason that the torch path is used."""
    if reason not in _WARNED:
        _WARNED.add(reason)
        warnings.warn('DRAW_KERNEL: using the torch path instead of the ' +
                      'HIP kernel ({}).'.format(reason), RuntimeWarning,
                      stacklevel=3)


def padded(n_dim):
    """Number of dimensions held in registers for `n_dim` dimensions."""
    return -(-n_dim // PAD) * PAD


def cache_path(source, arch, dtype, md, h2, h3, cache=None):
    """Path of the compiled library for a source text and its parameters.

    Parameters
    ----------
    source : bytes
        Kernel source.
    arch : str
        GPU architecture as reported by torch, e.g. 'gfx90a:sramecc+:xnack-'.
    dtype : torch.dtype
        Network dtype, torch.float32 or torch.float64.
    md, h2, h3 : int
        Padded dimension and second and third hidden widths.
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
    return Path(cache) / 'drawkernel-{}-{}-{}-d{}-{}-{}.so'.format(
        digest[:16], re.sub('[^0-9A-Za-z]+', '_', arch).strip('_'),
        {torch.float32: 'f32', torch.float64: 'f64'}[dtype], md, h2, h3)


def library(device, dtype, md, h2, h3):
    """Compiled kernel library, built on first use.

    Raises `Unsupported` if the device is not an AMD GPU or the library cannot
    be built or loaded. The outcome is remembered per device and parameters.
    """
    if device.type != 'cuda' or torch.version.hip is None:
        raise Unsupported('needs an AMD GPU and a ROCm build of torch')
    if dtype not in (torch.float32, torch.float64):
        raise Unsupported('network dtype {}'.format(dtype))
    key = (str(device), dtype, md, h2, h3)
    if key not in _LIBRARIES:
        try:
            _LIBRARIES[key] = _load(device, dtype, md, h2, h3)
        except Unsupported as error:
            _LIBRARIES[key] = error
    if isinstance(_LIBRARIES[key], Unsupported):
        raise _LIBRARIES[key]
    return _LIBRARIES[key]


def _load(device, dtype, md, h2, h3):
    source = SOURCE.read_bytes()
    arch = torch.cuda.get_device_properties(device).gcnArchName
    path = cache_path(source, arch, dtype, md, h2, h3)
    if not path.exists():
        hipcc = _hipcc()
        path.parent.mkdir(parents=True, exist_ok=True)
        real = {torch.float32: 'float', torch.float64: 'double'}[dtype]
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            out = Path(tmp) / path.name
            result = subprocess.run(
                [hipcc, *HIPCC_FLAGS, '--offload-arch=' + arch,
                 '-DDK_REAL=' + real, '-DDK_MD={}'.format(md),
                 '-DDK_H2={}'.format(h2), '-DDK_H3={}'.format(h3), '-o',
                 str(out), str(SOURCE)], capture_output=True, text=True)
            if result.returncode != 0:
                raise Unsupported('hipcc failed: ' + (
                    result.stderr or result.stdout)[-500:])
            os.replace(out, path)  # atomic: concurrent builds are harmless
    torch.cuda.init()
    runtimes = _hip_runtimes()
    lib = ctypes.CDLL(str(path))
    if len(_hip_runtimes()) > max(1, len(runtimes)):
        raise Unsupported('the kernel loads a second HIP runtime')
    lib.dk_error.restype = ctypes.c_char_p
    for name in ('dk_accept', 'dk_highest'):
        getattr(lib, name).argtypes = [ctypes.POINTER(ctypes.c_longlong),
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.c_void_p]
    if (lib.dk_real_size() != torch.finfo(dtype).bits // 8 or
            (lib.dk_md(), lib.dk_h2(), lib.dk_h3()) != (md, h2, h3)):
        raise Unsupported('library parameter mismatch')
    return lib


def _shapes(networks):
    """(H1, H2, H3) of stacked networks; Unsupported for other shapes."""
    if len(networks['w']) != 3:
        raise Unsupported('networks with {} hidden layers'.format(
            len(networks['w'])))
    h1 = networks['w'][0].shape[2]
    h2, h3 = networks['w'][1].shape[2], networks['w'][2].shape[2]
    if h2 > MAX_H2 or h3 > MAX_H3 or networks['w'][2].shape[3] != 1:
        raise Unsupported('hidden widths {} > ({}, {})'.format(
            (h1, h2, h3), MAX_H2, MAX_H3))
    return h1, h2, h3


def _pad_vectors(a, md, fill=0.0):
    out = a.new_full((a.shape[0], md), fill)
    out[:, :a.shape[1]] = a
    return out


def _pad_matrices(a, md):
    out = a.new_zeros((a.shape[0], md, md))
    out[:, :a.shape[1], :a.shape[2]] = a
    return out


def _cube_masks(dim_cube):
    bits = torch.as_tensor(
        np.left_shift(np.uint64(1), np.arange(dim_cube.shape[1],
                                              dtype=np.uint64)).view(np.int64),
        device=dim_cube.device)
    return torch.sum(torch.where(dim_cube, bits, 0), dim=1)


def _pack_networks(networks, md, h1, h2, h3):
    """Parameters of each network of each neural bound, flattened in the
    kernel's layout: W0^T (H1 x md), b0, W1, b1, W2, b2, W3, b3."""
    w_0, b_0 = networks['w_0'], networks['b_0']
    n_bounds, n_dim, _ = w_0.shape
    n_net = networks['n_networks']
    w0 = w_0.view(n_bounds, n_dim, n_net, h1).permute(0, 2, 3, 1)
    w0 = torch.cat([w0, w0.new_zeros((n_bounds, n_net, h1, md - n_dim))], 3)
    parts = [w0.reshape(n_bounds, n_net, -1),
             b_0.view(n_bounds, n_net, h1)]
    for w, b in zip(networks['w'], networks['b']):
        parts += [w.reshape(n_bounds, n_net, -1),
                  b.reshape(n_bounds, n_net, -1)]
    return torch.cat(parts, 2).contiguous()


def pack(part, n_dim, dtype):
    """Kernel tensors of one `_bound_stack` (cached in it)."""
    md = padded(n_dim)
    key = ('dk', md, dtype)
    if key in part:
        return part[key]
    if n_dim > MAX_DIM:
        raise Unsupported('{} dimensions > {}'.format(n_dim, MAX_DIM))
    if len(part['c_n']) > MAX_NEURAL:
        raise Unsupported('more than {} neural bounds'.format(MAX_NEURAL))
    networks = part['networks']
    shapes = (0, 1, 1) if networks is None else _shapes(networks)
    device = part['c_m'].device
    packed = dict(
        md=md, shapes=shapes, n_net=0 if networks is None else
        networks['n_networks'],
        mc=_pad_vectors(part['c_m'], md).contiguous(),
        mB=_pad_matrices(part['B_inv_T_m'], md).contiguous(),
        mcube=_cube_masks(part['dim_cube']).contiguous())
    k = len(part['c_n'])
    if k > 0:
        packed['nc'] = _pad_vectors(part['c_n'], md).contiguous()
        packed['nB'] = _pad_matrices(part['B_inv_T_n'], md).contiguous()
    else:
        packed['nc'] = torch.zeros((0, md), dtype=torch.float64,
                                   device=device)
        packed['nB'] = torch.zeros((0, md, md), dtype=torch.float64,
                                   device=device)
    if networks is not None and k > 0:
        packed['mean'] = _pad_vectors(networks['mean'], md).contiguous()
        packed['scale'] = _pad_vectors(networks['scale'], md, 1.0).contiguous()
        packed['w'] = _pack_networks(networks, md, *shapes)
        packed['thr'] = networks['threshold'].contiguous()
    part[key] = packed
    return packed


def _pointer(tensor):
    return None if tensor is None else tensor.data_ptr()


def _launch(name, packed, ints, tensors, device):
    h1, h2, h3 = packed['shapes']
    dtype = packed['w'].dtype if 'w' in packed else torch.float32
    lib = library(device, dtype, packed['md'], max(h2, 1), max(h3, 1))
    ints = (ctypes.c_longlong * len(ints))(*map(int, ints))
    pointers = (ctypes.c_void_p * len(tensors))(*map(_pointer, tensors))
    with torch.cuda.device(device):
        stream = ctypes.c_void_p(torch.cuda.current_stream(
            device).cuda_stream)
        code = getattr(lib, name)(ints, pointers, stream)
    if code != 0:
        raise Unsupported('launch failed: ' + lib.dk_error(code).decode())


def lds_plan(md, h1, h2, h3, size):
    """First-layer units per staged chunk and the LDS bytes of the two
    `dk_accept` kernels (union: one ellipsoid; networks: an ellipsoid with
    mean and scale, a chunk of the first two layers, or the rest)."""
    chunk = max(1, min(max(h1, 1), LDS_BUDGET // ((md + 1 + h2) * size)))
    union = 8 * (md + md * md)
    net = max(8 * (md * md + 3 * md), size * chunk * (md + 1 + h2),
              size * (h2 + h2 * h3 + 2 * h3 + 1))
    if max(union, net) > 65536:
        raise Unsupported('LDS {} bytes'.format(max(union, net)))
    return chunk, union, net


def accept(packed, x, u, union_cube):
    """Keep and accept decisions of one draw pass, see `_drawkernel.hip`.

    Parameters
    ----------
    packed : dict
        Kernel tensors of the bound, see `pack`.
    x : torch.Tensor
        Drawn points (float64, on the device).
    u : torch.Tensor
        A uniform number for each point (float64).
    union_cube : bool
        Whether the outer union is restricted to the unit cube.

    Returns
    -------
    keep, accept : torch.Tensor
        Boolean tensors; `accept` implies `keep`.

    """
    n, n_dim = x.shape
    x, u = x.contiguous(), u.contiguous()
    keep = torch.empty(n, dtype=torch.bool, device=x.device)
    acc = torch.empty(n, dtype=torch.bool, device=x.device)
    h1, h2, h3 = packed['shapes']
    k = len(packed['nc'])
    has_net = 'w' in packed
    size = packed['w'].element_size() if has_net else 4
    chunk, lds_union, lds_net = lds_plan(packed['md'], h1, h2, h3, size)
    count = torch.zeros(1, dtype=torch.int32, device=x.device)
    rows = torch.empty(n if has_net else 0, dtype=torch.int64, device=x.device)
    todo = torch.empty_like(rows)
    ints = [n, n_dim, len(packed['mc']), k, h1, packed['n_net'],
            packed['w'].shape[2] if has_net else 0, union_cube, has_net, chunk,
            lds_union, lds_net]
    tensors = [x, u, packed['mc'], packed['mB'], packed['mcube'],
               packed['nc'], packed['nB'], packed.get('mean'),
               packed.get('scale'), packed.get('w'), packed.get('thr'), keep,
               acc, count, rows, todo]
    _launch('dk_accept', packed, ints, tensors, x.device)
    return keep, acc


def highest(packed, x, mstart, nstart, bcube, first):
    """`highest_bound` for bounds `first` to `first + len(bcube) - 1`.

    Parameters
    ----------
    packed : dict
        Kernel tensors of the sampler's stacked bounds, see `pack`.
    x : torch.Tensor
        Points (float64, on the device).
    mstart, nstart : torch.Tensor
        First mixture and first neural bound (int64, on the device) of each
        bound in the range, and one past the last.
    bcube : torch.Tensor
        Whether each bound's outer union is restricted to the unit cube
        (bool, on the device).
    first : int
        Index of the first bound of the range.

    Returns
    -------
    result : torch.Tensor
        Largest index of a bound containing each point, or -1 (int64).

    """
    n, n_dim = x.shape
    x = x.contiguous()
    result = torch.empty(n, dtype=torch.int64, device=x.device)
    h1, h2, h3 = packed['shapes']
    has_net = 'w' in packed
    ints = [n, n_dim, len(packed['mc']), len(packed['nc']), h1,
            packed['n_net'], packed['w'].shape[2] if has_net else 0, 0,
            has_net, len(bcube), first]
    tensors = [x, None, packed['mc'], packed['mB'], packed['mcube'],
               packed['nc'], packed['nB'], packed.get('mean'),
               packed.get('scale'), packed.get('w'), packed.get('thr'),
               mstart, nstart, bcube, None, None, result]
    _launch('dk_highest', packed, ints, tensors, x.device)
    return result
