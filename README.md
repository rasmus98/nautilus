# nautilus (unofficial GPU fork)

> **Use at your own risk.** This is an unofficial, experimental fork of
> [nautilus](https://github.com/johannesulf/nautilus) by Johannes U. Lange. It is not affiliated with or endorsed by
> the nautilus author. **All credit for the method and the original code goes to the original nautilus.** If you use
> this fork, please cite the nautilus paper (Lange 2023, MNRAS 525, 3181) and the
> [original repository](https://github.com/johannesulf/nautilus).
>
> The fork only changes how the same algorithm is computed (an optional PyTorch/GPU backend). It is provided as is,
> without warranty of any kind (see LICENSE). It has been tested only on AMD MI250X GPUs (ROCm) and x86 CPUs, and it
> may contain bugs that the original does not. Please report problems with the fork in this repository, not upstream.
> For production work, the [original nautilus](https://github.com/johannesulf/nautilus) is the safe choice.

A fork of [nautilus](https://github.com/johannesulf/nautilus) 1.0.6 (Johannes U. Lange, MIT) that adds an optional
PyTorch backend. With `backend='torch'` the neural-network emulators, ellipsoid fits and bound sampling run on a GPU
(or on the CPU through PyTorch). Without a `backend` argument the fork **is nautilus 1.0.6**: seeded runs give the same
points, likelihood calls, shells and evidence bit for bit.

Please cite the nautilus paper (Lange 2023, MNRAS 525, 3181) for the method. This fork changes only how the same
algorithm is computed.

## When to use it
- **Likelihoods that already run on a GPU** (neural likelihoods, emulators): the sampler's own work (network
  training, ellipsoid fits, bound draws) moves to the same device, so it stops dominating the run time.
- **Cheap likelihoods, many dimensions**: on synthetic problems the fork on one GPU ran 11-29x faster than serial
  nautilus 1.0.6 at the same settings, with the same evidence accuracy within the seed scatter.
- **CPU only**: the torch backend on one core took 0.72-0.91x the time of serial nautilus 1.0.6 on 5 synthetic problems
  (float64 networks; 0.48-0.53x with float32 networks, whose accuracy was tested on the GPU only). Stock nautilus with
  `pool=(None, 4)` (below) is the other CPU option.

## Installation
```
pip install "nautilus-sampler[torch] @ git+https://github.com/rasmus98/nautilus@gpu"
```
PyTorch >= 2.1. The optional HIP kernels (AMD GPUs) need `hipcc` at first use; numba is optional (CPU helpers).

## Usage
```python
from nautilus import Sampler
sampler = Sampler(prior, likelihood, n_live=500, backend='torch', device='cuda')
sampler.run(verbose=True)
```
Everything else (prior, likelihood, `run`, `posterior`, checkpoints) is the nautilus 1.0.6 API.

Options of the torch backend (module attributes of `nautilus._torch`, set before creating the sampler):

| setting | default | effect |
|---|---|---|
| `NN_DTYPE` | `torch.float64` | network precision; `torch.float32` is about 1.6x faster per training step |
| `DRAW_KERNEL` | `False` | fused HIP kernels for bound membership in draws (AMD only; same draws bit for bit) |
| `TRAIN_KERNEL` | `False` | HIP split-minibatch training kernel (AMD only; same networks to rounding) |
| `warm_up(device)` | - | pay kernel builds and first launches once per process, before timing anything |

The fused draw kernel is compiled with `hipcc` the first time a given dimension and precision is used (about 8 s) and
cached in `$NAUTILUS_KERNEL_CACHE` (default `~/.cache/nautilus`); later runs and processes reuse it. The numba
helper of the batched ellipsoid fit is cached next to the installed package, so install the fork once (for example with
`pip install`) rather than running it from a fresh copy each time.

Differences from 1.0.6 on the torch backend (by design, all within the method):
- every neural bound draws its own training seed from the sampler's generator (1.0.6 uses fixed seeds 0..3);
- ellipsoids are fitted with a batched form of the same Khachiyan algorithm (agrees to ~1e-12);
- bound draws are oversampled to the expected need instead of fixed passes of 1000 (i.i.d. rejection sampling does not
  depend on the batch size).
So torch-backend runs are a different realisation of the same sampler, not bitwise copies of 1.0.6.

## Benchmarks
Settings: n_live 500, n_eff 2000, Nautilus defaults otherwise; synthetic problems with known evidence; LUMI (AMD EPYC
CPU cores; one AMD MI250X GCD). Fork, 1 core = torch backend on the CPU, float64 networks, speed-up against vanilla
run on the same node at the same time. Fork, 1 GPU = float32 networks with the fused kernels.

| problem | vanilla, 1 core | vanilla, pool of 4 | fork, 1 core | fork, 1 GPU |
|---|---|---|---|---|
| 2 Gaussian shells, 2-d | 70 s | 2.8x faster | 1.3x | 2.4 s (29x) |
| correlated Gaussian, 8-d | 68 s | 1.05x | 1.3x | 5.1 s (13x) |
| Rosenbrock-type banana, 8-d | 257 s | 1.4x | 1.1x | 10.2 s (25x) |
| LogGamma (4 modes), 10-d | 131 s | 1.2x | 1.3x | 12.0 s (11x) |
| correlated Gaussian, 16-d | 186 s | 1.1x | 1.4x | 13.7 s (14x) |
| correlated Gaussian, 32-d | 920 s | - | 1.1x (timed before the last trainer change) | 86 s (11x) |

The fork's log Z agrees with vanilla's within the seed scatter (0.01-0.03 nats) on every problem. Method, figures,
accuracy tables and caveats: [docs/fork/BENCHMARKS.md](docs/fork/BENCHMARKS.md).

## Recommended CPU setting
`Sampler(..., pool=(None, 4))` on 4 cores with one BLAS thread per process (`OMP_NUM_THREADS=1`) trains the four
networks in parallel: 1.05-2.8x faster than serial in the benchmarks, and faster than one process with 4 BLAS threads (which
was slower than 1 thread on most problems). This works with stock nautilus; it changes the random realisation, not the
algorithm.

## Status
Research code; tested on LUMI (AMD MI250X, ROCm 6.2, PyTorch 2.7). The torch backend is generic PyTorch and should
run on any PyTorch device, but it has not been tested on NVIDIA GPUs. `DRAW_KERNEL` and `TRAIN_KERNEL` are HIP code
for AMD GPUs only; elsewhere they fall back to the torch path with a warning. The default backend runs
the upstream test suite unchanged. Bug reports welcome; for anything about the method, see upstream.

---

# Original nautilus README

The text below is the upstream README of nautilus 1.0.6, unchanged (badges and links refer to the original project).

![Logo](https://raw.githubusercontent.com/johannesulf/nautilus/main/docs/nautilus_text_image.png "Logo")

[![Unit Testing Status](https://img.shields.io/github/actions/workflow/status/johannesulf/nautilus/tests.yml?branch=main&label=tests)](https://github.com/johannesulf/nautilus/actions)
[![Documentation Status](https://img.shields.io/readthedocs/nautilus-sampler)](https://nautilus-sampler.readthedocs.io/en/latest/)
[![Code Coverage](https://img.shields.io/coverallsCoverage/github/johannesulf/nautilus)](https://coveralls.io/github/johannesulf/nautilus?branch=main)
[![PyPI](https://img.shields.io/pypi/v/nautilus-sampler)](https://pypi.org/project/nautilus-sampler/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/nautilus-sampler)](https://pypi.org/project/nautilus-sampler/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/nautilus-sampler)](https://anaconda.org/conda-forge/nautilus-sampler)
[![Conda - Downloads](https://img.shields.io/conda/dn/conda-forge/nautilus-sampler)](https://anaconda.org/conda-forge/nautilus-sampler)
[![License: MIT](https://img.shields.io/github/license/johannesulf/nautilus)](https://raw.githubusercontent.com/johannesulf/nautilus/main/LICENSE)
![Language: Python](https://img.shields.io/github/languages/top/johannesulf/nautilus)

Nautilus is an MIT-licensed pure-Python package for Bayesian posterior and evidence estimation. It utilizes importance sampling and efficient space exploration using neural networks. Compared to traditional MCMC and Nested Sampling codes, it often needs fewer likelihood calls and produces much larger posterior samples. Additionally, nautilus is highly accurate and produces Bayesian evidence estimates with percent precision. It is widely used in many areas of astrophysical research.

## Example

This example, sampling a 3-dimensional Gaussian, illustrates how to use nautilus.

```python
import corner
import numpy as np
from nautilus import Prior, Sampler
from scipy.stats import multivariate_normal

prior = Prior()
for key in 'abc':
    prior.add_parameter(key)

def likelihood(param_dict):
    x = [param_dict[key] for key in 'abc']
    return multivariate_normal.logpdf(x, mean=[0.4, 0.5, 0.6], cov=0.01)

sampler = Sampler(prior, likelihood)
sampler.run(verbose=True)
points, log_w, log_l = sampler.posterior()
corner.corner(points, weights=np.exp(log_w), labels='abc')
```

## Installation

The most recent stable version of nautilus is listed in the Python Package Index (PyPI) and can be installed via ``pip``.

```shell
pip install nautilus-sampler
```

Additionally, nautilus is also on conda-forge. To install via ``conda`` use the following command.

```shell
conda install -c conda-forge nautilus-sampler
```

## Documentation

You can find the documentation at [nautilus-sampler.readthedocs.io](https://nautilus-sampler.readthedocs.io).

## Attribution

A paper describing nautilus's underlying methods and performance has been published in the [Monthly Notices of the Royal Astronomical Society](https://academic.oup.com/mnras/article/525/2/3181/7243406). A draft of the paper is also available on [arXiv](https://arxiv.org/abs/2306.16923). Please cite the paper if you find nautilus helpful in your research.

```
@article{nautilus,
    author = {Lange, Johannes U},
    title = "{nautilus: boosting Bayesian importance nested sampling with deep learning}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {525},
    number = {2},
    pages = {3181-3194},
    year = {2023},
    month = {08},
    doi = {10.1093/mnras/stad2441},
    url = {https://doi.org/10.1093/mnras/stad2441},
    eprint = {https://academic.oup.com/mnras/article-pdf/525/2/3181/51331635/stad2441.pdf},
}
```

## License

Nautilus is licensed under the MIT License. The logo uses an image from the Illustris Collaboration.
