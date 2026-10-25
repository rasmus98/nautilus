// Split-minibatch persistent training kernel for an ensemble of ReLU MLPs with one output (squared loss, scikit-learn
// Adam, no L2 penalty). Used by nautilus/_torch.py when TRAIN_KERNEL is set; built and launched by
// nautilus/_megakernel.py, which compiles this file with hipcc once per (source, GPU architecture, dtype), passing
// -DMK_REAL=float or -DMK_REAL=double, and loads it with ctypes. All pointers are torch device tensors.
//
// Grid: n_net * G workgroups, launched cooperatively so that all are co-resident. Workgroup (net, g) owns rows
// [g R, (g + 1) R) of every minibatch of network `net` and processes them in chunks of at most RC rows held in LDS.
// Per Adam step it gathers its rows through the epoch's permutation, runs forward and backward and writes one partial
// gradient per chunk (a "slot") to global memory. The G workgroups of the network then meet at a barrier, each sums
// the slots for its 1/G slice of the parameters and applies Adam to that slice, and they meet again.
//
// Early stopping (tol, n_iter_no_change) is decided per network at the end of every epoch, as in the graph trainer:
// the epoch loss is half the sum of the squared errors of the epoch's minibatch forward passes (before each update)
// divided by N. Every workgroup of a network sums the G per-workgroup partial sums in the same order and so reaches
// the same decision. A stopped network's workgroups leave the epoch loop, which freezes its weights.
//
// Shapes are runtime arguments: number of layers (<= MAXL), layer widths, rows, batch size, G, chunk rows. Only the
// weight placement is a template argument (LDSW: weights copied to LDS each step, else read from global memory), so
// that LDS reads stay LDS instructions. LDS is dynamic and sized by the host.
//
// Barriers time out after `timeout` wall-clock ticks: the workgroup then sets the error flag and returns, and every
// other workgroup returns at its next barrier. The host checks the flag after each launch and discards the result.
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef MK_REAL
#define MK_REAL double
#endif

namespace {
typedef MK_REAL real;
constexpr int MAXL = 8;  // weight layers (hidden layers + output layer)

struct Args {
  const real* X;          // [N][u0] inputs
  const real* Y;          // [N] targets
  const int* perm;        // [n_epochs][n_net][N] row order of each epoch and network
  const real* lr;         // [n_epochs * steps] bias-corrected Adam step size of each step
  real* W;                // [n_net][NP] parameters: W_0, b_0, W_1, b_1, ... (W_l row-major [u_l][u_{l+1}])
  real* M;                // [n_net][NP] Adam first moments
  real* V;                // [n_net][NP] Adam second moments
  real* part;             // [n_net][S][NP] partial gradients, one slot per (workgroup, chunk)
  real* lpart;            // [n_net][G] per-workgroup sums of squared errors of the current epoch
  real* best;             // [n_net] best epoch loss so far
  real* curve_loss;       // [max_iter][n_net] epoch losses
  int* istate;            // [n_net] epochs without improvement, [n_net] active flags, [1] error flag
  unsigned* bar;          // [n_net][2] barrier counter and generation
  unsigned char* curve_active;  // [max_iter][n_net] network active during the epoch
  long long timeout;      // barrier timeout in wall_clock64 ticks
  int n_net, G, N, B, n_epochs, epoch0, L, NP, R, RC, nch, S, n_iter_no_change, ASZ;
  int u[MAXL + 1], ow[MAXL], ob[MAXL], act[MAXL + 1];
  real b1, b2, omb1, omb2, eps, tol;
};

// out[r][j] = relu(b[j] + sum_k in[r][k] W[k][j]) for 4 * ng rows; each work item does one unit j of 4 rows.
__device__ __forceinline__ void fwd_relu(const real* __restrict__ W, const real* __restrict__ b,
                                         const real* __restrict__ in, real* __restrict__ out, int NI, int NO,
                                         int ng) {
  for (int it = threadIdx.x; it < NO * ng; it += blockDim.x) {
    const int j = it % NO, r0 = (it / NO) * 4;
    const real bj = b[j];
    real a0 = bj, a1 = bj, a2 = bj, a3 = bj;
    const real* x = in + r0 * NI;
#pragma unroll 4
    for (int k = 0; k < NI; ++k) {
      const real w = W[k * NO + j];
      a0 += x[k] * w;
      a1 += x[NI + k] * w;
      a2 += x[2 * NI + k] * w;
      a3 += x[3 * NI + k] * w;
    }
    real* o = out + r0 * NO + j;
    o[0] = a0 > real(0) ? a0 : real(0);
    o[NO] = a1 > real(0) ? a1 : real(0);
    o[2 * NO] = a2 > real(0) ? a2 : real(0);
    o[3 * NO] = a3 > real(0) ? a3 : real(0);
  }
}

// Output unit: residual res = out - y and d loss / d out = res / batch, both zero for padding rows (r >= rc).
__device__ __forceinline__ void out_layer(const real* __restrict__ W, const real* __restrict__ b,
                                          const real* __restrict__ h, const real* __restrict__ y,
                                          real* __restrict__ dz, real* __restrict__ res, int NI, int rc, int nr,
                                          real inv_bs) {
  for (int r = threadIdx.x; r < nr; r += blockDim.x) {
    real a = b[0];
    for (int k = 0; k < NI; ++k) a += h[r * NI + k] * W[k];
    const real d = r < rc ? a - y[r] : real(0);
    res[r] = d;
    dz[r] = d * inv_bs;
  }
}

// Partial weight and bias gradients of one layer over nr rows, written to global memory.
__device__ __forceinline__ void grad_w(const real* __restrict__ in, const real* __restrict__ dz,
                                       real* __restrict__ gW, real* __restrict__ gb, int NI, int NO, int nr) {
  const int KG = (NI + 3) / 4;
  for (int it = threadIdx.x; it < (KG + 1) * NO; it += blockDim.x) {
    const int j = it % NO, kg = it / NO;
    if (kg < KG) {
      const int k0 = kg * 4;
      real a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      for (int r = 0; r < nr; ++r) {
        const real d = dz[r * NO + j];
        const real* x = in + r * NI + k0;  // may read up to 3 past the row end: still inside the LDS activations
        a0 += x[0] * d;
        a1 += x[1] * d;
        a2 += x[2] * d;
        a3 += x[3] * d;
      }
      real* g = gW + k0 * NO + j;
      g[0] = a0;
      if (k0 + 1 < NI) g[NO] = a1;
      if (k0 + 2 < NI) g[2 * NO] = a2;
      if (k0 + 3 < NI) g[3 * NO] = a3;
    } else {
      real a = 0;
      for (int r = 0; r < nr; ++r) a += dz[r * NO + j];
      gb[j] = a;
    }
  }
}

// h (post-ReLU input activations of the layer, [rows][NI]) <- (dz W^T) * (h > 0), in place. The j loop starts at
// k % NO so that lanes with consecutive k read W at stride NO + 1 (no LDS bank conflicts).
__device__ __forceinline__ void back(const real* __restrict__ W, const real* __restrict__ dz, real* h, int NI,
                                     int NO, int ng) {
  for (int it = threadIdx.x; it < NI * ng; it += blockDim.x) {
    const int k = it % NI, r0 = (it / NI) * 4;
    real a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const real* d = dz + r0 * NO;
    int j = k % NO;
    for (int jj = 0; jj < NO; ++jj) {
      const real w = W[k * NO + j];
      a0 += d[j] * w;
      a1 += d[NO + j] * w;
      a2 += d[2 * NO + j] * w;
      a3 += d[3 * NO + j] * w;
      j = (j + 1 == NO) ? 0 : j + 1;
    }
    real* hp = h + r0 * NI + k;
    hp[0] = hp[0] > real(0) ? a0 : real(0);
    hp[NI] = hp[NI] > real(0) ? a1 : real(0);
    hp[2 * NI] = hp[2 * NI] > real(0) ? a2 : real(0);
    hp[3 * NI] = hp[3 * NI] > real(0) ? a3 : real(0);
  }
}

// Barrier over the G workgroups of one network (atomic counter + generation, agent-scope fences). Returns false when
// the error flag is set, either by an earlier failure or by this barrier timing out.
__device__ __forceinline__ bool net_barrier(unsigned* cnt, unsigned* gen, unsigned G, int* err, long long timeout,
                                            int* s_fail) {
  __threadfence();  // release this thread's global writes at agent scope
  __syncthreads();
  if (threadIdx.x == 0) {
    int fail = __hip_atomic_load(err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0;
    if (!fail) {
      const unsigned my = __hip_atomic_load(gen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      const unsigned arrived = __hip_atomic_fetch_add(cnt, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if (arrived == G - 1) {
        __hip_atomic_store(cnt, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_fetch_add(gen, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      } else {
        const uint64_t t0 = wall_clock64();
        while (__hip_atomic_load(gen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == my) {
          __builtin_amdgcn_s_sleep(1);
          if (__hip_atomic_load(err, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
            fail = 1;
            break;
          }
          if ((long long)(wall_clock64() - t0) > timeout) {
            __hip_atomic_store(err, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            fail = 1;
            break;
          }
        }
      }
    }
    *s_fail = fail;
    __threadfence();
  }
  __syncthreads();
  __threadfence();  // acquire for every wave (invalidates this CU's L1 before reading other workgroups' writes)
  return *s_fail == 0;
}

template <bool LDSW>
__global__ void __launch_bounds__(1024) train_kernel(Args a) {
  extern __shared__ real smem[];
  __shared__ int s_fail, s_active, s_noimp;
  __shared__ real s_best;
  real* sW = smem;                         // weights (LDSW only)
  real* sA = smem + (LDSW ? a.NP : 0);     // activations: layer l at act[l], [RC][u_l]
  real* sY = sA + a.ASZ;                   // targets of the chunk
  real* sR = sY + a.RC;                    // residuals of the chunk

  const int net = blockIdx.x / a.G, g = blockIdx.x % a.G, NP = a.NP, L = a.L;
  real* Wn = a.W + (size_t)net * NP;
  real* Mn = a.M + (size_t)net * NP;
  real* Vn = a.V + (size_t)net * NP;
  real* Pn = a.part + (size_t)net * a.S * NP;
  real* lp = a.lpart + (size_t)net * a.G;
  unsigned* cnt = a.bar + 2 * net;
  unsigned* gen = cnt + 1;
  int* err = a.istate + 2 * a.n_net;
  const real* Ws = [&]() -> const real* {
    if constexpr (LDSW) return sW;
    else return Wn;
  }();
  const int nb = (a.N + a.B - 1) / a.B;
  const int per = (NP + a.G - 1) / a.G, p0 = min(g * per, NP), p1 = min(p0 + per, NP);

  if (threadIdx.x == 0) {
    s_best = a.best[net];
    s_noimp = a.istate[net];
    s_active = a.istate[a.n_net + net];
  }
  __syncthreads();
  for (int e = 0; e < a.n_epochs; ++e) {
    if (!s_active) break;
    const int ep = a.epoch0 + e;
    if (g == 0 && threadIdx.x == 0) a.curve_active[(size_t)ep * a.n_net + net] = 1;
    real sse = 0;  // thread 0: this workgroup's squared errors in the epoch
    for (int b = 0; b < nb; ++b) {
      const int start = b * a.B, bs = min(a.B, a.N - start);
      const int* pr = a.perm + ((size_t)e * a.n_net + net) * a.N + start;
      const real inv_bs = real(1) / real(bs);
      if constexpr (LDSW) {
        for (int i = threadIdx.x; i < NP; i += blockDim.x) sW[i] = Wn[i];
      }
      for (int c = 0; c < a.nch; ++c) {
        const int lo = g * a.R + c * a.RC;
        const int hi = min(g * a.R + min((c + 1) * a.RC, a.R), bs);
        const int rc = max(0, hi - lo), ng = (rc + 3) / 4, nr = ng * 4;
        const int D0 = a.u[0];
        for (int i = threadIdx.x; i < nr * D0; i += blockDim.x) {
          const int r = i / D0, col = i - r * D0;
          sA[i] = r < rc ? a.X[(size_t)pr[lo + r] * D0 + col] : real(0);
        }
        for (int i = threadIdx.x; i < nr; i += blockDim.x) sY[i] = i < rc ? a.Y[pr[lo + i]] : real(0);
        __syncthreads();
        for (int l = 0; l < L - 1; ++l) {
          fwd_relu(Ws + a.ow[l], Ws + a.ob[l], sA + a.act[l], sA + a.act[l + 1], a.u[l], a.u[l + 1], ng);
          __syncthreads();
        }
        out_layer(Ws + a.ow[L - 1], Ws + a.ob[L - 1], sA + a.act[L - 1], sY, sA + a.act[L], sR, a.u[L - 1], rc,
                  nr, inv_bs);
        __syncthreads();
        if (threadIdx.x == 0)
          for (int r = 0; r < rc; ++r) sse += sR[r] * sR[r];
        real* P = Pn + (size_t)(g * a.nch + c) * NP;
        for (int l = L - 1; l >= 0; --l) {
          grad_w(sA + a.act[l], sA + a.act[l + 1], P + a.ow[l], P + a.ob[l], a.u[l], a.u[l + 1], nr);
          __syncthreads();
          if (l > 0) {
            back(Ws + a.ow[l], sA + a.act[l + 1], sA + a.act[l], a.u[l], a.u[l + 1], ng);
            __syncthreads();
          }
        }
      }
      const bool last = b == nb - 1;
      if (last && threadIdx.x == 0) lp[g] = sse;
      if (!net_barrier(cnt, gen, a.G, err, a.timeout, &s_fail)) return;
      if (last && threadIdx.x == 0) {
        // Early stopping as in scikit-learn; every workgroup of the network computes the same values.
        real s = 0;
        for (int i = 0; i < a.G; ++i) s += lp[i];
        const real loss = real(0.5) * s / real(a.N);
        s_noimp = loss > s_best - a.tol ? s_noimp + 1 : 0;
        s_best = fmin(s_best, loss);
        s_active = s_noimp <= a.n_iter_no_change;
        if (g == 0) a.curve_loss[(size_t)ep * a.n_net + net] = loss;
      }
      const real lr = a.lr[e * nb + b];
      for (int p = p0 + threadIdx.x; p < p1; p += blockDim.x) {
        real gs = 0;
        for (int s = 0; s < a.S; ++s) gs += Pn[(size_t)s * NP + p];
        real m = Mn[p] * a.b1;
        m = m + a.omb1 * gs;
        real v = Vn[p] * a.b2;
        v = v + a.omb2 * gs * gs;
        Mn[p] = m;
        Vn[p] = v;
        Wn[p] = Wn[p] - (m * lr) / (sqrt(v) + a.eps);
      }
      if (!net_barrier(cnt, gen, a.G, err, a.timeout, &s_fail)) return;
    }
  }
  if (g == 0 && threadIdx.x == 0) {
    a.best[net] = s_best;
    a.istate[net] = s_noimp;
    a.istate[a.n_net + net] = s_active;
  }
}

template <bool LDSW>
const void* kernel() {
  return (const void*)train_kernel<LDSW>;
}
}  // namespace

extern "C" {
int mk_real_size() { return (int)sizeof(real); }
int mk_max_layers() { return MAXL; }
const char* mk_error(int e) { return hipGetErrorString((hipError_t)e); }

// Co-resident workgroups per compute unit at (weights in LDS, threads, dynamic LDS bytes), the number of compute
// units and the wall clock rate (kHz) of the current device. Returns 0 or a HIP error code.
int mk_query(int ldsw, int threads, int lds, int* blocks_per_cu, int* n_cu, int* clock_khz) {
  int dev = 0;
  hipError_t e = hipGetDevice(&dev);
  if (e == hipSuccess) e = hipDeviceGetAttribute(n_cu, hipDeviceAttributeMultiprocessorCount, dev);
  if (e == hipSuccess) e = hipDeviceGetAttribute(clock_khz, hipDeviceAttributeWallClockRate, dev);
  const void* k = ldsw ? kernel<true>() : kernel<false>();
  (void)hipFuncSetAttribute(k, hipFuncAttributeMaxDynamicSharedMemorySize, lds);  // not needed on AMD; best effort
  if (e == hipSuccess) e = hipOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_cu, k, threads, lds);
  return (int)e;
}

// Launch on `stream` without waiting. ip: n_net, G, threads, lds, ldsw, N, B, n_epochs, epoch0, L, NP, R, RC, nch,
// S, n_iter_no_change, ASZ, u[L + 1], ow[L], ob[L], act[L + 1]. fp: beta_1, beta_2, epsilon, tol. ptr: X, Y, perm,
// lr, W, M, V, part, lpart, best, curve_loss, istate, bar, curve_active. Returns 0 or a HIP launch error code.
int mk_train(const int* ip, const double* fp, void* const* ptr, long long timeout, void* stream) {
  Args a;
  a.X = (const real*)ptr[0]; a.Y = (const real*)ptr[1]; a.perm = (const int*)ptr[2]; a.lr = (const real*)ptr[3];
  a.W = (real*)ptr[4]; a.M = (real*)ptr[5]; a.V = (real*)ptr[6]; a.part = (real*)ptr[7]; a.lpart = (real*)ptr[8];
  a.best = (real*)ptr[9]; a.curve_loss = (real*)ptr[10]; a.istate = (int*)ptr[11]; a.bar = (unsigned*)ptr[12];
  a.curve_active = (unsigned char*)ptr[13];
  a.timeout = timeout;
  a.n_net = ip[0]; a.G = ip[1];
  const int threads = ip[2], lds = ip[3], ldsw = ip[4];
  a.N = ip[5]; a.B = ip[6]; a.n_epochs = ip[7]; a.epoch0 = ip[8]; a.L = ip[9]; a.NP = ip[10]; a.R = ip[11];
  a.RC = ip[12]; a.nch = ip[13]; a.S = ip[14]; a.n_iter_no_change = ip[15]; a.ASZ = ip[16];
  if (a.L < 1 || a.L > MAXL) return (int)hipErrorInvalidValue;
  const int* q = ip + 17;
  for (int l = 0; l <= a.L; ++l) a.u[l] = *q++;
  for (int l = 0; l < a.L; ++l) a.ow[l] = *q++;
  for (int l = 0; l < a.L; ++l) a.ob[l] = *q++;
  for (int l = 0; l <= a.L; ++l) a.act[l] = *q++;
  a.b1 = (real)fp[0]; a.b2 = (real)fp[1]; a.omb1 = (real)(1.0 - fp[0]); a.omb2 = (real)(1.0 - fp[1]);
  a.eps = (real)fp[2]; a.tol = (real)fp[3];
  const void* k = ldsw ? kernel<true>() : kernel<false>();
  (void)hipFuncSetAttribute(k, hipFuncAttributeMaxDynamicSharedMemorySize, lds);
  void* args[] = {&a};
  return (int)hipLaunchCooperativeKernel(k, dim3(a.n_net * a.G), dim3(threads), args, (unsigned)lds,
                                         (hipStream_t)stream);
}
}
