// Fused membership kernels for the torch backend's draws and bound tests (DRAW_KERNEL). Built and launched by
// nautilus/_drawkernel.py, which compiles this file with hipcc once per (source, GPU architecture, network dtype,
// padded dimension DK_MD, second and third hidden widths DK_H2, DK_H3) and loads it with ctypes. All pointers are
// torch device tensors.
//
// One thread per point; the point (padded with zeros to DK_MD dimensions) lives in registers. Ellipsoids are stored
// as their centre c (DK_MD) and B_inv^T (DK_MD x DK_MD, zero-padded), so the transformed point is
// y_j = sum_i (x_i - c_i) B[i][j] and the point is inside if sum_j y_j^2 < 1, all in float64. A cube-ellipsoid
// mixture also needs its cube dimensions inside [0, 1) (bit mask over dimensions); a mixture without an ellipsoid
// has c = B = 0 and only the cube test, as in nautilus._torch.highest_bound.
//
// Networks: ReLU MLPs [d, H1, DK_H2, DK_H3, 1] in the network dtype (real). Per network the packed parameters are
// W0^T (H1 x DK_MD, zero-padded), b0 (H1), W1 (H1 x DK_H2), b1, W2 (DK_H2 x DK_H3), b2, W3 (DK_H3), b3. The first
// hidden layer is streamed one unit at a time into the second layer's accumulators, so only DK_MD + DK_H2 + DK_H3
// values per point are held. The input is ((y - mean) / scale) in float64, rounded to real, as in _torch._scores;
// a neural bound accepts a point if the mean network output exceeds its threshold (in real). Sums run in a fixed
// order (the torch path uses rocBLAS's), so scores agree with the torch path to rounding.
//
// dk_accept: one pass of NautilusBound.sample on drawn points x with uniform numbers u: n_bound = number of union
// members containing the point, keep = u > 1 - 1 / n_bound (and inside the unit cube if the union has one), then
// accept = keep and inside the outer ellipsoid and the networks of any neural bound. Points that need the networks
// are compacted per workgroup in LDS first, so that the network loop runs on full wavefronts.
//
// dk_highest: for each point the largest bound index in [first, first + J) whose outer union (and unit cube, if the
// bound has one) contains the point and one of whose neural bounds accepts it, else -1; bounds are tested from the
// last down and a wavefront stops as soon as all of its points have their answer.
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef DK_REAL
#define DK_REAL float
#endif
#ifndef DK_MD
#define DK_MD 32
#endif
#ifndef DK_H2
#define DK_H2 50
#endif
#ifndef DK_H3
#define DK_H3 20
#endif

namespace {
typedef DK_REAL real;
constexpr int MD = DK_MD, H2 = DK_H2, H3 = DK_H3, THREADS = 256;

struct Args {
  const double* x;      // [n][d] points
  const double* u;      // [n] uniform numbers (dk_accept)
  const double* mc;     // [M][MD] mixture ellipsoid centres
  const double* mB;     // [M][MD][MD] mixture ellipsoid B_inv^T
  const uint64_t* mcube;  // [M] cube dimensions of each mixture (bit mask)
  const double* nc;     // [K][MD] outer ellipsoid centres of the neural bounds
  const double* nB;     // [K][MD][MD] their B_inv^T
  const double* nmean;  // [K][MD] network input mean
  const double* nscale; // [K][MD] network input scale (padding: 1)
  const real* w;        // [K][n_net][NPW] packed network parameters
  const real* thr;      // [K] score thresholds
  const long long* mstart;  // [J + 1] first mixture of each bound (dk_highest)
  const long long* nstart;  // [J + 1] first neural bound of each bound (dk_highest)
  const unsigned char* bcube;  // [J] bound's union has the unit cube (dk_highest)
  unsigned char* keep;  // [n] (dk_accept)
  unsigned char* accept;  // [n] (dk_accept)
  long long* result;    // [n] (dk_highest)
  long long n, first;
  int d, M, K, H1, n_net, NPW, union_cube, has_net, J;
};

__device__ __forceinline__ void load_point(const Args& a, long long p, double (&x)[MD], uint64_t& outside) {
  outside = 0;
#pragma unroll
  for (int i = 0; i < MD; ++i) {
    x[i] = i < a.d ? a.x[p * a.d + i] : 0.0;
    if (i < a.d && (x[i] < 0.0 || x[i] >= 1.0)) outside |= (uint64_t)1 << i;
  }
}

// sum_j ((x - c) B)_j^2 < 1; the transformed point is written to y if Y.
template <bool Y>
__device__ __forceinline__ bool in_ellipsoid(const double (&x)[MD], const double* __restrict__ c,
                                             const double* __restrict__ B, double (&y)[MD]) {
  double dx[MD];
#pragma unroll
  for (int i = 0; i < MD; ++i) dx[i] = x[i] - c[i];
  double r = 0.0;
#pragma unroll
  for (int j = 0; j < MD; ++j) {
    double s = 0.0;
#pragma unroll
    for (int i = 0; i < MD; ++i) s += dx[i] * B[i * MD + j];
    if (Y) y[j] = s;
    r += s * s;
  }
  return r < 1.0;
}

__device__ __forceinline__ bool in_union(const Args& a, const double (&x)[MD], uint64_t outside, int m0, int m1) {
  double unused[MD];
  bool in = false;
  for (int m = m0; m < m1; ++m) {
    if (!in && (outside & a.mcube[m]) == 0)
      in = in_ellipsoid<false>(x, a.mc + (size_t)m * MD, a.mB + (size_t)m * MD * MD, unused);
  }
  return in;
}

// Mean network output of neural bound k for the transformed point y.
__device__ __forceinline__ real score(const Args& a, int k, const double (&y)[MD]) {
  real xn[MD];
  const double* mean = a.nmean + (size_t)k * MD;
  const double* scale = a.nscale + (size_t)k * MD;
#pragma unroll
  for (int i = 0; i < MD; ++i) xn[i] = (real)((y[i] - mean[i]) / scale[i]);
  const int H1 = a.H1;
  real total = 0;
  for (int net = 0; net < a.n_net; ++net) {
    const real* W0 = a.w + ((size_t)k * a.n_net + net) * a.NPW;
    const real* b0 = W0 + (size_t)H1 * MD;
    const real* W1 = b0 + H1;
    const real* b1 = W1 + (size_t)H1 * H2;
    const real* W2 = b1 + H2;
    const real* b2 = W2 + H2 * H3;
    const real* W3 = b2 + H3;
    const real* b3 = W3 + H3;
    real h2[H2];
#pragma unroll
    for (int o = 0; o < H2; ++o) h2[o] = 0;
    for (int j = 0; j < H1; ++j) {
      real s = 0;
#pragma unroll
      for (int i = 0; i < MD; ++i) s += xn[i] * W0[(size_t)j * MD + i];
      s += b0[j];
      const real r = s > real(0) ? s : real(0);
#pragma unroll
      for (int o = 0; o < H2; ++o) h2[o] += r * W1[(size_t)j * H2 + o];
    }
    real h3[H3];
#pragma unroll
    for (int q = 0; q < H3; ++q) h3[q] = 0;
#pragma unroll
    for (int o = 0; o < H2; ++o) {
      const real s = h2[o] + b1[o];
      const real r = s > real(0) ? s : real(0);
#pragma unroll
      for (int q = 0; q < H3; ++q) h3[q] += r * W2[o * H3 + q];
    }
    real out = 0;
#pragma unroll
    for (int q = 0; q < H3; ++q) {
      const real s = h3[q] + b2[q];
      out += (s > real(0) ? s : real(0)) * W3[q];
    }
    total += out + b3[0];
  }
  return total / (real)a.n_net;
}

// Whether neural bound k accepts the point (outer ellipsoid, then networks).
__device__ __forceinline__ bool neural(const Args& a, int k, const double (&x)[MD]) {
  double y[MD];
  if (!in_ellipsoid<true>(x, a.nc + (size_t)k * MD, a.nB + (size_t)k * MD * MD, y)) return false;
  return !a.has_net || score(a, k, y) > a.thr[k];
}

__global__ void __launch_bounds__(THREADS) accept_kernel(Args a) {
  __shared__ int list[THREADS];
  __shared__ uint64_t todo_list[THREADS];
  __shared__ int count;
  if (threadIdx.x == 0) count = 0;
  __syncthreads();
  const long long p = (long long)blockIdx.x * THREADS + threadIdx.x;
  bool kept = false;
  uint64_t todo = 0;
  if (p < a.n) {
    double x[MD], unused[MD];
    uint64_t outside;
    load_point(a, p, x, outside);
    int n_bound = 0;
    for (int m = 0; m < a.M; ++m)
      if ((outside & a.mcube[m]) == 0 &&
          in_ellipsoid<false>(x, a.mc + (size_t)m * MD, a.mB + (size_t)m * MD * MD, unused))
        ++n_bound;
    kept = a.u[p] > 1.0 - 1.0 / (double)n_bound;
    if (a.union_cube && outside != 0) kept = false;
    a.keep[p] = kept;
    if (kept)
      for (int k = 0; k < a.K; ++k)
        if (in_ellipsoid<false>(x, a.nc + (size_t)k * MD, a.nB + (size_t)k * MD * MD, unused))
          todo |= (uint64_t)1 << k;
    if (!a.has_net || a.K == 0 || todo == 0) a.accept[p] = a.K == 0 ? kept : (kept && todo != 0);
  }
  if (a.has_net && todo != 0) {
    const int slot = atomicAdd(&count, 1);
    list[slot] = threadIdx.x;
    todo_list[slot] = todo;
  }
  __syncthreads();
  if (!a.has_net || (int)threadIdx.x >= count) return;
  const long long q = (long long)blockIdx.x * THREADS + list[threadIdx.x];
  const uint64_t mask = todo_list[threadIdx.x];
  double x[MD];
  uint64_t outside;
  load_point(a, q, x, outside);
  bool acc = false;
  for (int k = 0; k < a.K; ++k) {
    if (!acc && ((mask >> k) & 1)) {
      double y[MD];
      in_ellipsoid<true>(x, a.nc + (size_t)k * MD, a.nB + (size_t)k * MD * MD, y);
      acc = score(a, k, y) > a.thr[k];
    }
  }
  a.accept[q] = acc;
}

__global__ void __launch_bounds__(THREADS) highest_kernel(Args a) {
  const long long p = (long long)blockIdx.x * THREADS + threadIdx.x;
  const bool live = p < a.n;
  double x[MD];
  uint64_t outside = 0;
  if (live) load_point(a, p, x, outside);
  long long result = -1;
  bool done = !live;
  for (int j = a.J - 1; j >= 0; --j) {
    if (!done) {
      bool in = in_union(a, x, outside, (int)a.mstart[j], (int)a.mstart[j + 1]);
      if (a.bcube[j] && outside != 0) in = false;
      if (in) {
        for (int k = (int)a.nstart[j]; k < (int)a.nstart[j + 1]; ++k) {
          if (!done && neural(a, k, x)) {
            done = true;
            result = a.first + j;
          }
        }
      }
    }
    if (__all(done)) break;
  }
  if (live) a.result[p] = result;
}

void unpack(Args& a, const long long* ip, void* const* ptr) {
  a.x = (const double*)ptr[0]; a.u = (const double*)ptr[1]; a.mc = (const double*)ptr[2];
  a.mB = (const double*)ptr[3]; a.mcube = (const uint64_t*)ptr[4]; a.nc = (const double*)ptr[5];
  a.nB = (const double*)ptr[6]; a.nmean = (const double*)ptr[7]; a.nscale = (const double*)ptr[8];
  a.w = (const real*)ptr[9]; a.thr = (const real*)ptr[10]; a.mstart = (const long long*)ptr[11];
  a.nstart = (const long long*)ptr[12]; a.bcube = (const unsigned char*)ptr[13];
  a.keep = (unsigned char*)ptr[14]; a.accept = (unsigned char*)ptr[15]; a.result = (long long*)ptr[16];
  a.n = ip[0]; a.d = (int)ip[1]; a.M = (int)ip[2]; a.K = (int)ip[3]; a.H1 = (int)ip[4]; a.n_net = (int)ip[5];
  a.NPW = (int)ip[6]; a.union_cube = (int)ip[7]; a.has_net = (int)ip[8]; a.J = (int)ip[9]; a.first = ip[10];
}
}  // namespace

extern "C" {
int dk_real_size() { return (int)sizeof(real); }
int dk_md() { return MD; }
int dk_h2() { return H2; }
int dk_h3() { return H3; }
const char* dk_error(int e) { return hipGetErrorString((hipError_t)e); }

// ip: n, d, M, K, H1, n_net, NPW, union_cube, has_net, J, first. ptr: x, u, mc, mB, mcube, nc, nB, nmean, nscale,
// w, thr, mstart, nstart, bcube, keep, accept, result (unused ones may be null). Launch on `stream` without
// waiting; returns 0 or a HIP error code.
int dk_accept(const long long* ip, void* const* ptr, void* stream) {
  Args a;
  unpack(a, ip, ptr);
  if (a.d > MD || a.K > 64) return (int)hipErrorInvalidValue;
  if (a.n == 0) return 0;
  hipLaunchKernelGGL(accept_kernel, dim3((unsigned)((a.n + THREADS - 1) / THREADS)), dim3(THREADS), 0,
                     (hipStream_t)stream, a);
  return (int)hipGetLastError();
}

int dk_highest(const long long* ip, void* const* ptr, void* stream) {
  Args a;
  unpack(a, ip, ptr);
  if (a.d > MD) return (int)hipErrorInvalidValue;
  if (a.n == 0) return 0;
  hipLaunchKernelGGL(highest_kernel, dim3((unsigned)((a.n + THREADS - 1) / THREADS)), dim3(THREADS), 0,
                     (hipStream_t)stream, a);
  return (int)hipGetLastError();
}
}
